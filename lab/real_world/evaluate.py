"""Evaluate a trained lab VideoMamba on prepared real-world Canny clips.

The manifest is JSONL, with paths relative to the manifest directory. Each
NPZ stores 32 consecutive binary frames packed along the spatial dimension.
The loader presents those frames directly without temporal duplication.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB / "training"))
import train_canny as base  # noqa: E402

# The annotation does not say whether a person is lying on the bed or floor.
# The output classes are therefore scored after collapsing those two outputs.
COARSE_LABELS = ["stand-and-walk", "sit", "lie", "stand-to-sit", "sit-to-stand",
                 "sit-to-lie", "lie-to-sit", "lie-on-floor-to-stand", "fall"]
MODEL_TO_COARSE = [0, 1, 2, 2, 3, 4, 5, 6, 7, 8]


class PreparedClips:
    def __init__(self, manifest: Path):
        self.root = manifest.resolve().parent
        self.samples = []
        with manifest.open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"Manifest line {number} is not an object")
                label_id = item.get("adapter_label_id")
                if isinstance(label_id, bool) or not isinstance(label_id, int) or not 0 <= label_id < len(base.LABELS):
                    raise ValueError(f"Invalid adapter_label_id at manifest line {number}")
                if item.get("adapter_label") != base.LABELS[label_id]:
                    raise ValueError(f"Adapter label/label_id mismatch at manifest line {number}")
                coarse = COARSE_LABELS[MODEL_TO_COARSE[label_id]]
                if item.get("coarse_label", coarse) != coarse:
                    raise ValueError(f"Coarse label/label_id mismatch at manifest line {number}")
                relative = item.get("path")
                if not isinstance(relative, str) or not relative:
                    raise ValueError(f"Missing clip path at manifest line {number}")
                path = (self.root / relative).resolve()
                if not path.is_relative_to(self.root) or not path.is_file():
                    raise ValueError(f"Missing or unsafe clip path at manifest line {number}: {relative}")
                self.samples.append(item)
        if not self.samples:
            raise ValueError("Manifest has no eligible labeled clips")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        record = self.samples[index]
        with np.load(self.root / record["path"], allow_pickle=False) as archive:
            packed = archive["packed_frames"]
            height, width = int(archive["height"]), int(archive["width"])
        if height <= 0 or width <= 0 or packed.dtype != np.uint8 or packed.shape != (32, (height * width + 7) // 8):
            raise ValueError(f"Invalid packed frames in {record['path']}: {packed.dtype} {packed.shape}, {height}x{width}; expected 32 source frames. Re-run preparation if this is an older 11-frame clip.")
        frames = np.unpackbits(packed, axis=1, count=height * width, bitorder="little")
        frames = frames.reshape(32, height, width) * np.uint8(255)
        return {"frames": frames, "metadata": {**record, "label_id": record["adapter_label_id"]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this script on a GPU compute node.")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("predictions.csv", "metrics.json"):
        if (args.output / name).exists():
            raise FileExistsError(f"Refusing to overwrite {args.output / name}; use a new output directory")

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    provenance = saved["provenance"]
    required = {"model": "VideoMamba-Tiny", "frames": 32, "image_size": 224,
                "labels": base.LABELS, "normalization": "imagenet",
                "resize": "antialiased_bilinear_letterbox_black"}
    for key, expected in required.items():
        if provenance.get(key) != expected:
            raise ValueError(f"Incompatible checkpoint {key}: {provenance.get(key)!r}; expected {expected!r}")
    source = args.repo / "videomamba/video_sm/models/videomamba.py"
    if base.sha256_file(source) != provenance.get("model_source_sha256"):
        raise ValueError("Model source differs from the source used during training")
    precision = provenance.get("precision")
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"Unsupported checkpoint precision: {precision!r}")

    clips = PreparedClips(args.manifest)
    dataset = base.CannyVideoDataset(clips, normalization=provenance["normalization"])
    loader = base.make_loader(dataset, args.batch_size, args.workers, False, provenance["seed"])
    model = base.create_model(args.repo, 0, 32).cuda()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    device = torch.device("cuda")
    confusion = torch.zeros((len(COARSE_LABELS), len(COARSE_LABELS)), dtype=torch.int64)
    rows = []
    print(f"Evaluating {len(clips)} prepared clips", flush=True)
    with torch.inference_mode():
        for frames, target, indices in loader:
            with base.amp_context(device, precision):
                logits = model(frames.to(device, non_blocking=True))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Evaluation produced non-finite logits")
            probs = logits.float().softmax(dim=1).cpu()
            predicted = probs.argmax(dim=1)
            mapping = torch.tensor(MODEL_TO_COARSE, dtype=torch.long)
            target_coarse, predicted_coarse = mapping[target], mapping[predicted]
            counts = torch.bincount(target_coarse * len(COARSE_LABELS) + predicted_coarse,
                                    minlength=len(COARSE_LABELS) ** 2)
            confusion += counts.reshape(len(COARSE_LABELS), len(COARSE_LABELS))
            for i, pred, prob in zip(indices.tolist(), predicted.tolist(), probs.tolist()):
                row = dict(clips.samples[i])
                # The preparation label_id is an adapter for the training
                # Dataset, not a claim about the location of a lying person.
                target_id = int(row.pop("adapter_label_id"))
                row.pop("adapter_label", None)
                row.pop("coarse_label", None)
                row["prediction"] = base.LABELS[pred]
                row["ground_truth"] = COARSE_LABELS[MODEL_TO_COARSE[target_id]]
                row["coarse_prediction"] = COARSE_LABELS[MODEL_TO_COARSE[pred]]
                row.update({f"p_{label}": float(prob[j]) for j, label in enumerate(base.LABELS)})
                rows.append(row)
            if len(rows) % 100 == 0 or len(rows) == len(clips):
                print(f"Evaluated {len(rows)}/{len(clips)}", flush=True)

    matrix = confusion.to(torch.float64)
    support, predicted_counts, correct = matrix.sum(1), matrix.sum(0), matrix.diag()
    precision_per_class = torch.where(predicted_counts > 0, correct / predicted_counts.clamp(min=1), 0.0)
    recall = torch.where(support > 0, correct / support.clamp(min=1), 0.0)
    f1 = torch.where(precision_per_class + recall > 0,
                     2 * precision_per_class * recall / (precision_per_class + recall).clamp(min=1e-12), 0.0)
    supported = support > 0
    metrics = {"accuracy": float(correct.sum().item() / support.sum().item()),
               "macro_f1_supported": float(f1[supported].mean().item()),
               "macro_f1_fixed_9": float(f1.mean().item()),
               "per_class": {name: {"support": int(support[i].item()),
                                    "precision": float(precision_per_class[i].item()),
                                    "recall": float(recall[i].item()), "f1": float(f1[i].item())}
                             for i, name in enumerate(COARSE_LABELS)},
               "confusion_matrix_true_by_predicted": confusion.tolist()}
    metrics.update({"samples": len(rows), "labels": COARSE_LABELS,
                    "model_labels": base.LABELS, "epoch": saved.get("epoch"),
                    "checkpoint_sha256": base.sha256_file(args.checkpoint),
                    "manifest_sha256": base.sha256_file(args.manifest),
                    "precision": precision,
                    "note": "Clip-level coarse metrics. Both lie outputs map to lie. Transition outputs map to sit-to-lie and lie-to-sit. Supported macro-F1 averages over classes present in the ground truth; fixed macro-F1 averages over all nine coarse classes. Raw ten-class predictions and probabilities are in predictions.csv."})
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    with (args.output / "metrics.json").open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: metrics[key] for key in ("samples", "accuracy", "macro_f1_supported", "macro_f1_fixed_9")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
