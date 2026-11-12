"""Train and evaluate VideoMamba-Tiny on activity-aligned Canny clips.

The VideoMamba repository and data remain external to this script.  See the
project documentation for the recommended setup and checkpoint provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("benchmark", "train"):
    print("[startup] Python started; importing NumPy and PyTorch", flush=True)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in ("benchmark", "train"):
    print("[startup] NumPy and PyTorch imported; loading dataset reader", flush=True)

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from canny_dataset import CannyDataset  # noqa: E402

LABELS = [
    "stand-and-walk", "sit", "lie-in-bed", "lie-on-floor", "stand-to-sit",
    "sit-to-stand", "sit-to-lie-in-bed", "lie-in-bed-to-sit",
    "lie-on-floor-to-stand", "fall",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")
    os.replace(tmp, path)


def atomic_save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def validate_dataset(root, frames=32):
    if frames not in (16, 32):
        raise ValueError("frames must be 16 or 32")
    root = Path(root)
    config = json.loads((root / "dataset_config.json").read_text(encoding="utf-8"))
    if config.get("labels") != LABELS:
        raise ValueError("Dataset label order differs from the ten expected classes")
    datasets = {split: CannyDataset(root, split=split, clip_len=frames) for split in ("train", "val", "test")}
    subjects = {}
    for split, dataset in datasets.items():
        if not dataset.samples:
            raise ValueError(f"Split {split} has no samples")
        subjects[split] = {r["subject"] for r in dataset.samples}
        for record in dataset.samples:
            label_id = record["label_id"]
            if label_id >= len(LABELS) or LABELS[label_id] != record["label"]:
                raise ValueError(f"Invalid label in {record['path']}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if subjects[a] & subjects[b]:
            raise ValueError(f"Subjects overlap in {a} and {b}: {subjects[a] & subjects[b]}")
    return datasets, {name: sorted(value) for name, value in subjects.items()}


class CannyVideoDataset(Dataset):
    """Use the base dataset's temporal sampling and letterbox the full frame."""

    def __init__(self, base, normalization="imagenet", size=224):
        self.base = base
        self.normalization = normalization
        self.size = size
        if normalization not in ("imagenet", "zero_one"):
            raise ValueError("Unknown normalization")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        # T,1,H,W, values in [0,1]. Bilinear antialiased resize preserves the
        # whole scene; edge intensities become grayscale, not binary.
        frames = torch.from_numpy(sample["frames"]).unsqueeze(1).float().div_(255)
        _, _, height, width = frames.shape
        scale = min(self.size / height, self.size / width)
        new_h = max(1, min(self.size, round(height * scale)))
        new_w = max(1, min(self.size, round(width * scale)))
        frames = F.interpolate(frames, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
        pad_h, pad_w = self.size - new_h, self.size - new_w
        frames = F.pad(frames, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=0.0)
        video = frames.permute(1, 0, 2, 3).repeat(3, 1, 1, 1)
        if self.normalization == "imagenet":
            mean = video.new_tensor(IMAGENET_MEAN).view(3, 1, 1, 1)
            std = video.new_tensor(IMAGENET_STD).view(3, 1, 1, 1)
            video = (video - mean) / std
        return video, int(sample["metadata"]["label_id"]), index


def load_factory(repo):
    source = Path(repo) / "videomamba/video_sm/models/videomamba.py"
    if not source.is_file():
        raise FileNotFoundError(f"VideoMamba model source not found: {source}")
    spec = importlib.util.spec_from_file_location("canny_videomamba_model", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model source {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.videomamba_tiny


def create_model(repo, checkpoint_blocks, frames=32, progress=None):
    if not 0 <= checkpoint_blocks <= 24:
        raise ValueError("checkpoint-blocks must be between 0 and 24")
    if frames not in (16, 32):
        raise ValueError("frames must be 16 or 32")
    if progress:
        progress("Importing VideoMamba model code")
    factory = load_factory(repo)
    if progress:
        progress("Model code imported; constructing VideoMamba-Tiny")
    return factory(pretrained=False, img_size=224, num_classes=len(LABELS),
                   num_frames=frames, kernel_size=1, use_checkpoint=checkpoint_blocks > 0,
                   checkpoint_num=checkpoint_blocks)


def load_k400_checkpoint(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise ValueError("Expected the official K400 checkpoint's direct state dict")
    state = dict(state)
    key = "temporal_pos_embedding"
    expected = model.state_dict().get(key)
    actual = state.get(key)
    if expected is None or actual is None or tuple(actual.shape) != tuple(expected.shape):
        got = None if actual is None else tuple(actual.shape)
        wanted = None if expected is None else tuple(expected.shape)
        raise ValueError(f"Checkpoint temporal embedding has shape {got}, expected {wanted}; use a checkpoint matching --frames")
    head_weight = state.pop("head.weight", None)
    head_bias = state.pop("head.bias", None)
    if head_weight is None or head_bias is None or tuple(head_weight.shape) != (400, 192) or tuple(head_bias.shape) != (400,):
        raise ValueError("Expected a 400-class VideoMamba-Tiny checkpoint head")
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != {"head.weight", "head.bias"} or result.unexpected_keys:
        raise ValueError(f"Incompatible pretrained backbone: missing={result.missing_keys}; unexpected={result.unexpected_keys}")


def metrics_from_confusion(matrix):
    matrix = matrix.to(torch.float64)
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)
    correct = matrix.diag()
    precision = torch.where(predicted > 0, correct / predicted.clamp(min=1), 0.0)
    recall = torch.where(support > 0, correct / support.clamp(min=1), 0.0)
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall).clamp(min=1e-12), 0.0)
    total = int(support.sum().item())
    return {"accuracy": float(correct.sum().item() / total) if total else 0.0,
            "macro_f1_fixed_10": float(f1.mean().item()),
            "per_class": {name: {"support": int(support[i].item()), "precision": float(precision[i].item()),
                                 "recall": float(recall[i].item()), "f1": float(f1[i].item())}
                          for i, name in enumerate(LABELS)},
            "confusion_matrix_true_by_predicted": matrix.to(torch.int64).tolist()}


def add_confusion(matrix, target, prediction):
    counts = torch.bincount((target * len(LABELS) + prediction).cpu(), minlength=len(LABELS) ** 2)
    matrix += counts.reshape(len(LABELS), len(LABELS))


def seed_epoch(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, workers, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle,
                      pin_memory=torch.cuda.is_available(), generator=generator, drop_last=False)


def amp_context(device, precision):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16")


def run_evaluation(model, loader, device, precision, collect_predictions=False):
    model.eval()
    matrix = torch.zeros((len(LABELS), len(LABELS)), dtype=torch.int64)
    loss_sum, seen, predictions = 0.0, 0, []
    with torch.no_grad():
        for frames, target, indices in loader:
            frames, target = frames.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with amp_context(device, precision):
                logits = model(frames)
                loss = F.cross_entropy(logits.float(), target, reduction="sum")
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("Evaluation produced non-finite logits or loss")
            predicted = logits.argmax(dim=1)
            add_confusion(matrix, target, predicted)
            loss_sum += float(loss.item())
            seen += len(target)
            if collect_predictions:
                probs = logits.float().softmax(dim=1).cpu().tolist()
                for idx, label, pred, prob in zip(indices.tolist(), target.cpu().tolist(), predicted.cpu().tolist(), probs):
                    record = loader.dataset.base.samples[idx]
                    predictions.append({"path": record["path"], "recording": record["recording"],
                                        "subject": record["subject"], "start_frame": record["start_frame"],
                                        "end_frame": record["end_frame"], "target": LABELS[label],
                                        "prediction": LABELS[pred], **{f"p_{name}": prob[i] for i, name in enumerate(LABELS)}})
    metrics = metrics_from_confusion(matrix)
    metrics["loss"] = loss_sum / seen
    metrics["samples"] = seen
    return metrics, predictions


def write_predictions(path, rows):
    if not rows:
        raise ValueError("No predictions")
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_partial_model(path, model, provenance, epoch, step, samples):
    """Save an evaluable model snapshot, not an exact-resume checkpoint."""
    atomic_save(path, {"model": model.state_dict(), "provenance": provenance,
                       "epoch": epoch, "epoch_complete": False, "step": step,
                       "samples_seen_in_epoch": samples, "resumable": False})


def train(args):
    startup = time.perf_counter()

    def progress(message):
        print(f"[train +{time.perf_counter() - startup:.1f}s] {message}", flush=True)

    progress("Starting training; checking arguments and CUDA device")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid epochs, batch size, workers, learning rate, or weight decay")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an allocated CUDA device")
    device = torch.device("cuda")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not report BF16 support; use --precision fp32")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.resume and Path(args.resume).resolve() != (output / "last.pt").resolve():
        raise ValueError("Resume must use last.pt from the same output directory")
    if (output / "last.pt").exists() and not args.resume:
        raise FileExistsError("Run already contains last.pt. Use --resume or select a new output directory.")
    progress("CUDA checks passed; validating dataset")
    datasets, subjects = validate_dataset(args.data, args.frames)
    progress(f"Dataset validated; {len(datasets['train'])} training clips and {len(datasets['val'])} validation clips")
    progress("Calculating dataset, model source, and checkpoint hashes")
    manifest_hash = sha256_file(Path(args.data) / "manifest.jsonl")
    provenance = {"manifest_sha256": manifest_hash, "subjects": subjects, "labels": LABELS,
                  "model": "VideoMamba-Tiny", "frames": args.frames, "image_size": 224,
                  "model_source_sha256": sha256_file(Path(args.repo) / "videomamba/video_sm/models/videomamba.py"),
                  "resize": "antialiased_bilinear_letterbox_black", "normalization": args.normalization,
                  "precision": args.precision, "checkpoint_blocks": args.checkpoint_blocks,
                  "batch_size": args.batch_size, "seed": args.seed, "lr": args.lr,
                  "weight_decay": args.weight_decay, "epochs": args.epochs,
                  "pretrained_sha256": sha256_file(args.pretrained) if args.pretrained else None,
                  "torch_version": str(torch.__version__), "torch_cuda": torch.version.cuda}
    seed_epoch(args.seed)
    progress("Provenance recorded; preparing VideoMamba-Tiny")
    model = create_model(args.repo, args.checkpoint_blocks, args.frames, progress=progress)
    progress("Model constructed")
    if args.pretrained:
        progress("Loading pretrained checkpoint")
        load_k400_checkpoint(model, args.pretrained)
        progress("Pretrained checkpoint loaded")
    progress("Moving model to GPU and creating optimizer")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start, best = 0, -1.0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["provenance"] != provenance:
            raise ValueError("Resume configuration/data/checkpoint differs from the saved run")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, best = saved["epoch"], saved["best_val_macro_f1"]
    write_json(output / "run_config.json", provenance)
    train_set = CannyVideoDataset(datasets["train"], args.normalization)
    val_set = CannyVideoDataset(datasets["val"], args.normalization)
    val_loader = make_loader(val_set, args.batch_size, args.workers, False, args.seed)
    log_path = output / "epochs.jsonl"
    progress(f"Training setup complete; saving partial model after the first step and approximately every 5 minutes to {output / 'partial.pt'}")
    for epoch in range(start, args.epochs):
        seed_epoch(args.seed + epoch)
        train_loader = make_loader(train_set, args.batch_size, args.workers, True, args.seed + epoch)
        model.train()
        matrix = torch.zeros((len(LABELS), len(LABELS)), dtype=torch.int64)
        total_loss, seen = 0.0, 0
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        last_partial_save = started
        progress(f"Epoch {epoch + 1}: starting DataLoader; {len(train_loader)} batches")
        for step, (frames, target, _) in enumerate(train_loader, start=1):
            if step == 1:
                progress(f"First batch loaded, input shape={tuple(frames.shape)}; running forward and backward")
            frames, target = frames.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if step == 1:
                progress("Running first training forward pass")
            with amp_context(device, args.precision):
                logits = model(frames)
                loss = F.cross_entropy(logits.float(), target)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite logits/loss at epoch {epoch + 1}, step {step}")
            if step == 1:
                progress("First forward pass complete; running backward pass")
            loss.backward()
            if step == 1:
                progress("First backward pass complete; running optimizer step")
            optimizer.step()
            add_confusion(matrix, target, logits.argmax(dim=1))
            total_loss += float(loss.item()) * len(target)
            seen += len(target)
            now = time.perf_counter()
            if step == 1 or now - last_partial_save >= 300:
                progress(f"Saving partial model at epoch {epoch + 1}, step {step}; this snapshot is for evaluation, not exact resume")
                save_partial_model(output / "partial.pt", model, provenance, epoch + 1, step, seen)
                last_partial_save = time.perf_counter()
                progress("Partial model saved")
            if step % args.log_interval == 0:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                print(f"epoch={epoch + 1} step={step}/{len(train_loader)} samples={seen} clips_per_second={seen / elapsed:.2f}", flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        train_metrics = metrics_from_confusion(matrix)
        train_metrics.update(loss=total_loss / seen, samples=seen, seconds=elapsed, clips_per_second=seen / elapsed,
                             peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3)
        progress(f"Epoch {epoch + 1} training complete; starting validation")
        val_metrics, _ = run_evaluation(model, val_loader, device, args.precision)
        scheduler.step()
        is_best = val_metrics["macro_f1_fixed_10"] > best
        if is_best:
            best = val_metrics["macro_f1_fixed_10"]
        payload = {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "best_val_macro_f1": best, "provenance": provenance}
        atomic_save(output / "last.pt", payload)
        if is_best:
            atomic_save(output / "best.pt", payload)
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics,
                  "best_val_macro_f1": best}
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"epoch={epoch + 1} val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1_fixed_10']:.4f} "
              f"train_clips_per_second={train_metrics['clips_per_second']:.2f} best={best:.4f}", flush=True)


def evaluate(args):
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("Invalid batch size or workers")
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires an allocated CUDA device")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    provenance = checkpoint["provenance"]
    frames = provenance["frames"]
    if frames not in (16, 32):
        raise ValueError("Checkpoint has an unsupported frame count")
    if args.frames is not None and args.frames != frames:
        raise ValueError("Evaluation --frames differs from the trained checkpoint")
    if provenance["model_source_sha256"] != sha256_file(Path(args.repo) / "videomamba/video_sm/models/videomamba.py"):
        raise ValueError("Evaluation model source differs from the training model source")
    if provenance["manifest_sha256"] != sha256_file(Path(args.data) / "manifest.jsonl"):
        raise ValueError("Evaluation dataset manifest differs from the training manifest")
    datasets, subjects = validate_dataset(args.data, frames)
    if subjects != provenance["subjects"] or provenance["labels"] != LABELS:
        raise ValueError("Evaluation dataset split/labels differ from the training dataset")
    model = create_model(args.repo, 0, frames).cuda()
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = CannyVideoDataset(datasets[args.split], provenance["normalization"])
    loader = make_loader(dataset, args.batch_size, args.workers, False, provenance["seed"])
    metrics, predictions = run_evaluation(model, loader, torch.device("cuda"), provenance["precision"], True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics.update(split=args.split, epoch=checkpoint["epoch"], checkpoint_sha256=sha256_file(args.checkpoint),
                   manifest_sha256=provenance["manifest_sha256"], labels=LABELS,
                   epoch_complete=checkpoint.get("epoch_complete", True))
    if not metrics["epoch_complete"]:
        metrics["step"] = checkpoint["step"]
    write_json(output / f"{args.split}_metrics.json", metrics)
    write_predictions(output / f"{args.split}_predictions.csv", predictions)
    print(json.dumps({"split": args.split, "samples": metrics["samples"], "accuracy": metrics["accuracy"],
                      "macro_f1_fixed_10": metrics["macro_f1_fixed_10"]}, indent=2), flush=True)


def benchmark(args):
    """Time actual data loading and optimizer steps without saving a model."""
    startup = time.perf_counter()

    def progress(message):
        print(f"[benchmark +{time.perf_counter() - startup:.1f}s] {message}", flush=True)

    progress("Starting benchmark; checking arguments and CUDA device")
    if args.batch_size < 1 or args.workers < 0 or args.warmup_steps < 0 or args.steps < 1 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid benchmark arguments")
    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark requires an allocated CUDA device")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not report BF16 support; use --precision fp32")
    progress(f"CUDA checks passed; reading dataset manifest from {args.data}")
    datasets, _ = validate_dataset(args.data, args.frames)
    progress(f"Manifest validated; {len(datasets['train'])} training clips")
    dataset = CannyVideoDataset(datasets["train"], args.normalization)
    loader = make_loader(dataset, args.batch_size, args.workers, True, args.seed)
    progress(f"DataLoader configured; batch size={args.batch_size}, workers={args.workers}, frames={args.frames}")
    if args.warmup_steps + args.steps > len(loader):
        raise ValueError("Requested benchmark steps exceed one training epoch")
    seed_epoch(args.seed)
    model = create_model(args.repo, args.checkpoint_blocks, args.frames, progress=progress)
    progress("Model constructed")
    if args.pretrained:
        progress(f"Loading pretrained checkpoint: {args.pretrained}")
        load_k400_checkpoint(model, args.pretrained)
        progress("Pretrained checkpoint loaded")
    progress("Moving model to GPU and creating optimizer")
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    progress("Model and optimizer ready; starting DataLoader iterator")
    device = torch.device("cuda")
    iterator = iter(loader)
    progress("DataLoader iterator ready")
    matrix = torch.zeros((len(LABELS), len(LABELS)), dtype=torch.int64)

    def step(trace=False, sync_trace=False):
        if trace:
            progress("Fetching first batch from DataLoader")
        frames, target, _ = next(iterator)
        if trace:
            progress(f"First batch loaded; input shape={tuple(frames.shape)}; copying to GPU")
        frames, target = frames.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if trace:
            progress("Running first forward pass")
        with amp_context(device, args.precision):
            logits = model(frames)
            loss = F.cross_entropy(logits.float(), target)
        if trace and sync_trace:
            torch.cuda.synchronize()
        if trace:
            progress("First forward pass complete; checking loss and logits")
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise RuntimeError("Benchmark produced non-finite logits or loss")
        if trace:
            progress("Running first backward pass")
        loss.backward()
        if trace and sync_trace:
            torch.cuda.synchronize()
        if trace:
            progress("First backward pass complete; running optimizer step")
        optimizer.step()
        if trace and sync_trace:
            torch.cuda.synchronize()
        if trace:
            progress("First optimizer step complete; collecting metrics")
        # Match the training loop's metric bookkeeping and synchronization.
        add_confusion(matrix, target, logits.argmax(dim=1))
        loss.item()
        return len(target)

    for i in range(args.warmup_steps):
        step(trace=i == 0, sync_trace=i == 0)
        print(f"warmup={i + 1}/{args.warmup_steps}", flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    count = 0
    for i in range(args.steps):
        count += step(trace=args.warmup_steps == 0 and i == 0)
        if (i + 1) % 10 == 0 or i + 1 == args.steps:
            print(f"measured_steps={i + 1}/{args.steps}", flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    print(json.dumps({"measured_steps": args.steps, "measured_samples": count, "seconds": seconds,
                      "clips_per_second": count / seconds, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                      "batch_size": args.batch_size, "frames": args.frames, "checkpoint_blocks": args.checkpoint_blocks,
                      "precision": args.precision, "workers": args.workers,
                      "approx_train_epoch_minutes": len(dataset) / (count / seconds) / 60}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--repo", type=Path, required=True)
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    initialization = train_parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--pretrained", type=Path)
    initialization.add_argument("--random-init", action="store_true")
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--workers", type=int, default=2)
    train_parser.add_argument("--frames", type=int, choices=(16, 32), default=32)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.05)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--checkpoint-blocks", type=int, default=24)
    train_parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    train_parser.add_argument("--normalization", choices=("imagenet", "zero_one"), default="imagenet")
    train_parser.add_argument("--log-interval", type=int, default=25)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--repo", type=Path, required=True)
    evaluate_parser.add_argument("--data", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=("val", "test"), required=True)
    evaluate_parser.add_argument("--batch-size", type=int, default=2)
    evaluate_parser.add_argument("--workers", type=int, default=2)
    evaluate_parser.add_argument("--frames", type=int, choices=(16, 32), default=None,
                                 help="Optional checkpoint frame-count assertion; by default use the stored value")
    benchmark_parser = sub.add_parser("benchmark")
    benchmark_parser.add_argument("--repo", type=Path, required=True)
    benchmark_parser.add_argument("--data", type=Path, required=True)
    benchmark_init = benchmark_parser.add_mutually_exclusive_group(required=True)
    benchmark_init.add_argument("--pretrained", type=Path)
    benchmark_init.add_argument("--random-init", action="store_true")
    benchmark_parser.add_argument("--batch-size", type=int, default=2)
    benchmark_parser.add_argument("--workers", type=int, default=2)
    benchmark_parser.add_argument("--frames", type=int, choices=(16, 32), default=32)
    benchmark_parser.add_argument("--warmup-steps", type=int, default=5)
    benchmark_parser.add_argument("--steps", type=int, default=30)
    benchmark_parser.add_argument("--lr", type=float, default=1e-4)
    benchmark_parser.add_argument("--weight-decay", type=float, default=0.05)
    benchmark_parser.add_argument("--seed", type=int, default=42)
    benchmark_parser.add_argument("--checkpoint-blocks", type=int, default=24)
    benchmark_parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    benchmark_parser.add_argument("--normalization", choices=("imagenet", "zero_one"), default="imagenet")
    args = parser.parse_args()
    if args.command == "train" and args.log_interval < 1:
        raise ValueError("log-interval must be positive")
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
