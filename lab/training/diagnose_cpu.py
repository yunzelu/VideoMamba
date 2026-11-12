"""Locate startup delays without using a GPU or modifying the dataset.

Run this on a short CPU-only compute allocation. Each step prints a timestamp
before it begins, so the final line identifies the operation being attempted if
the job reaches its time limit. The input sample comes from the training split.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import sys
import time
import traceback


def message(text):
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {text}", flush=True)


def step(label, function):
    message(f"START {label}")
    started = time.monotonic()
    try:
        result = function()
    except Exception:
        message(f"FAILED {label} after {time.monotonic() - started:.2f} seconds")
        traceback.print_exc()
        raise
    message(f"DONE  {label} ({time.monotonic() - started:.2f} seconds)")
    return result


def load_trainer(path):
    spec = importlib.util.spec_from_file_location("canny_diagnostic_train", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load trainer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    message("CPU-only diagnostic starting; no model training or GPU requested")
    message(f"Python executable: {sys.executable}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/project/def-mbolic/yunzelu/_dataset/canny/dataset"))
    parser.add_argument("--weights", type=Path, default=Path("/project/def-mbolic/yunzelu/VideoMamba/lab/checkpoints/videomamba_t16_k400_f32_res224.pth"))
    parser.add_argument("--skip-weights", action="store_true", help="Only inspect dataset and preprocessing")
    args = parser.parse_args()
    trainer_path = Path(__file__).resolve().with_name("train_canny.py")
    message(f"Trainer: {trainer_path}")
    message(f"Dataset: {args.data}")
    message(f"Checkpoint: {args.weights}")

    np = step("import NumPy", lambda: __import__("numpy"))
    torch = step("import PyTorch", lambda: __import__("torch"))
    message(f"Versions: NumPy {np.__version__}; PyTorch {torch.__version__}; torch CUDA build {torch.version.cuda}")
    # Match CPU resources requested by the Slurm script and avoid oversubscription.
    torch.set_num_threads(2)
    trainer = step("import dataset reader and training module", lambda: load_trainer(trainer_path))
    datasets, subjects = step("read dataset configuration and validate all split manifests", lambda: trainer.validate_dataset(args.data, 32))
    message("Segment counts: " + ", ".join(f"{name}={len(dataset)}" for name, dataset in datasets.items()))
    message("Subject counts: " + ", ".join(f"{name}={len(names)}" for name, names in subjects.items()))

    video_dataset = trainer.CannyVideoDataset(datasets["train"], "imagenet")
    for index in (0, len(video_dataset) - 1):
        video, label, returned_index = step(
            f"load and preprocess training sample {index}", lambda index=index: video_dataset[index]
        )
        if tuple(video.shape) != (3, 32, 224, 224) or not torch.isfinite(video).all().item():
            raise ValueError(f"Unexpected or non-finite sample {index}: shape {tuple(video.shape)}")
        record = video_dataset.base.samples[returned_index]
        message(f"Sample {index}: recording={record['recording']}, label={record['label']}, tensor_shape={tuple(video.shape)}, dtype={video.dtype}")

    if sys.platform == "win32":
        message("Two-worker DataLoader check skipped on Windows; run it in the Linux Slurm job")
    else:
        def load_worker_batch():
            loader = trainer.make_loader(video_dataset, batch_size=1, workers=2, shuffle=False, seed=42)
            # Bounds waiting for a batch; the Slurm time limit also bounds setup and shutdown.
            loader.timeout = 60
            iterator = iter(loader)
            try:
                videos, labels, indices = next(iterator)
                if tuple(videos.shape) != (1, 3, 32, 224, 224):
                    raise ValueError(f"Unexpected batch shape: {tuple(videos.shape)}")
                message(f"Two-worker batch shape: {tuple(videos.shape)}; labels={labels.tolist()}; indices={indices.tolist()}")
            finally:
                del iterator

        step("load first batch with two DataLoader workers (CPU only, 60-second batch timeout)", load_worker_batch)
        message("CPU worker test does not reproduce worker behavior after GPU initialization")

    if args.skip_weights:
        message("Checkpoint inspection skipped by request")
    else:
        def inspect_checkpoint():
            state = torch.load(args.weights, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                raise ValueError("Checkpoint is not a state dictionary")
            for key in ("head.weight", "head.bias", "temporal_pos_embedding"):
                value = state.get(key)
                message(f"Checkpoint {key}: {None if value is None else tuple(value.shape)}")
            expected = {"head.weight": (400, 192), "head.bias": (400,), "temporal_pos_embedding": (1, 32, 192)}
            for key, shape in expected.items():
                value = state.get(key)
                if value is None or tuple(value.shape) != shape:
                    raise ValueError(f"Unexpected checkpoint tensor {key}; expected {shape}")
            return len(state)

        keys = step("load and inspect 32-frame pretrained checkpoint on CPU", inspect_checkpoint)
        message(f"Checkpoint tensor entries: {keys}")

    message("CPU diagnostic completed successfully")


if __name__ == "__main__":
    main()
