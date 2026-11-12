"""Portable NumPy reader for activity-aligned Canny clips.

Each sample is an uncompressed NPY uint8 array. Its first dimension is time and
its second dimension contains flattened, little-bitorder packed binary images.
Memory mapping reads only the selected frame rows. The manifest stores paths relative to its
dataset root. Frames returned by this reader are uint8 binary images with
values 0 and 255, in (time, height, width) order.

This module intentionally has no PyTorch or OpenCV dependency. A future training
adapter can convert the result to the input layout expected by its model.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import numpy as np


def monotonic_uniform_indices(num_frames: int, clip_len: int = 16) -> np.ndarray:
    """Select nearest frames uniformly, including both endpoints when possible.

    This matches the local skeleton pipeline's MonotonicUniformResample,
    including monotonic repetition for clips shorter than ``clip_len`` and
    returning index 0 for a single requested output frame.
    """
    num_frames = int(num_frames)
    clip_len = int(clip_len)
    if num_frames <= 0:
        raise ValueError("Cannot resample an empty segment")
    if clip_len <= 0:
        raise ValueError("clip_len must be positive")
    if clip_len == 1:
        return np.zeros(1, dtype=np.int64)
    if num_frames == 1:
        return np.zeros(clip_len, dtype=np.int64)
    positions = np.arange(clip_len, dtype=np.float64) * float(num_frames - 1) / float(clip_len - 1)
    indices = np.floor(positions + 0.5).astype(np.int64)
    indices[0] = 0
    indices[-1] = num_frames - 1
    return np.maximum.accumulate(indices)


class CannyDataset:
    """Read a generated dataset and return uniformly sampled Canny sequences.

    Args:
        root: Dataset directory containing ``manifest.jsonl``.
        split: Optional manifest split name, typically train, val, or test.
        clip_len: Number of temporally ordered frames returned per sample.

    Indexing returns a dictionary with ``frames`` (T,H,W uint8 values 0/255),
    ``frame_indices`` (indices relative to the segment),
    ``source_frame_indices`` (zero-based indices in the source video), and
    ``metadata`` (the complete manifest record). Resize and channel replication
    are intentionally left to the training transform.
    """

    def __init__(self, root: str | Path, split: str | None = None, clip_len: int = 16):
        self.root = Path(root).resolve()
        self.clip_len = int(clip_len)
        if self.clip_len <= 0:
            raise ValueError("clip_len must be positive")
        self.split = split
        self.samples: list[dict] = []
        manifest = self.root / "manifest.jsonl"
        with manifest.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {manifest} at line {line_no}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object at manifest line {line_no}")
                self._validate_record(record, line_no)
                if split is not None and record["split"] != split:
                    continue
                self.samples.append(record)

    def _validate_record(self, record: dict, line_no: int) -> None:
        required = {"path", "subject", "label", "label_id", "num_frames", "height", "width", "start_frame", "end_frame", "recording", "split"}
        missing = required.difference(record)
        if missing:
            raise ValueError(f"Manifest line {line_no} is missing: {sorted(missing)}")
        for name in ("num_frames", "height", "width"):
            if not isinstance(record[name], int) or isinstance(record[name], bool) or record[name] <= 0:
                raise ValueError(f"Invalid {name} at manifest line {line_no}")
        for name in ("start_frame", "end_frame", "label_id"):
            if not isinstance(record[name], int) or isinstance(record[name], bool) or record[name] < 0:
                raise ValueError(f"Invalid {name} at manifest line {line_no}")
        if record["end_frame"] - record["start_frame"] != record["num_frames"]:
            raise ValueError(f"Segment length disagrees with frame bounds at manifest line {line_no}")
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or "\\" in record["path"]:
            raise ValueError(f"Invalid dataset-relative path at manifest line {line_no}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        record = self.samples[index]
        relative = PurePosixPath(record["path"])
        file_path = self.root.joinpath(*relative.parts).resolve()
        if not file_path.is_relative_to(self.root):
            raise ValueError(f"Sample escapes dataset directory: {record['path']}")
        n, height, width = record["num_frames"], record["height"], record["width"]
        frame_indices = monotonic_uniform_indices(n, self.clip_len)
        packed = np.load(file_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (n, (height * width + 7) // 8)
        if packed.dtype != np.uint8 or packed.shape != expected_shape:
            raise ValueError(f"Invalid packed frame array in {file_path}: {packed.dtype}, {packed.shape}; expected uint8 {expected_shape}")
        selected = packed[frame_indices]
        del packed
        frames = np.unpackbits(selected, axis=1, count=height * width, bitorder="little")
        frames = frames.reshape(self.clip_len, height, width) * np.uint8(255)
        return {
            "frames": frames,
            "frame_indices": frame_indices,
            "source_frame_indices": frame_indices + record["start_frame"],
            "metadata": record.copy(),
        }
