"""Time Canny data loading on one CPU worker group, without requesting a GPU.

This uses the normal training reader and transform. It does not reproduce the
concurrent four-rank GPU workload or measure GPU computation/communication.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import time


def log(message):
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('/project/def-mbolic/yunzelu/_dataset/canny/dataset'))
    parser.add_argument('--samples', type=int, default=32, help='Number of clips (maximum 32)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--also-zero-workers', action='store_true', help='Repeat identical sample order with no workers; cache state may differ')
    args = parser.parse_args()
    if not 1 <= args.samples <= 32 or args.workers < 0:
        parser.error('samples must be between 1 and 32; workers must be nonnegative')

    log('CPU-only data-loader profile starting; no GPU requested')
    log('Importing PyTorch and trainer')
    start_import = time.monotonic()
    import torch
    from torch.utils.data import DataLoader, Subset
    import train_canny as base
    log(f'Imports complete in {time.monotonic() - start_import:.2f}s')
    torch.set_num_threads(1)

    log('Validating manifest and subject splits')
    datasets, _ = base.validate_dataset(args.data, 32)
    dataset = base.CannyVideoDataset(datasets['train'], 'imagenet')
    # Match the rank-0 epoch-zero DistributedSampler order of the four-GPU job.
    generator = torch.Generator().manual_seed(42)
    order = torch.randperm(len(dataset), generator=generator).tolist()[0::4][:args.samples]
    if len(order) != args.samples:
        raise ValueError('Not enough training samples')
    subset = Subset(dataset, order)

    def run(workers):
        log(f'Starting loader with workers={workers}, batch_size=4, samples={len(subset)}')
        loader = DataLoader(subset, batch_size=4, num_workers=workers, shuffle=False,
                            pin_memory=False, timeout=60 if workers else 0)
        started = time.monotonic()
        iterator = iter(loader)
        init_seconds = time.monotonic() - started
        waits = []
        seen = 0
        try:
            for i in range(len(loader)):
                before = time.monotonic()
                frames, _, _ = next(iterator)
                waits.append(time.monotonic() - before)
                if tuple(frames.shape[1:]) != (3, 32, 224, 224):
                    raise ValueError(f'Unexpected input shape {tuple(frames.shape)}')
                seen += len(frames)
                log(f'workers={workers}: batch={i+1}/{len(loader)} clips={seen} wait={waits[-1]:.2f}s')
        finally:
            del iterator
        elapsed = time.monotonic() - started
        log(json.dumps({'workers': workers, 'clips': seen, 'batch_size': 4,
                        'iterator_start_seconds': init_seconds, 'total_seconds': elapsed,
                        'clips_per_second': seen / elapsed,
                        'batch_wait_mean_seconds': statistics.fmean(waits),
                        'batch_wait_median_seconds': statistics.median(waits),
                        'batch_wait_max_seconds': max(waits)}, sort_keys=True))

    run(args.workers)
    if args.also_zero_workers and args.workers != 0:
        log('Repeating the same samples without workers; cache state may differ')
        run(0)
    log('CPU loader profile completed; results do not measure GPU compute or four-rank contention')


if __name__ == '__main__':
    main()
