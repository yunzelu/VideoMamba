"""Single-node distributed VideoMamba training and real-data benchmarking.

Launch with torchrun, one process per GPU. `--batch-size` is per GPU. Epoch
checkpoints are resumable; partial snapshots are evaluation-only.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import sys
import time

print(f"[distributed rank={os.environ.get('RANK', '?')}] Loading dependencies", flush=True)
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

import train_canny as base


def log(message):
    if dist.get_rank() == 0:
        print(f"[distributed] {message}", flush=True)


def make_distributed_loader(dataset, batch_size, workers, seed, rank, world_size, shuffle, epoch):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=shuffle, seed=seed, drop_last=False)
    sampler.set_epoch(epoch)
    generator = torch.Generator().manual_seed(seed + epoch * world_size + rank)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        sampler=sampler, pin_memory=torch.cuda.is_available(),
                        generator=generator, drop_last=False)
    return loader


def reduce_train_metrics(matrix, loss_sum, seen, device):
    """Reduce locally accumulated statistics, returning identical results on all ranks."""
    counts = matrix.to(device=device, dtype=torch.float64)
    totals = torch.tensor([loss_sum, seen], device=device, dtype=torch.float64)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return counts.cpu(), float(totals[0].item()), int(totals[1].item())


def rank0_validation(model, loader, device, precision, group=None):
    """Run validation once on the complete validation set, then broadcast metrics."""
    result = [None]
    if dist.get_rank(group) == 0:
        # Using the unwrapped module avoids DDP forward collectives on rank zero.
        unwrapped = model.module if hasattr(model, "module") else model
        result[0], _ = base.run_evaluation(unwrapped, loader, device, precision)
    dist.broadcast_object_list(result, src=0, group=group)
    return result[0]


def initialize():
    if not torch.cuda.is_available():
        raise RuntimeError("This entry point requires a CUDA GPU for each rank")
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun, one process per GPU")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=3))
    properties = torch.cuda.get_device_properties(local_rank)
    print(f"[distributed rank={dist.get_rank()}] GPU {local_rank}: {properties.name}; "
          f"memory={properties.total_memory / 1024**3:.2f} GiB", flush=True)
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def model_and_loader(args, rank, world, device, training):
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Selected GPU does not support BF16")
    if args.batch_size < 1 or args.workers < 0 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid batch size, worker count, learning rate or weight decay")
    log("Validating dataset and subject splits")
    datasets, subjects = base.validate_dataset(args.data, args.frames)
    dataset = base.CannyVideoDataset(datasets["train"], args.normalization)
    if len(dataset) % world:
        log(f"DistributedSampler pads the training epoch with {world - len(dataset) % world} repeated clips; epoch metrics include them")
    loader = make_distributed_loader(dataset, args.batch_size, args.workers,
                                     args.seed, rank, world, True, 0)
    log(f"Training clips={len(dataset)}, GPUs={world}, per-GPU batch={args.batch_size}, global batch={world * args.batch_size}")
    base.seed_epoch(args.seed)
    log("Importing and constructing VideoMamba-Tiny")
    model = base.create_model(args.repo, args.checkpoint_blocks, args.frames,
                              progress=log if rank == 0 else None)
    log("Loading pretrained Kinetics-400 backbone")
    base.load_k400_checkpoint(model, args.pretrained)
    model = model.to(device)
    # The upstream model uses reentrant activation checkpointing. static_graph
    # permits DDP's supported reentrant-backward pattern on a fixed graph.
    model = DistributedDataParallel(model, device_ids=[device.index],
                                    output_device=device.index, static_graph=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log("Model and optimizer ready")
    return datasets, subjects, dataset, loader, model, optimizer


def batch_step(model, optimizer, batch, device, precision):
    frames, target, _ = batch
    frames, target = frames.to(device, non_blocking=True), target.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with base.amp_context(device, precision):
        logits = model(frames)
        loss = torch.nn.functional.cross_entropy(logits.float(), target)
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError("Non-finite logits or loss")
    loss.backward()
    optimizer.step()
    return logits, target, loss


def benchmark(args, rank, world, device):
    if args.warmup_steps < 0 or args.steps < 1:
        raise ValueError("Warmup steps must be nonnegative and steps positive")
    _, _, dataset, loader, model, optimizer = model_and_loader(args, rank, world, device, False)
    if args.warmup_steps + args.steps > len(loader):
        raise ValueError("Benchmark steps exceed one epoch")
    iterator = iter(loader)
    model.train()
    for step in range(args.warmup_steps):
        if step == 0:
            log("Fetching first batch; running forward, backward and optimizer")
        batch_step(model, optimizer, next(iterator), device, args.precision)
        torch.cuda.synchronize(device)
        log(f"Warmup {step + 1}/{args.warmup_steps} complete")
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    local_count = 0
    for step in range(args.steps):
        _, target, _ = batch_step(model, optimizer, next(iterator), device, args.precision)
        local_count += len(target)
        if (step + 1) % 10 == 0 or step + 1 == args.steps:
            log(f"Measured steps {step + 1}/{args.steps}")
    torch.cuda.synchronize(device)
    elapsed = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=device)
    samples = torch.tensor(local_count, dtype=torch.int64, device=device)
    memory = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
                          dtype=torch.int64, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    dist.all_reduce(samples, op=dist.ReduceOp.SUM)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    if rank == 0:
        seconds, count = float(elapsed.item()), int(samples.item())
        print(json.dumps({"world_size": world, "batch_per_gpu": args.batch_size,
                          "global_batch": world * args.batch_size, "frames": args.frames,
                          "measured_steps": args.steps, "measured_samples": count,
                          "seconds": seconds, "clips_per_second": count / seconds,
                          "max_peak_allocated_gib_per_gpu": int(memory[0].item()) / 1024**3,
                          "max_peak_reserved_gib_per_gpu": int(memory[1].item()) / 1024**3,
                          "approx_train_epoch_minutes": len(dataset) / (count / seconds) / 60,
                          "note": "Estimate excludes validation, startup and storage variability."}, indent=2), flush=True)


def provenance_for(args, subjects, world):
    return {"manifest_sha256": base.sha256_file(Path(args.data) / "manifest.jsonl"),
            "subjects": subjects, "labels": base.LABELS, "model": "VideoMamba-Tiny",
            "frames": args.frames, "image_size": 224,
            "model_source_sha256": base.sha256_file(Path(args.repo) / "videomamba/video_sm/models/videomamba.py"),
            "resize": "antialiased_bilinear_letterbox_black", "normalization": args.normalization,
            "precision": args.precision, "checkpoint_blocks": args.checkpoint_blocks,
            "batch_size": args.batch_size, "world_size": world, "global_batch_size": args.batch_size * world,
            "seed": args.seed, "lr": args.lr, "weight_decay": args.weight_decay, "epochs": args.epochs,
            "pretrained_sha256": base.sha256_file(args.pretrained),
            "torch_version": str(torch.__version__), "torch_cuda": torch.version.cuda}


def train(args, rank, world, device):
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    output = Path(args.output)
    # Rank zero alone performs filesystem writes. Never overwrite an earlier run.
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if args.resume and Path(args.resume).resolve() != (output / "last.pt").resolve():
            raise ValueError("Resume must use last.pt from the same output directory")
        if not args.resume and ((output / "last.pt").exists() or (output / "partial.pt").exists()):
            raise FileExistsError("Output already contains a checkpoint; use a new directory or explicitly resume last.pt")
    dist.barrier()
    datasets, subjects, train_set, _, model, optimizer = model_and_loader(args, rank, world, device, True)
    provenance = provenance_for(args, subjects, world)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start, best = 0, -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint["provenance"] != provenance:
            raise ValueError("Resume parameters, data, source or world size differ")
        model.module.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, best = checkpoint["epoch"], checkpoint["best_val_macro_f1"]
    if rank == 0:
        base.write_json(output / "run_config.json", provenance)
    val_set = base.CannyVideoDataset(datasets["val"], args.normalization)
    val_loader = base.make_loader(val_set, args.batch_size, args.workers, False, args.seed) if rank == 0 else None
    for epoch in range(start, args.epochs):
        base.seed_epoch(args.seed + epoch * world + rank)
        train_loader = make_distributed_loader(train_set, args.batch_size, args.workers,
                                               args.seed, rank, world, True, epoch)
        model.train()
        matrix = torch.zeros((len(base.LABELS), len(base.LABELS)), dtype=torch.int64)
        total_loss, seen = 0.0, 0
        dist.barrier()
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        last_snapshot = started
        log(f"Epoch {epoch + 1}/{args.epochs}: {len(train_loader)} distributed steps")
        # Time each rank independently without adding CUDA synchronization or
        # distributed collectives. Data wait is only the time blocked in
        # next(iterator), so it excludes background prefetch. Step wall time
        # includes training and metric collection, not checkpoint writes, and
        # is not an isolated GPU compute measurement.
        iterator_started = time.perf_counter()
        train_iterator = iter(train_loader)
        iterator_seconds = time.perf_counter() - iterator_started
        print(f"[timing rank={rank}] epoch={epoch + 1} iterator_start_s={iterator_seconds:.3f}", flush=True)
        wait_sum = wait_max = step_sum = step_max = 0.0
        timing_count = 0
        for step in range(1, len(train_loader) + 1):
            wait_started = time.perf_counter()
            batch = next(train_iterator)
            wait_seconds = time.perf_counter() - wait_started
            step_started = time.perf_counter()
            logits, target, loss = batch_step(model, optimizer, batch, device, args.precision)
            base.add_confusion(matrix, target, logits.argmax(dim=1))
            total_loss += float(loss.item()) * len(target)
            seen += len(target)
            step_seconds = time.perf_counter() - step_started
            wait_sum += wait_seconds
            wait_max = max(wait_max, wait_seconds)
            step_sum += step_seconds
            step_max = max(step_max, step_seconds)
            timing_count += 1
            if rank == 0 and (step == 1 or time.perf_counter() - last_snapshot >= 300):
                log(f"Saving evaluation-only partial model: epoch={epoch + 1} step={step}")
                base.save_partial_model(output / "partial.pt", model.module, provenance, epoch + 1,
                                        step, seen * world)
                last_snapshot = time.perf_counter()
            if step % args.log_interval == 0:
                log(f"epoch={epoch + 1} step={step}/{len(train_loader)} approximate_global_clips={seen * world} clips_per_second={(seen * world) / (time.perf_counter() - started):.2f}")
            if step % args.log_interval == 0 or step == len(train_loader):
                print(f"[timing rank={rank}] epoch={epoch + 1} step={step}/{len(train_loader)} "
                      f"steps={timing_count} data_wait_mean_s={wait_sum / timing_count:.3f} "
                      f"data_wait_max_s={wait_max:.3f} step_wall_mean_s={step_sum / timing_count:.3f} "
                      f"step_wall_max_s={step_max:.3f}", flush=True)
                wait_sum = wait_max = step_sum = step_max = 0.0
                timing_count = 0
        # Release workers before validation, matching normal DataLoader loop
        # exhaustion even though this loop requests exactly len(loader) batches.
        del train_iterator
        torch.cuda.synchronize(device)
        elapsed = torch.tensor(time.perf_counter() - started, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        global_matrix, global_loss, global_seen = reduce_train_metrics(matrix, total_loss, seen, device)
        metrics = base.metrics_from_confusion(global_matrix)
        metrics.update(loss=global_loss / global_seen, samples=global_seen, seconds=float(elapsed.item()),
                       clips_per_second=global_seen / float(elapsed.item()))
        log("Training epoch complete; running complete validation on GPU 0")
        validation = rank0_validation(model, val_loader, device, args.precision)
        scheduler.step()
        is_best = validation["macro_f1_fixed_10"] > best
        if is_best:
            best = validation["macro_f1_fixed_10"]
        if rank == 0:
            payload = {"epoch": epoch + 1, "model": model.module.state_dict(),
                       "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                       "best_val_macro_f1": best, "provenance": provenance}
            base.atomic_save(output / "last.pt", payload)
            if is_best:
                base.atomic_save(output / "best.pt", payload)
            with (output / "epochs.jsonl").open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps({"epoch": epoch + 1, "train": metrics, "val": validation,
                                           "best_val_macro_f1": best}, allow_nan=False) + "\n")
            log(f"epoch={epoch + 1} val_acc={validation['accuracy']:.4f} val_macro_f1={validation['macro_f1_fixed_10']:.4f} clips_per_second={metrics['clips_per_second']:.2f}")
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("benchmark", "train"):
        p = sub.add_parser(name)
        p.add_argument("--repo", type=Path, required=True)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--pretrained", type=Path, required=True)
        p.add_argument("--batch-size", type=int, default=4, help="Clips per GPU")
        p.add_argument("--workers", type=int, default=2, help="DataLoader workers per GPU")
        p.add_argument("--frames", type=int, choices=(16, 32), default=32)
        p.add_argument("--lr", type=float, default=0.0001)
        p.add_argument("--weight-decay", type=float, default=0.05)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--checkpoint-blocks", type=int, default=24)
        p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
        p.add_argument("--normalization", choices=("imagenet", "zero_one"), default="imagenet")
        if name == "benchmark":
            p.add_argument("--warmup-steps", type=int, default=3)
            p.add_argument("--steps", type=int, default=10)
        else:
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--epochs", type=int, default=30)
            p.add_argument("--resume", type=Path)
            p.add_argument("--log-interval", type=int, default=25)
    args = parser.parse_args()
    if args.command == "train" and args.log_interval < 1:
        raise ValueError("log-interval must be positive")
    rank, world, device = initialize()
    try:
        if args.command == "benchmark":
            benchmark(args, rank, world, device)
        else:
            train(args, rank, world, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
