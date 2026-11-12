#!/bin/bash
#SBATCH --job-name=canny_videomamba_ddp
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --gpus-per-node=h100:4
#SBATCH --time=01:00:00
#SBATCH --output=canny_ddp_%j.out
#SBATCH --error=canny_ddp_%j.err

# Single-node, four full H100 GPUs. Defaults to a short benchmark. For training,
# submit with CANNY_MODE=train and set an appropriate --time on sbatch.
# CANNY_BATCH_SIZE and CANNY_WORKERS are PER GPU / per process.
set -eo pipefail

module purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2

repo=${VIDEOMAMBA_REPO:-/project/def-mbolic/yunzelu/VideoMamba}
project=${CANNY_PROJECT:-"$repo/lab"}
dataset=${CANNY_DATASET:-/project/def-mbolic/yunzelu/_dataset/canny/dataset}
environment=${VIDEOMAMBA_VENV:-"$repo/.venv"}
mode=${CANNY_MODE:-benchmark}
frames=${CANNY_FRAMES:-32}
batch_size=${CANNY_BATCH_SIZE:-4}
workers=${CANNY_WORKERS:-4}
checkpoint_blocks=${CANNY_CHECKPOINT_BLOCKS:-24}
weights=${CANNY_PRETRAINED:-"$project/checkpoints/videomamba_t16_k400_f${frames}_res224.pth"}
out=${CANNY_OUTPUT:-"$project/runs/ddp-${mode}-${SLURM_JOB_ID}"}

case "$mode" in benchmark|train) ;; *) echo "CANNY_MODE must be benchmark or train" >&2; exit 1 ;; esac
case "$frames" in 16|32) ;; *) echo "CANNY_FRAMES must be 16 or 32" >&2; exit 1 ;; esac
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ || ! "$workers" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "CANNY_BATCH_SIZE must be positive and CANNY_WORKERS non-negative integers" >&2
  exit 1
fi
if [[ "$dataset" != /* ]]; then
  echo "CANNY_DATASET must be an absolute path to the dataset directory" >&2
  exit 1
fi
if [[ ! -f "$environment/bin/activate" || ! -f "$dataset/manifest.jsonl" || \
      ! -f "$project/canny_dataset.py" || ! -f "$project/training/train_canny.py" || \
      ! -f "$project/training/distributed_canny.py" || ! -f "$weights" ]]; then
  echo "Missing venv, manifest, dataset reader, trainer, distributed runner, or pretrained weights" >&2
  exit 1
fi
if [[ "$mode" == "train" && -n "${CANNY_RESUME:-}" ]]; then
  if [[ -z "${CANNY_OUTPUT:-}" ]]; then
    echo "Set CANNY_OUTPUT to the existing run directory when resuming" >&2
    exit 1
  fi
  if [[ ! -f "$CANNY_RESUME" ]]; then
    echo "Resume checkpoint not found: $CANNY_RESUME" >&2
    exit 1
  fi
fi

source "$environment/bin/activate"
# Each of the four training ranks owns its data-loader workers. Avoid CPU
# thread oversubscription; override with CANNY_CPU_THREADS if needed.
export OMP_NUM_THREADS=${CANNY_CPU_THREADS:-1}
export MKL_NUM_THREADS=${CANNY_CPU_THREADS:-1}
export OPENBLAS_NUM_THREADS=${CANNY_CPU_THREADS:-1}
export PYTHONUNBUFFERED=1

echo "Job: ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
echo "Python: $(command -v python)"
echo "VideoMamba: $repo"
echo "Dataset: $dataset"
echo "Mode: $mode; frames: $frames; GPU count: 4"
echo "Batch size per GPU: $batch_size; total batch size: $((4 * batch_size)); workers per GPU: $workers"
echo "Activation checkpoint blocks: $checkpoint_blocks"
echo "Pretrained weights: $weights"

common=(--repo "$repo" --data "$dataset" --pretrained "$weights" --frames "$frames"
        --batch-size "$batch_size" --workers "$workers" --checkpoint-blocks "$checkpoint_blocks" --precision bf16)

if [[ "$mode" == "benchmark" ]]; then
  echo "Starting four-GPU benchmark; no training checkpoint will be saved"
  python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    "$project/training/distributed_canny.py" benchmark "${common[@]}" \
    --warmup-steps "${CANNY_WARMUP_STEPS:-3}" --steps "${CANNY_BENCHMARK_STEPS:-30}"
else
  echo "Training output: $out"
  resume_args=()
  if [[ -n "${CANNY_RESUME:-}" ]]; then
    resume_args=(--resume "$CANNY_RESUME")
  fi
  python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    "$project/training/distributed_canny.py" train "${common[@]}" \
    --output "$out" --epochs "${CANNY_EPOCHS:-30}" \
    --lr "${CANNY_LR:-0.0001}" --weight-decay "${CANNY_WEIGHT_DECAY:-0.05}" "${resume_args[@]}"
fi

echo "Four-GPU $mode process exited successfully"
