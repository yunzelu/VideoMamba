#!/bin/bash
#SBATCH --job-name=canny_videomamba
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --output=canny_%j.out
#SBATCH --error=canny_%j.err

# Set CANNY_PROJECT / VIDEOMAMBA_REPO or use the documented defaults.
# Submit from a directory where Slurm may write its stdout/stderr logs.
set -eo pipefail

module purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2

repo=${VIDEOMAMBA_REPO:-/project/def-mbolic/yunzelu/VideoMamba}
project=${CANNY_PROJECT:-"$repo/lab"}
dataset=${CANNY_DATASET:-/project/def-mbolic/yunzelu/_dataset/canny/dataset}
if [[ "$dataset" != /* ]]; then
  echo "CANNY_DATASET must be an absolute path to the dataset directory." >&2
  exit 1
fi
environment=${VIDEOMAMBA_VENV:-"$repo/.venv"}
mode=${CANNY_MODE:-benchmark}
out=${CANNY_OUTPUT:-"$project/runs/${mode}-${SLURM_JOB_ID}"}
batch_size=${CANNY_BATCH_SIZE:-2}
workers=${CANNY_WORKERS:-2}
frames=${CANNY_FRAMES:-32}
case "$frames" in
  16|32) ;;
  *) echo "CANNY_FRAMES must be 16 or 32" >&2; exit 1 ;;
esac

if [[ ! -f "$environment/bin/activate" ]]; then
  echo "Virtual environment not found: $environment" >&2
  exit 1
fi
if [[ ! -f "$dataset/manifest.jsonl" ]]; then
  echo "Dataset manifest not found: $dataset/manifest.jsonl" >&2
  exit 1
fi
if [[ ! -f "$project/training/train_canny.py" || ! -f "$project/canny_dataset.py" ]]; then
  echo "Missing training script or dataset reader under $project" >&2
  exit 1
fi
source "$environment/bin/activate"
mkdir -p "$out"

echo "Job: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Python: $(command -v python)"
echo "VideoMamba: $repo"
echo "Dataset: $dataset"
echo "Output: $out"

if [[ "$mode" == "train" || "$mode" == "evaluate" ]]; then
  echo "[job] $mode process will check CUDA; skipping a second PyTorch import"
else
  echo "[job] Starting preliminary GPU check"
  python - <<'PY'
import torch
print('PyTorch:', torch.__version__, 'PyTorch CUDA:', torch.version.cuda, flush=True)
if not torch.cuda.is_available():
    raise RuntimeError('PyTorch cannot see a GPU in this job')
device = torch.cuda.get_device_properties(0)
print('GPU:', device.name, 'memory GiB:', round(device.total_memory / 1024**3, 2), flush=True)
PY
  echo "[job] Preliminary GPU check process exited successfully"
fi

case "$mode" in
  benchmark)
    weights=${CANNY_PRETRAINED:-"$project/checkpoints/videomamba_t16_k400_f${frames}_res224.pth"}
    if [[ ! -f "$weights" ]]; then
      echo "Pretrained checkpoint not found: $weights" >&2
      exit 1
    fi
    echo "[job] Launching benchmark Python process"
    python -u "$project/training/train_canny.py" benchmark \
      --repo "$repo" --data "$dataset" --pretrained "$weights" \
      --frames "$frames" --batch-size "$batch_size" --workers "$workers" \
      --warmup-steps "${CANNY_WARMUP_STEPS:-5}" --steps "${CANNY_BENCHMARK_STEPS:-30}" \
      --lr "${CANNY_LR:-0.0001}" --checkpoint-blocks "${CANNY_CHECKPOINT_BLOCKS:-24}" \
      --precision bf16
    echo "[job] Benchmark Python process exited successfully"
    ;;
  train)
    weights=${CANNY_PRETRAINED:-"$project/checkpoints/videomamba_t16_k400_f${frames}_res224.pth"}
    if [[ ! -f "$weights" ]]; then
      echo "Pretrained checkpoint not found: $weights" >&2
      exit 1
    fi
    resume_args=()
    if [[ -n "${CANNY_RESUME:-}" ]]; then
      if [[ ! -f "$CANNY_RESUME" ]]; then
        echo "Resume checkpoint not found: $CANNY_RESUME" >&2
        exit 1
      fi
      resume_args=(--resume "$CANNY_RESUME")
    fi
    echo "[job] Launching training Python process"
    python -u "$project/training/train_canny.py" train \
      --repo "$repo" --data "$dataset" --output "$out" --pretrained "$weights" \
      --epochs "${CANNY_EPOCHS:-1}" --frames "$frames" --batch-size "$batch_size" --workers "$workers" \
      --lr "${CANNY_LR:-0.0001}" --checkpoint-blocks "${CANNY_CHECKPOINT_BLOCKS:-24}" \
      --precision bf16 "${resume_args[@]}"
    echo "[job] Training Python process exited successfully"
    ;;
  evaluate)
    if [[ -z "${CANNY_CHECKPOINT:-}" || ! -f "$CANNY_CHECKPOINT" ]]; then
      echo "Set CANNY_CHECKPOINT to a trained checkpoint for evaluation." >&2
      exit 1
    fi
    echo "[job] Launching evaluation Python process"
    python -u "$project/training/train_canny.py" evaluate \
      --repo "$repo" --data "$dataset" --output "$out" --checkpoint "$CANNY_CHECKPOINT" \
      --split "${CANNY_EVAL_SPLIT:-test}" --batch-size "$batch_size" --workers "$workers"
    echo "[job] Evaluation Python process exited successfully"
    ;;
  *)
    echo "Unknown CANNY_MODE: $mode (expected benchmark, train or evaluate)" >&2
    exit 1
    ;;
esac
