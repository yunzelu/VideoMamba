#!/bin/bash
#SBATCH --job-name=vm_real_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --time=02:00:00
#SBATCH --output=vm_real_eval_%j.out
#SBATCH --error=vm_real_eval_%j.err

# Submit after preparing the evaluation manifest and clips. Set REAL_MANIFEST
# to its absolute path. The runtime is an estimate; override resources using
# sbatch options after measuring a small run.
set -euo pipefail

module purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2

repo=${VIDEOMAMBA_REPO:-/project/def-mbolic/yunzelu/VideoMamba}
project=${CANNY_PROJECT:-"$repo/lab"}
environment=${VIDEOMAMBA_VENV:-"$repo/.venv"}
checkpoint=${REAL_CHECKPOINT:-"$project/runs/ddp-train-26347306/best.pt"}
manifest=${REAL_MANIFEST:?Set REAL_MANIFEST to the absolute path of the prepared manifest.jsonl}
output=${REAL_OUTPUT:-"$project/runs/real-world-eval-${SLURM_JOB_ID}"}
batch_size=${REAL_BATCH_SIZE:-2}
workers=${REAL_WORKERS:-2}

for path in "$repo" "$project" "$environment" "$checkpoint" "$manifest" "$output"; do
  if [[ "$path" != /* ]]; then
    echo "Expected absolute path: $path" >&2
    exit 1
  fi
done
if [[ ! -f "$environment/bin/activate" || ! -f "$checkpoint" || ! -f "$manifest" || ! -f "$project/real_world/evaluate.py" ]]; then
  echo "Missing venv, checkpoint, manifest, or real-world evaluation script." >&2
  exit 1
fi
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ || ! "$workers" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "REAL_BATCH_SIZE must be positive and REAL_WORKERS non-negative." >&2
  exit 1
fi

source "$environment/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "Job: ${SLURM_JOB_ID:-unknown}; host: $(hostname); Python: $(command -v python)"
echo "Manifest: $manifest; checkpoint: $checkpoint; output: $output"
python -u "$project/real_world/evaluate.py" \
  --repo "$repo" --manifest "$manifest" --checkpoint "$checkpoint" --output "$output" \
  --batch-size "$batch_size" --workers "$workers"
echo "Evaluation finished."
