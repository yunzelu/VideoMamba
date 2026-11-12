#!/bin/bash
#SBATCH --job-name=canny_cpu_diag
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=cpu_diag_%j.out
#SBATCH --error=cpu_diag_%j.err

# CPU-only: no GPU is requested. Submit from VideoMamba/lab for local logs.
set -eo pipefail

repo=${VIDEOMAMBA_REPO:-/project/def-mbolic/yunzelu/VideoMamba}
lab=${CANNY_PROJECT:-"$repo/lab"}
dataset=${CANNY_DATASET:-/project/def-mbolic/yunzelu/_dataset/canny/dataset}
weights=${CANNY_PRETRAINED:-"$lab/checkpoints/videomamba_t16_k400_f32_res224.pth"}
environment=${VIDEOMAMBA_VENV:-"$repo/.venv"}

echo "[$(date --iso-8601=seconds)] START CPU diagnostic job ${SLURM_JOB_ID:-unknown}"
echo "Host: $(hostname)"
module purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2
if [[ ! -f "$environment/bin/activate" ]]; then
  echo "Virtual environment not found: $environment" >&2
  exit 1
fi
if [[ ! -f "$lab/training/diagnose_cpu.py" || ! -f "$lab/training/train_canny.py" || ! -f "$lab/canny_dataset.py" ]]; then
  echo "Diagnostic, trainer, or dataset reader missing under $lab" >&2
  exit 1
fi
source "$environment/bin/activate"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=""
echo "[$(date --iso-8601=seconds)] START diagnostic Python: $(command -v python)"
python -u "$lab/training/diagnose_cpu.py" --data "$dataset" --weights "$weights"
echo "[$(date --iso-8601=seconds)] CPU diagnostic job completed"
