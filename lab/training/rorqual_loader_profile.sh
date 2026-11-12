#!/bin/bash
#SBATCH --job-name=canny_loader_cpu
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=canny_loader_%j.out
#SBATCH --error=canny_loader_%j.err

# CPU-only: no GPU allocation. Submit from VideoMamba/lab for logs.
set -eo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2
repo=${VIDEOMAMBA_REPO:-/project/def-mbolic/yunzelu/VideoMamba}
project=${CANNY_PROJECT:-"$repo/lab"}
data=${CANNY_DATASET:-/project/def-mbolic/yunzelu/_dataset/canny/dataset}
environment=${VIDEOMAMBA_VENV:-"$repo/.venv"}
if [[ ! -f "$environment/bin/activate" || ! -f "$project/training/profile_loader_cpu.py" || ! -f "$project/training/train_canny.py" || ! -f "$project/canny_dataset.py" || ! -f "$data/manifest.jsonl" ]]; then
  echo 'Missing virtual environment, diagnostic script, dataset reader, trainer, or manifest' >&2
  exit 1
fi
source "$environment/bin/activate"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "CPU-only job ${SLURM_JOB_ID:-unknown}; host $(hostname); python $(command -v python)"
python -u "$project/training/profile_loader_cpu.py" --data "$data" --samples 32 --workers 4
echo 'CPU-only data-loader job completed'
