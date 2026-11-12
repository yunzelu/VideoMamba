#!/bin/bash
#SBATCH --job-name=canny_import_trace
#SBATCH --mail-user=yunzelu@outlook.com
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=canny_import_%j.out
#SBATCH --error=canny_import_%j.err

# Startup investigation only: no dataset access, checkpoint, training, or GPU.
set -eo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2
repo=${VIDEOMAMBA_REPO:-/project/def-mbolic/yunzelu/VideoMamba}
project=${CANNY_PROJECT:-"$repo/lab"}
environment=${VIDEOMAMBA_VENV:-"$repo/.venv"}
source "$environment/bin/activate"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Job ${SLURM_JOB_ID:-unknown}; host $(hostname); python $(command -v python)"

# Match the four-GPU rank's torch-first import order. Each stack trace reports
# Python frames, not necessarily the native operation that is blocked.
python -u - "$project/training" "$repo" <<'PY' &
import faulthandler
import importlib
import sys
import time

sys.path.insert(0, sys.argv[1])
faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)

def stage(label, fn):
    print(f'START {label}', flush=True)
    start = time.monotonic()
    result = fn()
    print(f'DONE {label}: {time.monotonic() - start:.2f}s', flush=True)
    return result

stage('import torch', lambda: importlib.import_module('torch'))
stage('import torch distributed and data utilities', lambda: (
    importlib.import_module('torch.distributed'),
    importlib.import_module('torch.nn.parallel'),
    importlib.import_module('torch.utils.data')))
trainer = stage('import train_canny and its dependencies', lambda: importlib.import_module('train_canny'))
stage('import VideoMamba model source and its dependencies', lambda: trainer.load_factory(sys.argv[2]))
faulthandler.cancel_dump_traceback_later()
print('Import diagnostic completed', flush=True)
PY
python_pid=$!

# Report the kernel wait channel if available. Its label is only a clue.
(
  while kill -0 "$python_pid" 2>/dev/null; do
    date --iso-8601=seconds
    ps -o pid,stat,wchan:32,etime,time,comm -p "$python_pid" || true
    sleep 60
  done
) &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
wait "$python_pid"
