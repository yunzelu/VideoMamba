#!/bin/bash
#SBATCH --job-name=vm_real_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=vm_real_prepare_%j.out
#SBATCH --error=vm_real_prepare_%j.err

# CPU-only preparation. Set REAL_SQLITE_DIR, REAL_ANNOTATIONS, REAL_PREPARED.
set -euo pipefail
module purge
module load StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2
repo=${VIDEOMAMBA_REPO:-/project/def-mbolic/yunzelu/VideoMamba}
project=${CANNY_PROJECT:-"$repo/lab"}
environment=${VIDEOMAMBA_VENV:-"$repo/.venv"}
sqlite_dir=${REAL_SQLITE_DIR:?Set REAL_SQLITE_DIR to the directory of standalone SQLite snapshots}
annotations=${REAL_ANNOTATIONS:?Set REAL_ANNOTATIONS to the annotation CSV path}
prepared=${REAL_PREPARED:?Set REAL_PREPARED to the prepared output directory}
stride=${REAL_STRIDE_SECONDS:-60}
gap=${REAL_MAX_GAP_SECONDS:-0.25}
window_labeling=${REAL_WINDOW_LABELING:-pure}
for path in "$repo" "$project" "$environment" "$sqlite_dir" "$annotations" "$prepared"; do
  if [[ "$path" != /* ]]; then echo "Expected absolute path: $path" >&2; exit 1; fi
done
if [[ ! -f "$environment/bin/activate" || ! -f "$annotations" || ! -d "$sqlite_dir" || ! -f "$project/real_world/prepare.py" ]]; then
  echo "Missing venv, annotation CSV, SQLite directory, or prepare.py." >&2; exit 1
fi
sqlite_files=()
while IFS= read -r -d '' f; do sqlite_files+=("$f"); done < <(find "$sqlite_dir" -maxdepth 1 -type f -name '*.sqlite' -print0 | sort -z)
if (( ${#sqlite_files[@]} == 0 )); then echo "No SQLite files found in $sqlite_dir" >&2; exit 1; fi
source "$environment/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "Job: ${SLURM_JOB_ID:-unknown}; host: $(hostname); Python: $(command -v python)"
echo "SQLite files: ${#sqlite_files[@]}; annotations: $annotations; output: $prepared"
python -u "$project/real_world/prepare.py" --sqlite "${sqlite_files[@]}" \
  --annotations "$annotations" --output-dir "$prepared" \
  --stride-seconds "$stride" --max-gap-seconds "$gap" --window-labeling "$window_labeling"
echo "Preparation finished."
