# Canny VideoMamba on Rorqual

This guide uses the user's existing repository and environment:

- Repository: `/project/def-mbolic/yunzelu/VideoMamba`
- Virtual environment: `/project/def-mbolic/yunzelu/VideoMamba/.venv`
- Canny dataset: `/project/def-mbolic/yunzelu/_dataset/canny/dataset/`
- Training code, checkpoints, and run outputs: `/project/def-mbolic/yunzelu/VideoMamba/lab/`

The user confirmed the dataset's manifest is in this directory. Its completeness
must still be checked. No Rorqual training or benchmark is claimed by these
instructions.

## 1. Upload the small code files

Run these commands in **Windows PowerShell** on the computer containing the
local project. They transfer only the reader and the training folder, not the
28.3 GiB dataset or the source videos.

```powershell
Set-Location 'D:\lu\project\retirement_home\canny_action_recognition'
ssh yunzelu@rorqual.alliancecan.ca 'mkdir -p /project/def-mbolic/yunzelu/VideoMamba/lab/checkpoints /project/def-mbolic/yunzelu/VideoMamba/lab/training'
scp canny_dataset.py 'yunzelu@rorqual.alliancecan.ca:/project/def-mbolic/yunzelu/VideoMamba/lab/'
scp training/train_canny.py training/rorqual_job.sh training/RORQUAL.md 'yunzelu@rorqual.alliancecan.ca:/project/def-mbolic/yunzelu/VideoMamba/lab/training/'
```

The remote code layout should then include:

```text
VideoMamba/lab/
  canny_dataset.py
  training/train_canny.py
  training/rorqual_job.sh
  checkpoints/
```

On the Rorqual login node, verify that the existing environment is visible at
the path the job will use:

```bash
test -x /project/def-mbolic/yunzelu/VideoMamba/.venv/bin/python && echo 'Environment found' || echo 'Environment not found'
```

## 2. Check the uploaded dataset

On the Rorqual **login node**, check the confirmed dataset location:

```bash
ls -l /project/def-mbolic/yunzelu/_dataset/canny/dataset/manifest.jsonl
ls -l /project/def-mbolic/yunzelu/_dataset/canny/dataset/dataset_config.json
```

The job uses this location by default, so `CANNY_DATASET` does not need to be
set. If you had previously set it to a different location in the same shell,
run `unset CANNY_DATASET` before submitting the job. An explicit
`CANNY_DATASET` can still be used to override the default.

Check that every file listed in the manifest has the expected size:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('/project/def-mbolic/yunzelu/_dataset/canny/dataset')
manifest = root / 'manifest.jsonl'
if not manifest.is_file():
    raise SystemExit(f'Manifest not found: {manifest}')
with manifest.open(encoding='utf-8') as stream:
    samples = [json.loads(line) for line in stream if line.strip()]
bad = [r['path'] for r in samples
       if not (root / r['path']).is_file()
       or (root / r['path']).stat().st_size != r['file_bytes']]
print('Dataset root:', root)
print('Samples:', len(samples), 'missing or size mismatch:', len(bad))
print('\n'.join(bad[:10]))
PY
```

Expect 8,902 samples and zero missing or size mismatches. This checks sizes,
not cryptographic hashes or the contents of every clip.

## 3. Download the 32-frame pretrained weights

On the **login node**, download the official 32-frame Kinetics-400
VideoMamba-Tiny checkpoint into `lab/checkpoints/`:

```bash
project=/project/def-mbolic/yunzelu/VideoMamba/lab
mkdir -p "$project/checkpoints"
weights="$project/checkpoints/videomamba_t16_k400_f32_res224.pth"
test -e "$weights" || curl -fL --retry 3 -o "$weights" \
  'https://huggingface.co/OpenGVLab/VideoMamba/resolve/main/videomamba_t16_k400_f32_res224.pth'
printf '%s  %s\n' \
  '75497b63bfe10850fc2fd879256a2cf4a794c9527ffa472d7b93a3eb85468116' \
  "$weights" | sha256sum -c -
```

The final command must print `OK`. Existing checkpoint files are not
intentionally overwritten. Stop if the download or verification fails.

## 4. Short benchmark with 32 frames

Run from the Rorqual login node:

```bash
cd /project/def-mbolic/yunzelu/VideoMamba/lab
bash -n training/rorqual_job.sh
sbatch training/rorqual_job.sh
```

The `bash -n` command checks the script syntax and normally prints nothing if
successful. Do not submit the job if it reports an error.

If Slurm requires an allocation, use
`sbatch --account=YOUR_ACCOUNT training/rorqual_job.sh`. The script loads
`StdEnv/2023 gcc/12.3 python/3.10 cuda/12.2`, activates
`/project/def-mbolic/yunzelu/VideoMamba/.venv`, and requests one 20 GB H100
MIG instance, four CPU cores, 24 GB host RAM, and two hours. Defaults are 32
frames, batch size 2, and two data-loader workers.

The default mode is a short benchmark: five warmup steps followed by 30 timed
steps with real training clips, forward/backward passes, and optimizer steps.
It prints clips per second and peak PyTorch GPU memory. It does not save a
trained model or evaluate validation/test clips. It excludes validation,
checkpoint saving, setup and queue time from its epoch estimate.

Check the returned job ID and logs:

```bash
squeue -u "$USER"
cat canny_JOBID.out
cat canny_JOBID.err
```

Replace `JOBID` with the number returned by `sbatch`. The earlier GPU smoke test
covered 16 frames. Actual 32-frame VideoMamba training and checkpoint loading
on Rorqual remain untested until this benchmark succeeds.

## 5. Optional 16-frame benchmark

For a matched 16-frame comparison, download its checkpoint on the login node:

```bash
project=/project/def-mbolic/yunzelu/VideoMamba/lab
mkdir -p "$project/checkpoints"
weights="$project/checkpoints/videomamba_t16_k400_f16_res224.pth"
test -e "$weights" || curl -fL --retry 3 -o "$weights" \
  'https://huggingface.co/OpenGVLab/VideoMamba/resolve/main/videomamba_t16_k400_f16_res224.pth'
printf '%s  %s\n' \
  'a335d728ae4dbe4f49a435022f95c6cf98108d20fe084120db1f18cb73e84f4a' \
  "$weights" | sha256sum -c -
```

After the checksum prints `OK`, submit:

```bash
cd /project/def-mbolic/yunzelu/VideoMamba/lab
CANNY_MODE=benchmark CANNY_FRAMES=16 sbatch training/rorqual_job.sh
```

The script chooses the checkpoint matching `CANNY_FRAMES` unless
`CANNY_PRETRAINED` is explicitly set. Of 7,188 training segments, 2,068
(28.8%) contain fewer than 32 source frames, mostly short transitions. The
sampler repeats frames in those segments. The 16- and 32-frame performance,
speed and memory use have not yet been compared on the target hardware.

## 6. Training and separate final test

After checking the benchmark, explicitly set a run directory, the mode, an
epoch count, and a suitable time limit. Thirty epochs is illustrative, not a
claim that this is the right number:

```bash
cd /project/def-mbolic/yunzelu/VideoMamba/lab
export CANNY_OUTPUT="$PWD/runs/experiment_01"
export CANNY_EPOCHS=30
export CANNY_MODE=train
sbatch --time=12:00:00 training/rorqual_job.sh
```

The training pipeline samples 32 frames from each activity segment, resizes to
fit 224 x 224 with antialiased bilinear interpolation, pads with black,
replicates into three channels, and applies ImageNet normalization. Stored edge
frames remain binary, but the resized model inputs can contain gray values.
Training selects `best.pt` using Rose's validation macro-F1 over the fixed ten
classes. Yunze's test clips are held out for the separate evaluation command.

For an interrupted run, use the same output, epoch count, frame count, batch
size, learning rate, checkpoint settings, and pretrained file. Checkpoints are
written at epoch boundaries; an interrupted partial epoch starts again.

```bash
export CANNY_RESUME="$CANNY_OUTPUT/last.pt"
sbatch --time=12:00:00 training/rorqual_job.sh
```

After training is complete, evaluate the selected checkpoint on the test split:

```bash
cd /project/def-mbolic/yunzelu/VideoMamba/lab
export CANNY_MODE=evaluate
export CANNY_CHECKPOINT="$PWD/runs/experiment_01/best.pt"
export CANNY_OUTPUT="$PWD/runs/experiment_01_test"
sbatch training/rorqual_job.sh
```

Environment variables stay set in the current shell. Run the following before
returning to defaults:

```bash
unset CANNY_MODE CANNY_CHECKPOINT CANNY_OUTPUT CANNY_EPOCHS CANNY_RESUME CANNY_FRAMES
```

For a 10 GB instance, override the scheduler resources and batch size, for
example:

```bash
CANNY_MODE=benchmark CANNY_BATCH_SIZE=1 CANNY_WORKERS=0 \
  sbatch --gpus=h100_1g.10gb:1 --cpus-per-task=1 --mem=15G training/rorqual_job.sh
```

The earlier synthetic 16-frame batch-one model test fit in a 10 GB instance.
Real 32-frame training, including the optimizer, has not yet been measured.
