# Evaluate VideoMamba on retirement-home Canny data

This pipeline samples annotated, single-person intervals, prepares Canny clips,
and evaluates the trained ten-class VideoMamba-Tiny checkpoint. It includes a
local preparation and a GPU Slurm job for inference. It does not retrain the
model. The cluster commands below have not been run as part of writing this
pipeline.

## Interpretation

Annotation rows start intervals that end immediately before the next row. The
`END` row closes the final interval. Data outside these bounds, unannotated
labels, and intervals with zero or multiple people are excluded. By default the
preparation samples centers at 60-second spacing and at least one per eligible
interval, subject to enough valid frames. The results are **clip-level**, not
a time-weighted measure of the full recording. This sampling gives short
intervals more weight relative to their duration than long intervals.

Each clip uses 32 consecutive recorded frames, without duplication. At a nominal
10 fps, the first and last frames are approximately 3.1 seconds apart. Actual
timestamps may differ. The default `pure` mode requires all frames to lie in
the same annotation interval. Short transitions may therefore be absent from the
score. The optional `center` mode labels a window by the activity at its center
and permits adjacent supported, single-person activities in the window. It
records their frame counts and the fraction belonging to the center activity.
Keep the two modes' results separate. In both modes, the default maximum
consecutive-frame gap is 0.25 seconds, and invalid windows are excluded. Check
the preparation summary and clip counts before interpreting the results.

Lab training uniformly sampled 32 frames across each annotated activity
segment, which had a variable duration. Using consecutive frames at 10 fps can
produce a similar spacing for some lab segments, but it does not guarantee the
same temporal distribution. The same spatial resize, black padding, channel
replication, and ImageNet normalization are used at evaluation.

**Prepare new clips for this version.** Previously prepared 11-frame clips for
the repetition-based evaluation are incompatible. The evaluator rejects them.
Use a new preparation output directory and transfer the updated scripts to the
cluster before submitting jobs.

The model produces its original ten classes. For evaluation, `lie-in-bed` and
`lie-on-floor` predictions are merged into `lie`, because the real annotation
does not state the lying location. The model's `sit-to-lie-in-bed` and
`lie-in-bed-to-sit` predictions are compared to the annotation's generic
`sit_to_lie` and `lie_to_sit`. This does not establish where the person is
lying or whether the model can recognize floor-specific transitions. Raw
fine-grained predictions are retained for inspection. The model's ten-class
top prediction is chosen first and then mapped to a coarse class; probabilities
for the two lying classes are not summed before choosing the prediction. Predictions in classes
with no corresponding ground-truth examples are errors when the annotated
class differs. Report per-class results along with overall accuracy and
macro-F1. These results describe the sampled eligible windows, not every hour,
person, or possible activity in the home.

## 1. Transfer the prepared samples to Rorqual

The local archive `D:\lu\project\retirement_home\real_world_samples\real_world_samples_10fps.tar`
contains the prepared center clips and two manifests. The pure manifest points
to the same clips, so each clip only needs to be transferred once. The large
SQLite databases and annotation CSV are not needed on the cluster for evaluation.

In Windows PowerShell:

```powershell
ssh yunzelu@rorqual.alliancecan.ca 'mkdir -p /project/def-mbolic/yunzelu/VideoMamba/lab/real_world /project/def-mbolic/yunzelu/_dataset/retirement_home/real_world_samples'
scp 'D:\lu\project\retirement_home\VideoMamba\lab\real_world\evaluate.py' 'D:\lu\project\retirement_home\VideoMamba\lab\real_world\rorqual_evaluate.sh' 'yunzelu@rorqual.alliancecan.ca:/project/def-mbolic/yunzelu/VideoMamba/lab/real_world/'
scp 'D:\lu\project\retirement_home\real_world_samples\real_world_samples_10fps.tar' 'yunzelu@rorqual.alliancecan.ca:/project/def-mbolic/yunzelu/_dataset/retirement_home/real_world_samples/'
```

After logging in to Rorqual, extract the archive:

```bash
tar -xf /project/def-mbolic/yunzelu/_dataset/retirement_home/real_world_samples/real_world_samples_10fps.tar -C /project/def-mbolic/yunzelu/_dataset/retirement_home/real_world_samples/
```

The cluster should already contain
`/project/def-mbolic/yunzelu/VideoMamba/lab/runs/ddp-train-26347306/best.pt`,
`lab/training/train_canny.py`, `lab/canny_dataset.py`, the VideoMamba source,
and the virtual environment used for training. The lab dataset itself is not
required for inference. Before inference, inspect `center/summary.json` and `center/pure_summary.json`
and verify the source coverage, clip counts, and exclusions.

## 2. Submit the GPU inference jobs

Log in to Rorqual and submit the two evaluations separately:

```bash
cd /project/def-mbolic/yunzelu/VideoMamba/lab
bash -n real_world/rorqual_evaluate.sh
export REAL_MANIFEST=/project/def-mbolic/yunzelu/_dataset/retirement_home/real_world_samples/center/pure_manifest.jsonl
sbatch real_world/rorqual_evaluate.sh

export REAL_MANIFEST=/project/def-mbolic/yunzelu/_dataset/retirement_home/real_world_samples/center/manifest.jsonl
sbatch real_world/rorqual_evaluate.sh
```

Each job requests one 20 GB H100 MIG instance, four CPUs, 24 GB RAM, and two
hours. These are starting estimates, not measured requirements for this data.
The batch size and worker count default to 2 and can be set with
`REAL_BATCH_SIZE` and `REAL_WORKERS`. `REAL_CHECKPOINT` selects a different
checkpoint. Each job defaults to its own `lab/runs/real-world-eval-JOBID/`
output. If setting `REAL_OUTPUT`, choose a different absolute path for each job.
Slurm options can override the requested time and resources.

`predictions.csv` contains individual clip results and `metrics.json` contains
aggregate results. Retain the preparation summaries, predictions, metrics, and
job logs. Read supported-class macro-F1 and per-class support alongside the
fixed-class macro-F1. Keep pure and center results separate, and account for
the different domains and sampling when comparing with lab results.

## Preparing clips again

`prepare.py` can extract clips directly on Windows with NumPy installed. Supply
all relevant SQLite shards with `--sqlite`, the matching CSV with
`--annotations`, a new empty `--output-dir`, and choose
`--window-labeling pure` or `center`. By default, SQLite is opened in read-only
mode with the WAL visible. Keep each database and its sidecars together. Do not
copy only a live SQLite database, since it can omit data in its WAL. Check the
summary for source coverage and exclusions after preparing.

The optional `--immutable-snapshots` flag is only for closed, standalone
snapshots that cannot change while they are read. It applies to every supplied
shard and rejects any shard with a `-wal`, `-shm`, or `-journal` sidecar.
Never use it for an active database: immutable mode ignores the WAL. The flag
cannot determine whether an otherwise sidecar-free database is actively being
modified, so confirm that condition before use.
