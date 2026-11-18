# water-debris-seg — implementation

Benchmark code (adapters, eval helpers, Slurm wrappers). Heavy deps and clones live on scratch.

## Paths (Snellius)

| Variable | Default |
|---|---|
| `WATER_DEBRIS_SCRATCH_ROOT` | `/scratch-shared/$USER/water-debris-seg` |
| `WATER_DEBRIS_DATA_ROOT` | `$SCRATCH_ROOT/data` |
| `WATER_DEBRIS_THIRD_PARTY` | `$SCRATCH_ROOT/third_party` |

Code stays in the repository; datasets and `third_party/` clones go under the scratch root.

## Quick commands

```bash
# GPU sanity (charges SBUs)
bash water-debris-seg/scripts/snellius/run_cuda_smoke.sh

# LaRS minimum path (~1 GB on scratch)
bash water-debris-seg/scripts/download_lars.sh --stills-only

# EoMT env (submit batch job)
sbatch water-debris-seg/scripts/snellius/setup_eomt_venv.sbatch
```

## Depth benchmark pipeline (`scripts/depth/`)

| Step | Script | Slurm |
|---|---|---|
| Ground truth from LiDAR | `build_depth_gt.py`, `summarize_gt_build.py` | `depth_gt_build.sbatch` |
| Splits | `make_standard_split.py`, `make_location_split.py` | — |
| Dataset for Ultralytics | `prepare_depth_dataset.py` | `depth_prepare_dataset.sbatch` |
| Training | `train_yolo26_depth.py` | `depth_train.sbatch` |
| Scoring (zero-shot and fine-tuned) | `eval_depth.py` (fine-tuned: called by `train_yolo26_depth.py` after training) | `depth_eval_zeroshot.sbatch`, `depth_train.sbatch` |
| Latency (TensorRT FP16) | `bench_depth_latency.py` | `depth_latency.sbatch` |
| Tables, LaTeX, plots | `make_d01_tables.py`, `make_depth_section_tex.py`, `plot_d01_latency.py` | — |
| Consistency check | `audit_d01_consistency.py` | — |
