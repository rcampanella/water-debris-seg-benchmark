# water-debris-seg — implementation

Benchmark code (adapters, eval helpers, Slurm wrappers). Heavy deps and clones live on scratch.

## Paths (Snellius)

| Variable | Default |
|---|---|
| `WATER_DEBRIS_SCRATCH_ROOT` | `/scratch-shared/$USER/water-debris-seg` |
| `WATER_DEBRIS_DATA_ROOT` | `$SCRATCH_ROOT/data` |
| `WATER_DEBRIS_THIRD_PARTY` | `$SCRATCH_ROOT/third_party` |

Repo stays in home; datasets and `third_party/` clones are under scratch (see [`data/water_debris/DISK_BUDGET.md`](../data/water_debris/DISK_BUDGET.md)).

## Prep sprint checklist

- [x] `third_party/{eomt,videomt,ultralytics}` cloned on scratch (SHAs in `sources/third_party_pins.json`)
- [x] Module PyTorch CUDA smoke (`scripts/snellius/run_cuda_smoke.sh`)
- [x] EoMT venv — `/scratch-shared/.../envs/eomt` (`torch 2.7.0+cu126`, job **27590120**)
- [x] LaRS stills + annotations + `verify_lars.py` (198 val images)
- [x] CANSURF Zenodo v2 — `download_cansurf.sh` (train/val YOLO on scratch)
- [x] Wave 2 smoke — YOLO **27591918**; EoMT **27594939** (pipeline OK)
- [ ] Ablation 01 — `sbatch scripts/snellius/ablation01_eomt_lars.sbatch` (+ wandb: `docs/WANDB.md`)
- [ ] LaRS sequences — `bash scripts/download_lars_sequences.sh` (VidEoMT / ablation 07)
- [ ] FloW — ORCA portal → layout in `docs/FLOW_DATA.md`; then `ablation04_*` sbatch
- [x] WSODD — **out of scope v1** (ablation 06 cancelled; see decision log)

## Quick commands

```bash
# GPU sanity (charges SBUs)
bash water-debris-seg/scripts/snellius/run_cuda_smoke.sh

# LaRS minimum path (~1 GB on scratch)
bash water-debris-seg/scripts/download_lars.sh --stills-only

# EoMT env (submit batch job)
sbatch water-debris-seg/scripts/snellius/setup_eomt_venv.sbatch
```
