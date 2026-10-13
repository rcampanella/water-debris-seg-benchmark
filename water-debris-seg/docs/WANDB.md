# Weights & Biases (wandb)

All **EoMT** train/validate jobs use `WandbLogger` (`project: water-debris-seg`). **YOLO** jobs run `yolo settings wandb=True` and rely on `WANDB_*` from `wandb_env.sh`. **Mask→box eval** uses `wandb.init` in `eval_eomt_cansurf_mask2box.py` / `eval_coco_bbox.py`.

## Setup (once per login)

```bash
mkdir -p ~/.config/water-debris-seg
cat > ~/.config/water-debris-seg/wandb.env <<'EOF'
export WANDB_API_KEY=your_key_here
export WANDB_ENTITY=your_team_optional
export WANDB_PROJECT=water-debris-seg
EOF
chmod 600 ~/.config/water-debris-seg/wandb.env
```

Slurm jobs source `water-debris-seg/scripts/snellius/wandb_env.sh`, which loads that file and sets `WANDB_DIR` on scratch.

If `WANDB_API_KEY` is unset, jobs run with **`WANDB_MODE=offline`** (metrics saved under `WANDB_DIR`; sync later with `wandb sync`).

**Note:** `wandb verify` is for **self-hosted** W&B only. On Snellius cloud (`api.wandb.ai`) it always prints *Cannot run wandb verify against api.wandb.ai* — that is expected, not an auth failure. Check the key with:

```bash
source ~/.config/water-debris-seg/wandb.env
python -c "import wandb; print(wandb.Api().viewer.username)"
```

New API keys look like `wandb_…` (~86 chars). Use **`wandb>=0.20`** in the EoMT venv (0.19.x only accepts legacy 40-char keys).

## Run groups

| Group | Job script |
|---|---|
| `smoke_yolo` | `smoke_yolo_cansurf.sbatch` |
| `smoke_eomt` | `smoke_eomt_lars.sbatch` (add logger overrides if needed) |
| `ablation01_zeroshot_lars` | `ablation01_eomt_lars.sbatch` |
| `ablation03_finetune_lars_panoptic` | `ablation03_eomt_lars.sbatch` |
| `ablation04_eomt_flow_zeroshot` | `ablation04_eomt_flow_zeroshot.sbatch` |
| `ablation04_eomt_flow_larsft` | `ablation04_eomt_flow_larsft.sbatch` |
| `ablation05_eomt_cansurf_zeroshot` | `ablation05_eomt_cansurf.sbatch`, `ablation05_eval_only.sbatch` |
| `ablation05_eomt_cansurf_larsft` | `ablation05_eomt_cansurf_larsft.sbatch` |
| `ablation08_yolo_cansurf` | `ablation08_yolo_cansurf.sbatch`, `ablation08_yolo_wandb_val.sbatch` |
| `ablation09_semantic_cansurf` | `ablation09_semantic_cansurf.sbatch`, `ablation09_semantic_cansurf_min256.sbatch` |
| `ablation09_semantic_flow` | `ablation09_semantic_flow.sbatch` |
| `ablation07_videomt_temporal` | `ablation07_videomt_train.sbatch`, `ablation07_eomt_seq_flicker.sbatch`, `ablation07_eomt_panoptic_flicker.sbatch`, `ablation07_lars_ytvis_export.sbatch`, `ablation07_lars_ytvis_load_smoke.sbatch`, `ablation07_videomt_prep.sbatch` |
| `ablation02_finetune_lars_semantic` | `ablation02_eomt_lars.sbatch`, `ablation02_validate_lars.sbatch` |

All Slurm jobs **source** `wandb_env.sh` (`WANDB_PROJECT=water-debris-seg`, online when `WANDB_API_KEY` is set). EoMT uses `WandbLogger`; YOLO uses `wandb=True` (respects `WANDB_PROJECT` / `WANDB_NAME` / `WANDB_RUN_GROUP`).

Logs are also copied under `eval_outputs/water_debris_seg/` for the benchmark report.
