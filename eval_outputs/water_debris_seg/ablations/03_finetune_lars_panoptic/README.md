# Ablation 03 — EoMT fine-tune LaRS panoptic (11-class)

**Question.** Can panoptic fine-tuning on LaRS separate **debris-like things** (Float, Other) from **dynamic obstacles** (Boat, Buoy, Swimmer)?

**Expected outcome.** PQ **↑** vs ablation 01; thing AP on Buoy/Boat better than zero-shot; still may trail YOLO on tiny cans (cross-dataset in 09).

**Hard rule.** LaRS panoptic train **only**.

## Setup

| Item | Value |
|---|---|
| Init | Ablation 02 ckpt or COCO panoptic EoMT |
| Annotations | `panoptic_annotations.json` + `panoptic_masks/` |
| Metrics | PQ, SQ, RQ |

## Results

| Metric | Zero-shot | Fine-tuned | Source |
|---|---|---|---|
| PQ val | ≈ 0.0014 (ablation 01 COCO-bridge) | **≈ 0.395** (`val_pq_all`, 11-class) | W&B fit `67kk33hk`, val `qmi79nsh`, re-val `xcl7zvm8` (27660922) |
| PQ by scene attribute | — | — | *planned* (`lars_val_attributes.json` for stratification) |

## Verdict

*Pending.*

## Resources copied here

| Original | Copy in this folder | Present |
|---|---|---|
| — | sources/ | — |

## Code components

- Config: `water-debris-seg/configs/ablation03_lars_panoptic_finetune.yaml`
- Slurm: `ablation03_eomt_lars.sbatch` (job **27607195**)
- W&B group: `ablation03_finetune_lars_panoptic`
