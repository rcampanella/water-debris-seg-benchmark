# Ablation 02 — EoMT fine-tune LaRS semantic (3-class)

**Question.** Does LaRS-only semantic fine-tuning improve **obstacle / water / sky** mIoU for USV nearshore scenes?

**Expected outcome.** mIoU **↑** vs ablation 01 on val; largest gain on water boundary and glare cases (attribute slices).

**Hard rule.** Training data = **LaRS train semantic masks only**. No FloW/CANSURF.

## Setup

| Item | Value |
|---|---|
| Init | COCO/ADE EoMT checkpoint |
| Flags | `--model.load_ckpt_class_head False` |
| Classes | 0 obstacle, 1 water, 2 sky (LaRS semantic PNG IDs) |
| Batch | 16 total (DDP or grad accum) |

## Results

| Metric | Zero-shot | Fine-tuned | Source |
|---|---|---|---|
| mIoU val | — | **98.0** (`val_iou_all` ≈ 0.980) | W&B fit `mk2w7dr8`, val `duguh1wo` / job 27660680 |
| mIoU by attribute | — | — | *pending* |

**Slurm fit:** 27658486 · **Ckpt:** `epoch=30-step=2356.ckpt` under W&B run `mk2w7dr8`.

## Verdict

Strong LaRS val mIoU; CANSURF transfer via obstacle CC→box is **poor** overall mAP (0.085) — see ablation 09.

## Resources copied here

| Original | Copy in this folder | Present |
|---|---|---|
| fit log | `sources/fit_27658486.log` | yes |

## Code components

- `pack_lars_semantic_zips.py --full`, `lars_semantic.py`, `configs/ablation02_lars_semantic_finetune.yaml`
- Slurm: `ablation02_eomt_lars.sbatch`, `ablation02_validate_lars.sbatch`
- W&B group: `ablation02_finetune_lars_semantic`
