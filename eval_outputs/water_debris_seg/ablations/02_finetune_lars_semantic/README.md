# Ablation 02 — EoMT fine-tune LaRS semantic (3-class)

**Question.** Does LaRS-only semantic fine-tuning improve **obstacle / water / sky** mIoU for USV nearshore scenes?

**Expected outcome.** mIoU **↑** vs ablation 01 on val; largest gain on water boundary and glare cases (attribute slices).

**Hard rule.** Training data = **LaRS train semantic masks only**. No FloW/CANSURF/WSODD.

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
| mIoU val | — | — | *pending* |
| mIoU by attribute | — | — | *pending* |

## Verdict

*Pending.*

## Resources copied here

| Original | Copy in this folder | Present |
|---|---|---|
| — | sources/ | — |

## Code components

- EoMT config yaml (future)
- W&B / Slurm job id (future)
