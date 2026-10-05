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
| PQ val | — | — | *pending* |
| PQ by scene attribute | — | — | *pending* |

## Verdict

*Pending.*

## Resources copied here

| Original | Copy in this folder | Present |
|---|---|---|
| — | sources/ | — |

## Code components

- EoMT panoptic yaml (future)
