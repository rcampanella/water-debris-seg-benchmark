# Ablation 01 — EoMT zero-shot on LaRS (+ box-set mask→box)

**Question.** How well do **unfine-tuned** COCO/ADE EoMT weights transfer to maritime panoptic/semantic labels and to debris-like detection on box-only sets?

**Expected outcome.** Reasonable water/sky stuff from ADE-like priors; **low** PQ on debris things (Float/Other) and **low** mask→box mAP on FloW/CANSURF/WSODD vs in-domain YOLO (CANSURF paper: ~12× gap).

**Not training.** No `fit` on any dataset in this ablation.

## Setup

| Item | Value |
|---|---|
| Checkpoint | Official EoMT DINOv2 (e.g. COCO panoptic B/L) |
| LaRS eval | val split — PQ / semantic mIoU |
| Box sets | FloW/CANSURF/WSODD val/test — **mask→box mAP** only |
| Class map | Document COCO/ADE → LaRS / waste / can / rubbish |

## Results

| Split | mIoU / PQ | mAP | AP_S | Source |
|---|---|---|---|---|
| LaRS semantic | — | — | — | *pending* |
| LaRS panoptic | — | — | — | *pending* |
| FloW-Img | — | — | — | *pending* |
| CANSURF | — | — | — | *pending* |
| WSODD (rubbish) | — | — | — | *pending* |

## Verdict

*Pending.*

## Resources copied here

| Original | Copy in this folder | Present |
|---|---|---|
| — | sources/ | — |

## Code components

- `third_party/eomt/main.py validate` (future)
- Shared mask→box eval script (future)
