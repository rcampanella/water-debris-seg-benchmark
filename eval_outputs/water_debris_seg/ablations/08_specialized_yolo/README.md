# Ablation 08 — YOLOv11 specialized baselines

**Question.** What is the **in-domain detection ceiling** on box-native datasets vs general mask transformers?

**Expected outcome.** YOLO **beats** EoMT mask→box on CANSURF/FloW/WSODD; matches or exceeds cited paper baselines where we retrain.

**Training allowed** on FloW, CANSURF, WSODD (boxes are valid GT). This is the primary train path for box datasets.

## Setup

| Train set | Recipe | Metrics |
|---|---|---|
| CANSURF raw | Paper/YOLOv11 defaults; **no** 57k aug on disk | val mAP |
| FloW-Img | If available | test mAP, AP_S |
| WSODD | rubbish-focused or 14-class | mAP |

Optional: SAHI on CANSURF/FloW for far-field recall.

## Results

| Model | Dataset | mAP | AP_S | Source |
|---|---|---|---|---|
| YOLOv11 | CANSURF | — | — | *pending* |
| YOLOv11 | FloW | — | — | *pending* |
| YOLOv11 | WSODD | — | — | *pending* |

## Verdict

*Pending.*

## Resources copied here

| Original | Copy in this folder | Present |
|---|---|---|
| — | sources/data.yaml | — |

## Code components

- Ultralytics YOLOv11
- CANSURF [train.py](https://github.com/ZaidAljundiHW2/CANSURF)
