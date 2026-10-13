# Ablation 08 — YOLOv11 specialized baselines

**Question.** What is the **in-domain detection ceiling** on box-native datasets vs general mask transformers?

**Expected outcome.** YOLO **beats** EoMT mask→box on CANSURF/FloW; matches or exceeds cited paper baselines where we retrain.

**Training allowed** on FloW and CANSURF (boxes are valid GT). This is the primary train path for box datasets.

## Setup

| Train set | Recipe | Metrics |
|---|---|---|
| CANSURF raw | Paper/YOLOv11 defaults; **no** 57k aug on disk | val mAP |
| FloW-Img | If available | test mAP, AP_S |
Optional: SAHI on CANSURF/FloW for far-field recall.

## Results

| Run | Dataset | mAP50-95 | mAP50 | Precision | Recall | Slurm | W&B / files |
|---|---|---|---|---|---|---|---|
| YOLO11n 50ep train | CANSURF val | — | **0.966** | 0.978 | 0.924 | **27648816** | Ultralytics local run |
| W&B val replay | CANSURF val | **0.861** | **0.966** | 0.978 | 0.924 | **27660923** | `n3vuqqbi` |
| YOLO11n 50ep train (1080 train / 120 val) | FloW **test** (800) | **0.391** | **0.822** | 0.856 | 0.767 | **27942882** | [`flow_test_val_27942882.log`](sources/flow_test_val_27942882.log); W&B run `flow_train_20261012_153258` |

These are Ultralytics' own metrics. FloW split: `training/` minus a seeded 10% val slice ([`flow_val_split.txt`](sources/flow_val_split.txt)); the official test split is scored only. Recipe identical to CANSURF (imgsz 640, batch 16, 50 epochs) — note FloW images are 1280×720, so 640 halves resolution for the many small objects.

## Results (same scorer as EoMT)

pycocotools via `eval_coco_bbox.py`, same GT files as ablations 04/05 so numbers compare directly with EoMT mask→box. Audit: [`RESCORE.md`](../../RESCORE.md).

| Model | Dataset | mAP | AP50 | AP_small | AR100 | Source |
|---|---|---|---|---|---|---|
| YOLOv11n | FloW test | **0.390** | 0.815 | **0.249** | 0.491 | job 27942882; [`preds`](sources/yolo11n_flow_test_preds_coco.json), [`metrics`](sources/yolo11n_flow_test_preds_coco.metrics.json) |
| YOLOv11n | CANSURF val (aligned) | **0.857** | 0.960 | (19 small GT; n/a) | 0.870 | preds job **28000242**; [`preds`](sources/yolo11n_cansurf_val_preds_coco.json), [`metrics`](sources/yolo11n_cansurf_val_preds_coco.metrics.json) |

The FloW legacy-scorer value (0.566 mAP / 0.858 AP_small, `*.metrics.legacy.json`) is superseded; the mismatch with Ultralytics' 0.391 is what exposed the legacy scorer defects (decision log 2026-10-13).

## Verdict

YOLO trained in-domain is the ceiling on both box sets: **0.857** mAP on CANSURF and **0.390** on FloW, vs best EoMT mask→box **0.095** and **0.082**. FloW is much harder than CANSURF even for YOLO, driven by small objects (AP_small 0.249; 59% of FloW GT boxes are < 32²). A 1280-px FloW run (`IMGSZ=1280 NAME=flow_train_1280 sbatch scripts/snellius/ablation08_yolo_flow.sbatch`) is the obvious follow-up to test whether resolution is the bottleneck.

## Resources copied here

| Original | Copy in this folder | Present |
|---|---|---|
| — | sources/data.yaml | — |

## Code components

- Ultralytics YOLOv11
- CANSURF [train.py](https://github.com/ZaidAljundiHW2/CANSURF)
