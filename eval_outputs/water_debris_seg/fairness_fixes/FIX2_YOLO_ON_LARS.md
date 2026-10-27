# Fix 2 — Train YOLO on LaRS (boxes traced from the masks) and score it on LaRS

> **Results (2026-10-26):** all runs of this fix are done; values with links to their JSON are in [RESULTS.md](RESULTS.md). ⏳ cells below are the plan as written before the runs.

**Status: box AP done (all runs finished 2026-10-21); YOLOv11s + SAM (PQ, mIoU) planned; cross-domain row not started. Decisions of 2026-10-23: YOLOv11s is the YOLO in the comparison; YOLOv11n results below stay as references.** Back to the summary: [README.md](README.md).

## Motivation

EoMT and VidEoMT were trained on LaRS; YOLO never was. Without a LaRS-trained YOLO, the two model families never meet on the same training data. LaRS has masks, not boxes, so YOLO is trained on boxes traced from the instance masks (min/max per axis), checked to match LaRS's own stored boxes exactly (0 mismatches over 2,803 images).

## What was done (ablation 10)

Full configuration and decisions: [TRAINING.md](../ablations/10_yolo_lars_boxes/TRAINING.md); results: [README.md](../ablations/10_yolo_lars_boxes/README.md).

- 9 classes (Static Obstacle + 8 object classes, Float separate; water/sky dropped), 640 px.
- Held-out protocol: stage A on 90% of train sequences picks the epoch count on the other 10%; stage B retrains on the full train set; LaRS val scored once.
- Full fine-tuning and LoRA checkpoints; focal-loss and follow-up variants.

### Results so far (LaRS val, pycocotools, stage-B `last.pt`)

| Run | mAP 8 obj | mAP 9 cls | AP_small obj | AR100 obj | Eval JSON | W&B (stage B) |
|---|---|---|---|---|---|---|
| YOLOv11n full, BCE | 0.145 | 0.168 | 0.054 | 0.247 | [`eval_lars_full_stageB_28132656.json`](../ablations/10_yolo_lars_boxes/sources/eval_lars_full_stageB_28132656.json) | [lars_full_stageB](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_stageB_20261016_130702) |
| YOLOv11n LoRA r8, BCE | 0.127 | 0.153 | 0.033 | 0.235 | [`eval_lars_lora_stageB_28132657.json`](../ablations/10_yolo_lars_boxes/sources/eval_lars_lora_stageB_28132657.json) | [lars_lora_stageB](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_stageB_20261016_131411) |
| YOLOv11n full, focal α 0.25 | 0.129 | 0.158 | 0.038 | 0.230 | [`eval_lars_full_focal_stageB_28140699.json`](../ablations/10_yolo_lars_boxes/sources/eval_lars_full_focal_stageB_28140699.json) | [lars_full_focal_stageB](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_focal_stageB_20261016_145813) |
| YOLOv11n LoRA r8, focal α 0.25 | 0.124 | 0.154 | 0.030 | 0.230 | [`eval_lars_lora_focal_stageB_28140700.json`](../ablations/10_yolo_lars_boxes/sources/eval_lars_lora_focal_stageB_28140700.json) | [lars_lora_focal_stageB](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_focal_stageB_20261016_150405) |
| **YOLOv11s full, BCE** | **0.176** | **0.195** | **0.062** | **0.251** | [`eval_lars_full_11s_stageB_28378811.json`](../ablations/10_yolo_lars_boxes/sources/eval_lars_full_11s_stageB_28378811.json) | [lars_full_11s_stageB](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_11s_stageB_20261020_170700) |
| YOLOv11n full, focal α 0.75 | 0.143 | 0.170 | 0.046 | 0.237 | [`eval_lars_full_focal_a075_stageB_28378812.json`](../ablations/10_yolo_lars_boxes/sources/eval_lars_full_focal_a075_stageB_28378812.json) | [lars_full_focal_a075_stageB](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_focal_a075_stageB_20261020_165813) |
| YOLOv11n LoRA r8, lr ×10 | 0.110 | 0.139 | 0.038 | 0.207 | [`eval_lars_lora_lr10_stageB_28378809.json`](../ablations/10_yolo_lars_boxes/sources/eval_lars_lora_lr10_stageB_28378809.json) | [lars_lora_lr10_stageB](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_lr10_stageB_20261020_165808) |
| YOLOv11n LoRA **r32**, BCE | 0.136 | 0.161 | 0.043 | 0.240 | [`eval_lars_lora_r32_stageB_28409804.json`](../ablations/10_yolo_lars_boxes/sources/eval_lars_lora_r32_stageB_28409804.json) | [lars_lora_r32_stageB](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_lora_r32_stageB_20261021_115911) |

W&B group: [`ablation10_yolo_lars`](https://wandb.ai/riccardo-campanella/water-debris-seg/groups/ablation10_yolo_lars) (stage-A runs, held-out curves, `train/total_loss`, `val/total_loss`). Related studies: [FOCAL_STUDY.md](../ablations/10_yolo_lars_boxes/FOCAL_STUDY.md), [TRAINING_FOLLOWUP.md](../ablations/10_yolo_lars_boxes/TRAINING_FOLLOWUP.md).

## Open item A — PQ and mIoU for YOLO: YOLO + SAM with water/sky (decided 2026-10-21)

A box detector cannot be scored with PQ or mIoU: both are pixel metrics, and YOLO predicts neither object outlines nor water/sky. **Decision: build a YOLO + SAM pipeline** that produces a full LaRS panoptic map. It is reported as **its own row** ("YOLOv11s + SAM"), because the masks come from SAM, a frozen foundation model much larger than EoMT; the plain YOLOv11s row keeps its box-only results. (Built for YOLOv11s only, decision 2026-10-23.)

### Method

| Step | What | Detail |
|---|---|---|
| 1. Labels | Boxes for **11 classes**: the current 9 + **Water** + **Sky** | Traced from the LaRS masks exactly like the other classes (min/max per segment), with the same box check |
| 2. Train YOLO (only YOLO is trained) | **YOLOv11s**, 11 classes | Same held-out protocol (stage A on train90 → epoch count; stage B on train_all, `last.pt`) and same hyper-parameters as ablation 10 |
| 3. SAM masks (frozen) | Each predicted box above a confidence threshold prompts SAM | Same SAM model and version as fix 3 (⏳ to confirm: SAM 2.1 large); confidence threshold chosen with the **stage-A model on held10** (held out for that model), then applied to the stage-B model; never tuned on LaRS val |
| 4. Panoptic map | Overlapping masks merged into one segment per pixel | Paint order: objects (highest score first), then Static Obstacle, then Water, then Sky; pixels claimed by nobody = void; a segment that keeps less than the EoMT overlap fraction of its mask is dropped, mirroring EoMT's `to_per_pixel_preds_panoptic` |
| 5. Score | Same scorers as EoMT | **PQ**: torchmetrics PQ via EoMT's `update_metrics_panoptic` (as `eval_eomt_lars_panoptic_pq.py`); **mIoU**: merge to obstacle / water / sky with the fix 1 mIoU scorer; **box AP**: the 11-class YOLO's boxes, Water and Sky excluded, on the same 8 / 9-class GT |

### Pilot check (on held10, before scoring LaRS val)

| Check | Measure | Stop rule (proposed) |
|---|---|---|
| Water and sky masks | IoU of SAM's water / sky mask with the GT region, held10 images | median IoU < 0.8 for either: stop and report (SAM segmenting a boat or the horizon band instead of the region) |
| Object masks | IoU of SAM masks with GT instance masks for matched boxes, by size | reported by size; no stop rule |
| Adding water/sky to YOLO | object box AP of the 11-class YOLO vs the 9-class YOLO on held10 | > 0.01 drop: reported next to the results |
| Visual | 12 held10 images: GT, YOLO boxes, SAM masks, panoptic map | rendered for review |

### Outputs and references

| Output | Path / W&B |
|---|---|
| 11-class labels and box check | `/scratch-shared/.../lars/yolo_boxes_11cls/`, `fairness_fixes/sources/fix2_lars_11cls_label_stats.json` |
| YOLO training runs | `ablations/10_yolo_lars_boxes/runs/lars_full_11s_11cls_stage{A,B}/` (YOLOv11s); W&B ⏳ group `ablation10_yolo_lars` |
| Pilot report | `fairness_fixes/FIX2_YOLO_SAM_PILOT.md`, `fairness_fixes/yolo_sam_pilot/*.png` |
| LaRS val PQ, mIoU, box AP | ⏳ `fairness_fixes/sources/fix2_yolo11s_sam_lars_val.json` (checkpoint, SAM weights and GT SHA-256); W&B ⏳ group `fix2_yolo_lars` |

### Caveats

- **Hybrid system:** mask quality is mostly SAM's. The row answers "how far do YOLO's boxes plus a foundation model get?", not "is YOLO a better segmenter than EoMT?".
- **One operating point:** PQ needs a fixed confidence threshold; it is chosen with the stage-A model on held10 and reported. (The EoMT + SAM control in [fix 1](FIX1_LARS_SPLIT_METRICS.md) needs no threshold: it keeps EoMT's own segments.)
- **Water/sky as boxes** cover most of the image; the pilot decides whether SAM segments the whole region.
- **Compute:** SAM adds a large model at inference; runtime is reported next to EoMT's. Training cost: YOLOv11s ~1–1.5 h (stage A + B).

Alternative considered and not chosen: YOLOv11n-seg trained on the masks (masks from YOLO itself, coarser at 1/4 resolution).

## Open item B — LaRS-trained YOLO on FloW and CANSURF (cross-domain row)

Ablation 09 scores LaRS-trained EoMT on FloW and CANSURF; no LaRS-trained YOLO has been scored there. Scoring the ablation 10 checkpoints on the same test sets gives the symmetric cross-domain row, without training.

| Item | Value |
|---|---|
| Checkpoints | YOLOv11s full BCE (`lars_full_11s_stageB/weights/last.pt`) |
| Test sets | FloW official test (800 images) and CANSURF val, aligned GT (2,209 images): the same GT files as ablations 04, 05, 08, 09 |
| Class mapping | FloW and CANSURF have one class. **Main number: debris-only** (Float + Other, the taxonomy's debris classes in [REPORT.md](../REPORT.md)), decision 2026-10-23. **Reference: class-agnostic** (every predicted object box of the 8 object classes; Static Obstacle excluded). Same rule as EoMT in fix 1b. Every prediction keeps its original class. |
| Scorer | `eval_coco_bbox.py` (pycocotools) |
| Outputs | `fairness_fixes/sources/fix2_yolo_lars_<model>_{flow,cansurf}.json` (+ `.metrics.json`) |
| W&B | ⏳ group `fix2_yolo_lars` (LaRS training runs stay in `ablation10_yolo_lars`) |
| Cost | inference only, < 30 min |

## Expected final state of fix 2

YOLOv11n (0.145 on LaRS) and the LoRA / focal variants stay as references in the results table above and in ablation 10. **LoRA is discarded for YOLO** (decision 2026-10-23); LoRA is compared for EoMT only ([fix 4](FIX4_EOMT_LORA.md)).

| Model (trained on LaRS) | LaRS PQ | LaRS mIoU | LaRS box AP 8 obj | FloW box AP | CANSURF box AP |
|---|---|---|---|---|---|
| YOLOv11s full | not needed (see YOLOv11s + SAM) | not needed (see YOLOv11s + SAM) | 0.176 | ⏳ debris-only | ⏳ debris-only |
| YOLOv11s + SAM (11 classes incl. water/sky) | ⏳ → SAM (eval) | ⏳ → SAM (eval) | ⏳ | not needed | not needed |
| EoMT panoptic FT (for reference) | 0.395 | ⏳ fix 1 | ⏳ fix 1 | ~~0.014~~ ⏳ fix 1b | ~~0.034~~ ⏳ fix 1b |

## Caveats

- Single seed per run; differences under ~0.01 mAP are not conclusive.
- Training code inherits Ultralytics 8.4.173, whose assigner already enlarges tiny boxes (YOLO26's "STAL"); not the original YOLOv11 recipe. To be noted in TRAINING.md.
- Float has 3 val boxes on LaRS: not measurable there.
