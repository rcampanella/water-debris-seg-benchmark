# Fair model comparison: three fixes (summary, for audit)

**Status: decisions agreed 2026-10-23; results pending.** Fixes 1, 1b, 3 and 4 have not been run; fix 2's LaRS training is done. Each fix has its own file with motivation, method, expected outputs and references. Points to raise with Giacomo: [NOTES_FOR_GIACOMO.md](NOTES_FOR_GIACOMO.md).

## The problem (raised after the last presentation)

1. **Unequal training across datasets.** On FloW and CANSURF, YOLO was trained on each dataset ([ablation 08](../ablations/08_specialized_yolo/README.md)), while EoMT and VidEoMT were trained only on LaRS and tested there zero-shot ([04](../ablations/04_flow_mask2box/README.md), [05](../ablations/05_cansurf_small_object/README.md), [09](../ablations/09_cross_dataset_transfer/README.md)). That compares a trained detector with an untrained segmenter.
2. **Different LaRS metrics in one column.** The REPORT's "LaRS PQ / mIoU" column ([REPORT.md](../REPORT.md), main table) holds mIoU for EoMT semantic (98.0), PQ for EoMT panoptic (≈0.395) and flicker/AP for VidEoMT. Rows cannot be compared.
3. **Found while preparing fix 1: EoMT boxes had a constant confidence.** Ablations 04, 05 and 09 gave every EoMT box score 0.5, which understates EoMT's box AP against YOLO's ranked detections.

## The fixes

| Fix | What changes | File | Status |
|---|---|---|---|
| **1** | LaRS reported in **separate columns** (PQ, mIoU, box AP; temporal metrics apart). The LaRS-trained **EoMT panoptic model fills all three** from the same predictions: PQ exists; mIoU and box AP are new, box AP on the same boxes as YOLO, with real confidence scores. Plus an **EoMT panoptic + SAM control** row | [FIX1_LARS_SPLIT_METRICS.md](FIX1_LARS_SPLIT_METRICS.md) | Not started |
| **1b** | **Re-score EoMT box AP on FloW and CANSURF** with real confidence scores (problem 3), zero-shot and panoptic checkpoints; debris-only classes as the main number, class-agnostic as reference | [FIX1_LARS_SPLIT_METRICS.md § 5](FIX1_LARS_SPLIT_METRICS.md#5-consequence-for-flow-and-cansurf-ablations-04-05-09) | Not started |
| **2** | **YOLOv11s trained on LaRS** boxes traced from the masks; box AP on LaRS val. **YOLOv11s + SAM** (YOLO retrained with water/sky boxes, frozen SAM turns boxes into masks) for LaRS PQ and mIoU. LaRS-trained YOLOv11s scored on FloW/CANSURF, debris-only (cross-domain row, mirrors ablation 09) | [FIX2_YOLO_ON_LARS.md](FIX2_YOLO_ON_LARS.md) | LaRS box AP **done** (all 9 runs); YOLO + SAM planned; cross-domain not started |
| **4** | **EoMT full fine-tuning vs LoRA** on LaRS, FloW and CANSURF (same held-out protocol for both arms; EoMT's attention-mask annealing scaled to the run length); LoRA not used for YOLO | [FIX4_EOMT_LORA.md](FIX4_EOMT_LORA.md) | Not started |
| **3** | **EoMT trained on FloW and CANSURF** using SAM masks prompted by the ground-truth boxes; **YOLOv11s** retrained on the native boxes with the same held-out protocol; both scored on the native boxes (detection only) | [FIX3_SAM_BOX_DATASETS.md](FIX3_SAM_BOX_DATASETS.md) | Not started; **reverses the REPORT's "hard rule"**, needs a decision-log entry |

## Decisions agreed (2026-10-23)

| Question | Decision | Raise with Giacomo |
|---|---|---|
| YOLO has no masks: YOLO-seg or YOLO + SAM for LaRS PQ / mIoU? | **YOLO + SAM** (no extra model type to train; keeps the benchmark coherent) | yes |
| YOLOv11n and YOLOv11s? | **YOLOv11s only** in the comparison (matches the CANSURF paper, `yolo11s.pt`); YOLOv11n results stay as a reference in ablations 08 and 10 | yes: nano would add model-scale and edge-device data |
| YOLO26 instead of YOLOv11s? | **No**, YOLOv11s stays (CANSURF comparability; v11 uses Distribution Focal Loss, our code and results are on v11). YOLO26s is a possible later reference row | yes |
| Keep EoMT semantic fine-tuning? | **Dropped from the main table**; EoMT panoptic gives PQ, mIoU and box AP. Ablation 02 kept as a reference. Condition: panoptic-derived mIoU within ~1 point of 98.0 (fix 1) | yes |
| Keep VidEoMT? | **Yes, in a separate temporal table**: it needs LaRS video sequences and outputs per-keyframe mask AP and flicker, not PQ / mIoU / box AP | no |
| Classes on FloW / CANSURF | **Debris-only** (D-LaRS, D-COCO) as the main number, **class-agnostic** as reference; original class kept on every prediction | no |
| COCO → LaRS mapping (zero-shot) | **Tightened** to maritime-plausible classes: land animals, **railroad and playingfield dropped** ([COVERAGE_CLASSES.md § 6](COVERAGE_CLASSES.md#6-classes-under-the-agreed-decisions-2026-10-23)) | no |
| LoRA vs full fine-tuning | **Compared for EoMT only** (LaRS, FloW, CANSURF; [fix 4](FIX4_EOMT_LORA.md)); **LoRA discarded for YOLO** (ablation 10's YOLO LoRA runs stay as references) | no |
| Timeline | **Runs parallelised** in waves ([run schedule](#run-schedule-parallel)); limits flagged there | no |

Still open: YOLO zero-shot (never tested; inference only), the exact SAM version, EoMT's starting weights for fix 3, CANSURF held-out split.

## Main comparison table (decisions enforced)

Every value links to the JSON it comes from (filled 2026-10-26, waves 1–3). Each cell also names the **classes scored** (codes below).

| Model | Trained on | Predicts | LaRS PQ | LaRS mIoU | LaRS box AP (8 obj / 9 cls) | FloW box AP | CANSURF box AP |
|---|---|---|---|---|---|---|---|
| EoMT zero-shot | COCO | COCO-133 | [0.298](sources/fix1_coco133_lars_map-tightened_28511590.json) · L11 via map | [93.1](sources/fix1_coco133_lars_map-tightened_28511590.json) · L3 | [0.023 / 0.070](sources/fix1_coco133_lars_map-tightened_28511590.json) mask → min/max · L8 / L9 via map | [0.071 (0.076)](sources/fix1_coco133_flow_map-tightened_28511592.json) mask → min/max · D-COCO | [0.316 (0.305)](sources/fix1_coco133_cansurf_map-tightened_28511593.json) mask → min/max · D-COCO |
| **EoMT panoptic** (full fine-tuning) | LaRS | L11 | [**0.405**](sources/fix1_lars_panoptic_lars_eomt_full_28511607.json) (ablation 03: [0.395](../ablations/03_finetune_lars_panoptic/README.md)) · L11 | [97.1](sources/fix1_lars_panoptic_lars_eomt_full_28511607.json) · L3 | [0.134 / 0.171](sources/fix1_lars_panoptic_lars_eomt_full_28511607.json) mask → min/max · L8 / L9 | [0.137 (0.092)](sources/fix1_lars_panoptic_flow_eomt_full_28547310.json) mask → min/max · D-LaRS *(cross)* | [0.090 (0.097)](sources/fix1_lars_panoptic_cansurf_eomt_full_28547312.json) mask → min/max · D-LaRS *(cross)* |
| EoMT panoptic **LoRA** | LaRS | L11 | [0.379](sources/fix1_lars_panoptic_lars_eomt_lora_28511609.json) · L11 | [97.2](sources/fix1_lars_panoptic_lars_eomt_lora_28511609.json) · L3 | [0.132 / 0.168](sources/fix1_lars_panoptic_lars_eomt_lora_28511609.json) mask → min/max · L8 / L9 | [0.049 (0.090)](sources/fix1_lars_panoptic_flow_eomt_lora_28547313.json) mask → min/max · D-LaRS *(cross)* | [0.023 (0.161)](sources/fix1_lars_panoptic_cansurf_eomt_lora_28547314.json) mask → min/max · D-LaRS *(cross)* |
| EoMT panoptic + SAM (control) | LaRS | L11 | [0.343](sources/sam_panoptic_eomt_val_control_28514487.json) → SAM (eval) · L11 | [70.2](sources/sam_panoptic_eomt_val_control_28514487.json) → SAM (eval) · L3 | not needed ([same boxes as EoMT panoptic](FIX1_LARS_SPLIT_METRICS.md#control-eomt-panoptic--sam-masks-replaced-by-sam)) | not needed | not needed |
| EoMT (SAM masks, full fine-tuning) | FloW | waste | not needed | not needed | not needed | [**0.236**](sources/fix1_box_instance_flow_eomt_full_28547302.json) mask → min/max, SAM (train) · waste | not needed |
| EoMT (SAM masks, **LoRA**) | FloW | waste | not needed | not needed | not needed | [0.220](sources/fix1_box_instance_flow_eomt_lora_28547305.json) mask → min/max, SAM (train) · waste | not needed |
| EoMT (SAM masks, full fine-tuning) | CANSURF | can | not needed | not needed | not needed | not needed | [**0.631**](sources/fix1_box_instance_cansurf_eomt_full_28547307.json) mask → min/max, SAM (train) · can |
| EoMT (SAM masks, **LoRA**) | CANSURF | can | not needed | not needed | not needed | not needed | [0.628](sources/fix1_box_instance_cansurf_eomt_lora_28547309.json) mask → min/max, SAM (train) · can |
| **YOLOv11s** | LaRS | L9 | not needed ([→ YOLOv11s + SAM](FIX2_YOLO_ON_LARS.md#open-item-a--pq-and-miou-for-yolo-yolo--sam-with-watersky-decided-2026-10-21)) | not needed ([→ YOLOv11s + SAM](FIX2_YOLO_ON_LARS.md#open-item-a--pq-and-miou-for-yolo-yolo--sam-with-watersky-decided-2026-10-21)) | [**0.176 / 0.195**](../ablations/10_yolo_lars_boxes/README.md) · L8 / L9 | [0.033 (0.073)](sources/fix2_yolo11s_lars_on_flow_28511596.json) · D-LaRS *(cross)* | [0.006 (0.023)](sources/fix2_yolo11s_lars_on_cansurf_28511597.json) · D-LaRS *(cross)* |
| YOLOv11s + SAM (threshold 0.20) | LaRS | L11 (L9 + Water, Sky) | [0.346](sources/sam_panoptic_yolo_val_final_28515871.json) → SAM (eval) · L11 | [68.0](sources/sam_panoptic_yolo_val_final_28515871.json) → SAM (eval) · L3 | [0.175 / 0.196](../ablations/10_yolo_lars_boxes/sources/eval_lars_full_11s_11cls_stageB_28515717.json) · L8 / L9 | not needed | not needed |
| **YOLOv11s** | FloW | waste | not needed | not needed | not needed | [**0.401**](sources/eval_flow_full_11s_stageB_28511601.json) · waste | not needed |
| **YOLOv11s** | CANSURF | can | not needed | not needed | not needed | not needed | [**0.860**](sources/eval_cansurf_full_11s_stageB_28511603.json) · can |

FloW / CANSURF in-domain rows: EoMT is trained on SAM masks whose boxes are tighter than the GT boxes; SAM-mask boxes alone score [0.461 / 0.601](sources/sam_box_ceiling_ap.json) box AP against GT (AP50 0.97). Compare with AP50 too: CANSURF EoMT 0.967 vs YOLOv11s 0.952, FloW 0.670 vs 0.827 ([table and finding 11](RESULTS.md#findings)). Results and findings: [RESULTS.md](RESULTS.md).

**Class codes**

| Code | Classes |
|---|---|
| **COCO-133** | 80 COCO objects + 53 COCO regions (full list: [COVERAGE_CLASSES.md § 2](COVERAGE_CLASSES.md#2-classes-each-training-source-can-predict-complete-lists)) |
| **L11** | Static Obstacle, Water, Sky, Boat/ship, Row boats, Paddle board, Buoy, Swimmer, Animal, Float, Other |
| **L9** | Static Obstacle + the 8 objects of L8 |
| **L8** | Boat/ship, Row boats, Paddle board, Buoy, Swimmer, Animal, Float, Other |
| **L3** | obstacle (Static Obstacle + L8 merged), water, sky |
| **D-LaRS** (debris) | Float, Other |
| **D-COCO** (debris-like COCO) | bottle, cup, frisbee, sports ball, kite, backpack, handbag, suitcase, umbrella |
| **waste** / **can** | FloW's single class / CANSURF's single class (GT file name `object`) |
| **via map** | COCO predictions mapped to LaRS with the tightened mapping ([COVERAGE_CLASSES.md § 6](COVERAGE_CLASSES.md#6-classes-under-the-agreed-decisions-2026-10-23)) |

**Legend.** ⏳ = to be produced by the fix named. not needed = covered by another row (named) or a cross-domain direction that is not studied. *(cross)* = tested on a dataset the model was not trained on; compare only with other cross-domain cells. **mask → min/max** = predicted boxes are the min/max of predicted masks (EoMT); untagged box cells use boxes predicted directly (YOLO). **→ SAM (eval)** = the masks scored are produced by SAM at evaluation. **SAM (train)** = the model was trained on SAM pseudo-masks; the scored masks are its own.

**Notes.**
- **Box AP** = pycocotools mAP@[.50:.95] on the native box ground truth; on LaRS, the boxes traced from the val masks. FloW / CANSURF cells: **debris-only** is the main number; a **class-agnostic** score (any predicted object counts) is reported next to it.
- **Why YOLO box models cannot give LaRS PQ and mIoU:** masks can be converted to boxes (min/max), boxes cannot be converted to masks. The **YOLOv11s + SAM** row fills them: YOLOv11s is retrained with Water and Sky as extra box classes and frozen SAM turns each predicted box into a mask. It is a separate row because the masks come from SAM, a model much larger than EoMT.
- **EoMT panoptic + SAM (control):** EoMT's own segments with masks replaced by SAM. Against EoMT panoptic it isolates mask quality; against YOLOv11s + SAM it isolates detection (same mask source).
- **Not in this table (kept as references in their pages):** EoMT semantic ([02](../ablations/02_finetune_lars_semantic/README.md); mIoU 98.0 and cross-domain results), YOLOv11n ([08](../ablations/08_specialized_yolo/README.md): FloW 0.390, CANSURF 0.857; [10](../ablations/10_yolo_lars_boxes/README.md): LaRS 0.145 / 0.168), the previous EoMT FloW / CANSURF values scored with a constant confidence ([04](../ablations/04_flow_mask2box/README.md), [05](../ablations/05_cansurf_small_object/README.md), [09](../ablations/09_cross_dataset_transfer/README.md)), and LoRA / focal / follow-up YOLO variants ([10](../ablations/10_yolo_lars_boxes/README.md)).
- **SAM** is the same frozen model in fix 1 (EoMT + SAM control), fix 2 (YOLOv11s + SAM) and fix 3 (pseudo-masks for EoMT training).
- **Ground truth on LaRS** is itself min/max boxes traced from the LaRS masks, for every model.

## Extended runs (epoch budget and learning rate)

Started 2026-10-26 after LaRS LoRA's held-out PQ was still rising at the last of 12 epochs. Same held-out protocol as the main table; only the stage A budget and the base learning rate change (`train_eomt_heldout.py --epochs 24 --lr-mult 5`; EoMT's layer-wise lr decay still applies). The main comparison table above keeps the agreed 12-epoch, lr 1e-4 runs until this check is complete.

**Selection rule (fixed before the results):** for each dataset and arm, the reported configuration is the one with the best **stage A** held-out metric (LaRS: PQ on held10; FloW / CANSURF: mask AP on the SAM pseudo-masks of the held-out split). Stage B runs of the other configurations are references only, never chosen by their val / test score. Full fine-tuning and LoRA get the same three configurations, so the arms stay comparable.

| Dataset | Arm | Epoch budget | lr | Stage A held-out (best epoch) | Stage B result | LaRS → FloW / CANSURF box AP *(cross)* | W&B training run | Slurm A → B |
|---|---|---|---|---|---|---|---|---|
| LaRS | full | 12 (baseline) | 1e-4 | 0.404 (ep 12) ([json](sources/epoch_selection_eomt_lars_full.json)) | PQ 0.405 · things 0.204 · mIoU 97.1 · box 0.134 / 0.171 ([json](sources/fix1_lars_panoptic_lars_eomt_full_28511607.json)) | 0.137 (0.092) ([json](sources/fix1_lars_panoptic_flow_eomt_full_28547310.json)) / 0.090 (0.097) ([json](sources/fix1_lars_panoptic_cansurf_eomt_full_28547312.json)) | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/yphvg0cr) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/zxokqelg) | 28511606 → 28511607 |
| LaRS | full | 24 | 1e-4 | 0.429 (ep 21) · **selected** ([json](sources/epoch_selection_eomt_lars_full_e24.json)) | PQ 0.407 · things 0.208 · mIoU 97.2 · box 0.129 / 0.165 ([json](sources/fix1_lars_panoptic_lars_eomt_full_e24_28615356.json)) | not run | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/jw2pwsyh) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/jynrh5x0) | 28615355 → 28615356 |
| LaRS | full | 24 | 5e-4 (×5) | 0.323 (ep 24) ([json](sources/epoch_selection_eomt_lars_full_e24_lr5.json)) | PQ 0.309 · things 0.078 · mIoU 96.2 · box 0.042 / 0.084 ([json](sources/fix1_lars_panoptic_lars_eomt_full_e24_lr5_28615358.json)) | not run | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/wmxp9yaz) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/hjlsipnv) | 28615357 → 28615358 |
| LaRS | LoRA | 12 (baseline) | 1e-4 | 0.369 (ep 12) ([json](sources/epoch_selection_eomt_lars_lora.json)) | PQ 0.379 · things 0.170 · mIoU 97.2 · box 0.132 / 0.168 ([json](sources/fix1_lars_panoptic_lars_eomt_lora_28511609.json)) | 0.049 (0.090) ([json](sources/fix1_lars_panoptic_flow_eomt_lora_28547313.json)) / 0.023 (0.161) ([json](sources/fix1_lars_panoptic_cansurf_eomt_lora_28547314.json)) | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/8xdisbps) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/7wrw1rkt) | 28511608 → 28511609 |
| LaRS | LoRA | 24 | 1e-4 | 0.404 (ep 22) ([json](sources/epoch_selection_eomt_lars_lora_e24.json)) | PQ 0.399 · things 0.194 · mIoU 97.2 · box 0.144 / 0.179 ([json](sources/fix1_lars_panoptic_lars_eomt_lora_e24_28592561.json)) | 0.087 (0.091) ([json](sources/fix1_lars_panoptic_flow_eomt_lora_e24_28615359.json)) / 0.049 (0.138) ([json](sources/fix1_lars_panoptic_cansurf_eomt_lora_e24_28615360.json)) | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/2zrz9zf0) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/suvji5e1) | 28590790 → 28592561 |
| LaRS | LoRA | 24 | 5e-4 (×5) | **0.452** (ep 23) · **selected** ([json](sources/epoch_selection_eomt_lars_lora_e24_lr5.json)) | PQ **0.424** · things 0.231 · mIoU 97.3 · box 0.152 / 0.186 ([json](sources/fix1_lars_panoptic_lars_eomt_lora_e24_lr5_28592562.json)) | 0.136 (0.103) ([json](sources/fix1_lars_panoptic_flow_eomt_lora_e24_lr5_28615361.json)) / 0.089 (0.081) ([json](sources/fix1_lars_panoptic_cansurf_eomt_lora_e24_lr5_28615362.json)) | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/qolmyo20) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/5s869fcq) | 28590791 → 28592562 |
| FloW | full | 12 (baseline) | 1e-4 | 0.487 (ep 12) ([json](sources/epoch_selection_eomt_flow_full.json)) | box AP 0.236 ([json](sources/fix1_box_instance_flow_eomt_full_28547302.json)) | not needed | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/whcbnw3e) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/tdxrxg6w) | 28547280 → 28547302 |
| FloW | full | 24 | 1e-4 | 0.515 (ep 24) · **selected** ([json](sources/epoch_selection_eomt_flow_full_e24.json)) | box AP 0.256 ([json](sources/fix1_box_instance_flow_eomt_full_e24_28615364.json)) | not needed | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/vneqafov) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/064twh5d) | 28615363 → 28615364 |
| FloW | full | 24 | 5e-4 (×5) | 0.237 (ep 24) ([json](sources/epoch_selection_eomt_flow_full_e24_lr5.json)) | box AP 0.100 ([json](sources/fix1_box_instance_flow_eomt_full_e24_lr5_28615366.json)) | not needed | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/55uhn5tt) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/d9q5a98a) | 28615365 → 28615366 |
| FloW | LoRA | 12 (baseline) | 1e-4 | 0.478 (ep 11) ([json](sources/epoch_selection_eomt_flow_lora.json)) | box AP 0.220 ([json](sources/fix1_box_instance_flow_eomt_lora_28547305.json)) | not needed | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/eg9pabqv) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/5oz1099z) | 28547303 → 28547305 |
| FloW | LoRA | 24 | 1e-4 | 0.511 (ep 23) · **selected** ([json](sources/epoch_selection_eomt_flow_lora_e24.json)) | box AP 0.247 ([json](sources/fix1_box_instance_flow_eomt_lora_e24_28615368.json)) | not needed | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/zirmyngu) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/mp455ps3) | 28615367 → 28615368 |
| FloW | LoRA | 24 | 5e-4 (×5) | 0.507 (ep 24) ([json](sources/epoch_selection_eomt_flow_lora_e24_lr5.json)) | box AP 0.244 ([json](sources/fix1_box_instance_flow_eomt_lora_e24_lr5_28615370.json)) | not needed | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/go8pptpb) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/7g5r733k) | 28615369 → 28615370 |
| CANSURF | full | 12 (baseline) | 1e-4 | 0.776 (ep 11) ([json](sources/epoch_selection_eomt_cansurf_full.json)) | box AP 0.631 ([json](sources/fix1_box_instance_cansurf_eomt_full_28547307.json)) | not needed | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/8188sdjb) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/p8mi8h0k) | 28547306 → 28547307 |
| CANSURF | full | 24 | 1e-4 | 0.779 (ep 24) · **selected** ([json](sources/epoch_selection_eomt_cansurf_full_e24.json)) | box AP 0.631 ([json](sources/fix1_box_instance_cansurf_eomt_full_e24_28615372.json)) | not needed | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/holxa91r) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/hfrz934o) | 28615371 → 28615372 |
| CANSURF | full | 24 | 5e-4 (×5) | 0.715 (ep 24) ([json](sources/epoch_selection_eomt_cansurf_full_e24_lr5.json)) | box AP 0.578 ([json](sources/fix1_box_instance_cansurf_eomt_full_e24_lr5_28615374.json)) | not needed | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/jtd1oxlv) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/wt28sup2) | 28615373 → 28615374 |
| CANSURF | LoRA | 12 (baseline) | 1e-4 | 0.766 (ep 9) ([json](sources/epoch_selection_eomt_cansurf_lora.json)) | box AP 0.628 ([json](sources/fix1_box_instance_cansurf_eomt_lora_28547309.json)) | not needed | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/ip8at1js) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/75g0ocvd) | 28547308 → 28547309 |
| CANSURF | LoRA | 24 | 1e-4 | 0.777 (ep 24) ([json](sources/epoch_selection_eomt_cansurf_lora_e24.json)) | box AP 0.632 ([json](sources/fix1_box_instance_cansurf_eomt_lora_e24_28615376.json)) | not needed | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/wgrdgcv9) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/mmxwvgbt) | 28615375 → 28615376 |
| CANSURF | LoRA | 24 | 5e-4 (×5) | 0.783 (ep 24) · **selected** ([json](sources/epoch_selection_eomt_cansurf_lora_e24_lr5.json)) | box AP 0.635 ([json](sources/fix1_box_instance_cansurf_eomt_lora_e24_lr5_28615378.json)) | not needed | [B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/y07rjmb0) · [A](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/zi2c9puw) | 28615377 → 28615378 |

Stage B result: LaRS val (PQ, things PQ, mIoU, box AP 8 obj / 9 cls); FloW test / CANSURF val box AP (single class, so debris-only = class-agnostic). Cross-domain: debris-only (class-agnostic). **B** = stage B W&B run (scored checkpoint), **A** = stage A run. **selected** = the configuration the stage A rule picks for that dataset and arm (all jobs finished 2026-10-27; every stage B checkpoint's SHA-256 matches its JSON and its epoch count equals the stage A pick). Findings: [RESULTS.md finding 16](RESULTS.md#findings); decision log 2026-10-26 / 2026-10-27.

**Outcome (selection rule applied, 2026-10-27).** Stage A picks and their stage B result vs the 12-epoch baseline:

| Dataset | Arm | Selected | Stage A | Stage B | 12-epoch baseline (stage B) |
|---|---|---|---|---|---|
| LaRS | full | 24 epochs, 1e-4 | 0.429 vs 0.404 | PQ **0.407** | 0.405 |
| LaRS | LoRA | 24 epochs, 5e-4 | 0.452 vs 0.369 | PQ **0.424** | 0.379 |
| FloW | full | 24 epochs, 1e-4 | 0.515 vs 0.487 | box AP **0.256** | 0.236 |
| FloW | LoRA | 24 epochs, 1e-4 | 0.511 vs 0.478 | box AP **0.247** | 0.220 |
| CANSURF | full | 24 epochs, 1e-4 | 0.779 vs 0.776 | box AP **0.631** | 0.631 |
| CANSURF | LoRA | 24 epochs, 5e-4 | 0.783 vs 0.766 | box AP **0.635** | 0.628 |

- 24 epochs is picked for every dataset and arm; lr ×5 only for LoRA on LaRS and CANSURF. **Full fine-tuning at lr ×5 degrades** on every dataset (LaRS PQ 0.309, things 0.078; FloW 0.100; CANSURF 0.578).
- Gains: LoRA LaRS +0.045 PQ (now above full, 0.424 vs 0.407); FloW +0.020 / +0.027; CANSURF flat (0.000 / +0.007), already near its ceiling.
- After selection, LoRA ≥ full on LaRS and CANSURF; full > LoRA on FloW by 0.009. Both remain below YOLOv11s on FloW (0.401) and CANSURF (0.860).
- Cross-domain (LaRS LoRA → FloW / CANSURF, debris-only): e24 0.087 / 0.049, e24 lr ×5 0.136 / 0.089, vs 0.049 / 0.023 at 12 epochs.
- The main table above still shows the 12-epoch runs; switching it to the selected configurations is a pending decision.

## Temporal table (VidEoMT, separate from the main comparison)

VidEoMT needs **video**: LaRS 9-frame sequences before each annotated keyframe (YouTube-VIS export, labels on the keyframe only). It outputs **per-keyframe mask AP** and a **flicker proxy** (1 − mean consecutive-frame per-class mIoU over 35 val clips). It produces no PQ / mIoU / box AP on LaRS, so it is not in the main table.

| Model | Trained on | W&B training run | Predicts | Per-keyframe mask AP (L8) | Flicker (lower = steadier) | Source |
|---|---|---|---|---|---|---|
| EoMT panoptic, per frame (reference) | LaRS stills | [67kk33hk](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/67kk33hk) (ablation 03, fixed 12 epochs) | L11 | — | 0.229 (all classes) ([json](../ablations/07_videomt_temporal/sources/eomt_panoptic_seq_flicker_27715321.json), [W&B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/5aoppsyb)) | [07](../ablations/07_videomt_temporal/README.md) |
| VidEoMT, 4k iterations | LaRS sequences | [lars_vitb_27772987](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_27772987) · [charts](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/stepfix_lars_vitb_27772987) | L8 | 0.016 ([json](../ablations/07_videomt_temporal/sources/videomt_rescore_keyframe_27775358.json), [W&B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/ab07_keyframe_ap_27775358)) | 0.612 (8 object classes) ([json](../ablations/07_videomt_temporal/sources/videomt_seq_flicker_27776809.json), [W&B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/wzpe700z)) | [07](../ablations/07_videomt_temporal/README.md) |
| VidEoMT, 16k iterations | LaRS sequences | [lars_vitb_16k_27776810](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_16k_27776810) · [charts](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/stepfix_lars_vitb_16k_27776810) | L8 | 0.009 ([json](../ablations/07_videomt_temporal/sources/videomt_rescore_keyframe_27806956.json), [W&B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/ab07_keyframe_ap_27806956)) | 0.311 (8 object classes) ([json](../ablations/07_videomt_temporal/sources/videomt_seq_flicker_27807251.json), [W&B](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/r4ugnnd5)) | [07](../ablations/07_videomt_temporal/README.md) |

W&B training run = the run that produced the scored checkpoint; **charts** = its correction run (2026-10-29): the original VidEoMT runs plot every point at `trainer/global_step` = 0 (upstream `define_metric`, never written by Detectron2), so their charts are empty; the correction run re-logs the same values against `global_step` (`scripts/wandb_videomt_step_fix.py`, 10,000 loss values checked against the training logs). In the 4k run every `segm/*` value is 0: in-training eval used the upstream all-frames VIS AP, 0 by construction on LaRS. The 16k run's `segm/*` is per-keyframe AP in percent; the links inside the metric cells are the evaluation runs (keyframe AP: eval-only rescore with up to 50 tracks per clip; flicker: sequence-flicker job). The EoMT reference checkpoint is ablation 03's (`67kk33hk/checkpoints/epoch=11-step=7812.ckpt`, recorded in its flicker JSON); VidEoMT checkpoints are the last iteration of each run (`model_final.pth`, path in the eval logs). **They were on scratch and have been purged**, with no SHA-256 recorded: the VidEoMT numbers can be audited from the logs, JSONs and W&B runs, but not re-scored without retraining.

**Not yet like-for-like:** EoMT panoptic's flicker averages over all its classes, VidEoMT's over the 8 object classes only; recomputing EoMT panoptic flicker over L8 aligns them (inference only). ID-switch rate was planned in ablation 07 but never computed. The earlier EoMT semantic flicker (0.041, obstacle / water / sky) stays in ablation 07 as a reference.

## Data and class coverage

Full reference: [COVERAGE_CLASSES.md](COVERAGE_CLASSES.md): coverage matrix, complete class lists, classes counted today, proposals, and the classes under the agreed decisions (§ 6).

| Topic | Key point | Details |
|---|---|---|
| Coverage | EoMT zero-shot is tested on all three datasets; **YOLO zero-shot (COCO only) has never been tested**. Cross-domain runs one way only: LaRS → FloW / CANSURF | [§ 1](COVERAGE_CLASSES.md#1-coverage-matrix-training-data--test-dataset) |
| Classes per training source | COCO 80 objects + 53 regions (EoMT zero-shot), COCO 80 (YOLO zero-shot), LaRS 11 / 9 / 11, FloW 1 (`waste`), CANSURF 1 (cans) | [§ 2](COVERAGE_CLASSES.md#2-classes-each-training-source-can-predict-complete-lists) |
| Classes counted until now | On FloW / CANSURF, every predicted object counted as debris (all 80 COCO classes for zero-shot, all 8 LaRS objects for LaRS models) | [§ 3](COVERAGE_CLASSES.md#3-classes-counted-in-each-evaluation-current-rule) |
| Agreed rule | Debris-only main number, class-agnostic reference, original class kept, tightened COCO → LaRS mapping | [§ 4](COVERAGE_CLASSES.md#4-classes-that-do-not-belong-in-an-evaluation-and-proposals), [§ 6](COVERAGE_CLASSES.md#6-classes-under-the-agreed-decisions-2026-10-23) |
| Common / unique classes | Only the concept "floating debris" is shared (FloW waste, CANSURF cans, LaRS Float + Other); LaRS val has 3 Float and 73 Other segments | [§ 5](COVERAGE_CLASSES.md#5-common-and-unique-classes) |

## Run schedule (parallel)

All jobs use one A100 each (EoMT and YOLO trainings were single-GPU so far), so independent runs go side by side. Each wave starts when its inputs and approvals are ready.

| Wave | Runs in parallel | Waits for | GPU time | Wall-clock |
|---|---|---|---|---|
| 0 | Implementation: fix 1/1b scorer, YOLOv11s + SAM pipeline, SAM masks + quality report, EoMT training on FloW/CANSURF, EoMT LoRA module + tests; **pre-built label caches** | — | — | ~3 days (sequential: one implementer) |
| 1 | fix 1 + 1b inference (EoMT LaRS metrics, SAM control, 4 re-score jobs); fix 2 YOLOv11s cross-domain (2); YOLOv11s 11-class stage A; **YOLOv11s FloW and CANSURF stage A** (fix 3, native boxes need no SAM); **SAM masks FloW and CANSURF** (2); **EoMT LaRS full and LoRA stage A** (fix 4) | wave 0 | ~10 h | ~3–4 h (longest: YOLOv11s CANSURF, EoMT LaRS stage A) |
| 2 | YOLOv11s 11-class stage B → YOLO + SAM pilot on held10; YOLOv11s FloW / CANSURF stage B; EoMT LaRS full / LoRA stage B; SAM quality report | wave 1 (per model) | ~6 h | ~2–3 h |
| 3 | YOLO + SAM scored on LaRS val; **EoMT FloW and CANSURF, full and LoRA, stage A** (4 jobs) | **your approval** of the YOLO + SAM pilot and of the SAM quality report | ~6 h | ~3–4 h (CANSURF) |
| 4 | EoMT FloW / CANSURF full and LoRA, stage B (4 jobs) | wave 3 | ~5 h | ~2–3 h |

**Total:** ~25–30 GPU hours (fix 4 adds ~8–11), about **1–1.5 days of wall-clock compute** in parallel, after ~3 days of implementation and the two approval points.

**Flags: what cannot be parallelised, or blocks action**

| Limit | Effect | Mitigation |
|---|---|---|
| Stage B needs stage A's epoch count | each model's two stages run in sequence | none (protocol) |
| Approval gates (YOLO + SAM pilot, SAM quality check) | waves 3–4 wait for your sign-off | review promptly; everything not depending on SAM runs in waves 1–2 |
| **Implementation is sequential** (~3 days, one implementer) | the real bottleneck: compute is ~1–1.5 days | order the code by wave so wave 1 can start as soon as its pieces are ready |
| **Shared label-cache race** | parallel YOLO jobs rebuilding the same cache crashed the LoRA r32 run (2026-10-20) | build all caches in one short job before launching a wave |
| Startup stall (~10 min per training job) | +10 min per job | find and fix before wave 1 |
| Queue and allocation | not measured: how many A100 jobs run concurrently depends on Snellius load and our budget | waves fit in ~6–10 concurrent jobs; if fewer run, the wave stretches |
| Shared scratch I/O | many jobs reading the same datasets slowed reads ("slow image access") | pre-cache or stagger dataset reads |
| **No action possible yet** | the plan is still under your audit; no job can start until approval and wave 0 | — |

## Common rules for all fixes

- **Scorer:** pycocotools via `water-debris-seg/scripts/eval_coco_bbox.py` for every box AP; same GT files as ablations 04–10. EoMT boxes always carry real confidences from EoMT's instance inference.
- **Held-out protocol** for any new training: stage A picks the epoch count on a held-out slice of the training split; stage B retrains on the full training split and keeps the last checkpoint; the test split is scored once ([ablation 10 TRAINING.md § 3](../ablations/10_yolo_lars_boxes/TRAINING.md)).
- **W&B:** project [`water-debris-seg`](https://wandb.ai/riccardo-campanella/water-debris-seg); groups `fix1_lars_metrics` (fix 1 and 1b), `fix2_yolo_lars` (new fix 2 runs; LaRS training runs stay in [`ablation10_yolo_lars`](https://wandb.ai/riccardo-campanella/water-debris-seg/groups/ablation10_yolo_lars)), `fix3_sam_box_datasets`.
- **Audit trail:** every number links to its JSON/log in `sources/` and its W&B run; checkpoints and GT files recorded with SHA-256.
