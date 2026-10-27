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

## Temporal table (VidEoMT, separate from the main comparison)

VidEoMT needs **video**: LaRS 9-frame sequences before each annotated keyframe (YouTube-VIS export, labels on the keyframe only). It outputs **per-keyframe mask AP** and a **flicker proxy** (1 − mean consecutive-frame per-class mIoU over 35 val clips). It produces no PQ / mIoU / box AP on LaRS, so it is not in the main table.

| Model | Trained on | Predicts | Per-keyframe mask AP (L8) | Flicker (lower = steadier) | Source |
|---|---|---|---|---|---|
| EoMT panoptic, per frame (reference) | LaRS stills | L11 | — | 0.229 (all classes) | [07](../ablations/07_videomt_temporal/README.md) |
| VidEoMT, 4k iterations | LaRS sequences | L8 | 0.016 | 0.612 (8 object classes) | [07](../ablations/07_videomt_temporal/README.md) |
| VidEoMT, 16k iterations | LaRS sequences | L8 | 0.009 | 0.311 (8 object classes) | [07](../ablations/07_videomt_temporal/README.md) |

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
