# Fix 1 — One EoMT model, three LaRS metrics in separate columns (PQ, mIoU, box AP)

> **Results (2026-10-26):** all runs of this fix are done; values with links to their JSON are in [RESULTS.md](RESULTS.md). ⏳ cells below are the plan as written before the runs.

**Status: plan, not started (aligned with the decisions of 2026-10-23: EoMT semantic dropped from the main table, debris-only scoring on FloW / CANSURF, tightened COCO → LaRS mapping).** Back to the summary: [README.md](README.md).

## Motivation

1. **Mixed column.** The REPORT's "LaRS PQ / mIoU" column ([REPORT.md](../REPORT.md), main results table) mixes three metrics from different models:

| Row | Value | Metric actually reported |
|---|---|---|
| EoMT LaRS semantic FT | mIoU 98.0 | 3-class semantic mIoU (obstacle, water, sky) |
| EoMT LaRS panoptic FT | PQ ≈ 0.395 | 11-class panoptic quality |
| VidEoMT LaRS | flicker 0.041 / 0.229 / 0.612 / 0.311; AP 0.016 / 0.009 | temporal proxy; per-keyframe mask AP |

2. **No EoMT box AP on LaRS**, so the LaRS-trained YOLO (box AP only) cannot be compared with EoMT on LaRS.

3. **New finding while preparing this fix: EoMT's box AP elsewhere used a constant score.** In ablations 04, 05 and 09, every EoMT box was given `score = 0.5` (`water-debris-seg/scripts/eval_eomt_cansurf_mask2box.py`, `preds_to_detections`). AP ranks detections by confidence; with one constant score the precision–recall curve cannot be traced, and EoMT's FloW/CANSURF AP is **understated** relative to YOLO, whose boxes carry real scores. See § 5.

## 1. The approach: the LaRS-trained EoMT panoptic model fills all three columns

EoMT panoptic (ablation 03) is trained on LaRS masks and tested on LaRS val. From the same predictions:

| Column | How it is computed from EoMT panoptic | Ground truth | Status |
|---|---|---|---|
| **PQ** (11 classes; PQ_things, PQ_stuff) | EoMT's panoptic inference (`to_per_pixel_preds_panoptic`) → torchmetrics PQ, as EoMT's own validation | LaRS val panoptic PNGs | **exists**: 0.395 |
| **mIoU** (obstacle / water / sky) | Same panoptic map, merged to 3 classes: Static Obstacle + all 8 object classes → obstacle; Water; Sky. Confusion-matrix mIoU, ignore 255 | LaRS val semantic PNGs (the ablation 02 GT) | **new** |
| **Box AP** (8 object classes; 9 with Static Obstacle) | EoMT's **instance** inference (its own `mask_classification_instance.eval_step`): top-100 (query, class) pairs, score = class probability × mean mask probability, mask > 0.5 → min/max box | `ablations/10_yolo_lars_boxes/sources/lars_val_gt_coco_boxes.json` (same boxes as YOLO; crowd ignored) | **new** |

The **semantic** model (ablation 02) is **dropped from the main table** (decision 2026-10-23); it keeps its own mIoU (98.0) on its ablation page and is used once more here, to validate the mIoU scorer (§ 3). It labels pixels as obstacle / water / sky only: no object classes and no separation of touching objects, so it cannot give 11-class PQ or per-class box AP. Those cells are **not needed**: the panoptic model covers EoMT's LaRS PQ and box AP (same family, same LaRS data). A class-agnostic obstacle-component AP is reported only as a side note, as in ablation 09. Applying SAM to the semantic model's boxes would not help: SAM adds neither classes nor instance separation. The **zero-shot** model (ablation 01) gets PQ (exists: 0.298), mIoU and box AP through the COCO → LaRS prediction remap, **tightened** to maritime-plausible classes ([COVERAGE_CLASSES.md § 6](COVERAGE_CLASSES.md#6-classes-under-the-agreed-decisions-2026-10-23)); the PQ is reported with both the current and the tightened mapping.

### Why instance-style scores for box AP

PQ needs one non-overlapping segment per pixel, so panoptic output is thresholded and has no confidence per segment. AP needs ranked detections with confidences, which is what EoMT's instance head produces and what its authors use for instance AP. YOLO is scored the same way (all detections down to conf 0.001, ranked). For transparency the table also shows box AP from the panoptic output with the old constant score.

### Control: EoMT panoptic + SAM (masks replaced by SAM)

YOLO + SAM ([fix 2](FIX2_YOLO_ON_LARS.md#open-item-a--pq-and-miou-for-yolo-yolo--sam-with-watersky-decided-2026-10-21)) differs from EoMT panoptic in two ways at once: **detection** (which objects, which classes) and **mask source** (SAM vs EoMT's own masks). A second SAM row separates the two:

| Step | What |
|---|---|
| Segments | Exactly the segments EoMT panoptic keeps for PQ (`to_per_pixel_preds_panoptic`: class, score, keep rule unchanged) |
| Masks | Each segment's min/max box prompts the same frozen SAM as fix 2 and fix 3; SAM's mask replaces EoMT's |
| Panoptic map | Same merge as YOLO + SAM: objects (highest score first), then Static Obstacle, Water, Sky; unclaimed pixels void |
| Scored | PQ and mIoU with the same scorers. Box AP is not re-computed (same boxes as EoMT panoptic) |

| Comparison | Isolates |
|---|---|
| EoMT panoptic + SAM vs EoMT panoptic | mask quality: SAM's masks vs EoMT's own, same detections |
| YOLO + SAM vs EoMT panoptic + SAM | detection: same mask source (SAM), different detector |

No threshold is tuned: EoMT's own keep rule decides the segments, so LaRS val is used only for scoring.

## 2. Same protocol for both models?

| | EoMT panoptic (ab03) | EoMT semantic (ab02) | YOLO (ab10) |
|---|---|---|---|
| Trained on | LaRS train | LaRS train | LaRS train |
| Epoch count | fixed recipe: 12 epochs (`configs/ablation03_lars_panoptic_finetune.yaml`) | fixed: 31 epochs (`ablation02_lars_semantic_finetune.yaml`) | chosen on held-out train sequences (stage A), then retrained on all train (stage B) |
| Checkpoint kept | **last** (`epoch=11-step=7812.ckpt`) | **last** (`epoch=30-step=2356.ckpt`) | **last** (`last.pt`) |
| Selected on LaRS val? | No | No | No |
| Scored on | LaRS val, once | LaRS val | LaRS val, once |

Neither family selected its checkpoint on LaRS val, so no retraining is needed for a fair comparison. The difference (fixed recipe vs held-out-chosen epoch count) is documented next to the table.

## 3. Validation of the new scorers before reporting

- **mIoU scorer:** run on the ablation 02 semantic checkpoint first; it must reproduce EoMT's own validation mIoU 98.0 (within 0.2), else stop.
- **PQ:** already validated against EoMT's Lightning validation (`eval_eomt_lars_panoptic_pq.py`, ab03 0.395).
- **Box AP:** pycocotools via `eval_coco_bbox.py`, the scorer already used for YOLO on LaRS; a GT-as-prediction sanity check must give AP = 1.0.

## 4. Expected outputs

| Output | Path |
|---|---|
| Script | `water-debris-seg/scripts/eval_eomt_lars_three_metrics.py` (reuses `build_model_*`, `predict_image` and EoMT's instance post-processing) |
| Per checkpoint: PQ, mIoU, box AP (+ constant-score box AP) | `fairness_fixes/sources/fix1_eomt_<ckpt>_lars_val.json` (checkpoint + GT SHA-256) |
| Box predictions (COCO json) | `fairness_fixes/sources/fix1_eomt_<ckpt>_lars_val_boxes.json` |
| Log | `fairness_fixes/sources/fix1_<job>.log` |
| REPORT table with split columns; ablation 01–03 pages updated | `REPORT.md`, `ablations/0{1,2,3}_*/README.md` |
| Decision-log entry | `docs/water_debris_seg_decision_log.md` |
| W&B | ⏳ group `fix1_lars_metrics`, one run per checkpoint, with the JSON attached as evidence |

**Target table (LaRS val, 198 images):**

| Model | Trained on | PQ | mIoU (3 cls) | Box AP 8 obj | Box AP 9 cls | AP_small (obj) | AR100 (obj) | Box AP 8 obj, constant score |
|---|---|---|---|---|---|---|---|---|
| EoMT zero-shot | COCO | 0.298 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| **EoMT panoptic FT** | LaRS | 0.395 | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| EoMT panoptic + SAM (control) | LaRS | ⏳ → SAM (eval) | ⏳ → SAM (eval) | not needed (same boxes as panoptic) | not needed | not needed | not needed | — |
| YOLOv11s full | LaRS boxes | not needed (→ YOLOv11s + SAM, fix 2) | not needed (→ YOLOv11s + SAM, fix 2) | 0.176 | 0.195 | 0.062 | 0.251 | — (real scores) |

Not in this table (decisions 2026-10-23): EoMT semantic (mIoU 98.0, ablation 02) and YOLOv11n (0.145 / 0.168, ablation 10) stay as references on their pages.

Legend: ⏳ pending; not needed = covered by another row (named); → SAM (eval) = masks produced by SAM at evaluation; — = not applicable to this row (YOLO boxes already carry real scores; the SAM control re-uses EoMT's boxes).

## 5. Consequence for FloW and CANSURF (ablations 04, 05, 09)

The constant-score issue affects every EoMT box AP reported so far (FloW 0.034 / 0.082 / 0.014; CANSURF 0.095 / 0.063 / 0.078 / 0.034). They should be **re-scored with the same instance-style scores** before any EoMT-vs-YOLO comparison on those datasets, including the fix 3 baselines. The **zero-shot and panoptic** checkpoints are re-scored with EoMT's instance inference (the semantic checkpoint is dropped from the comparison, decision 2026-10-23; its old values stay on its pages). **Classes counted** (decision 2026-10-23): the main number is **debris-only** (zero-shot: bottle, cup, frisbee, sports ball, kite, backpack, handbag, suitcase, umbrella; panoptic: Float, Other); a **class-agnostic** score (any predicted object) is reported next to it; every saved prediction keeps its **original class**. The saved prediction files only contain score 0.5 and category 0, so this needs a new inference pass: FloW test (800 images) and CANSURF val (2,209) × 2 checkpoints, under 1 GPU hour. Proposed as **fix 1b**, run together with fix 1; old values kept in the ablation pages and marked superseded.

## 6. References (existing results and W&B)

| Number | Source file | W&B |
|---|---|---|
| EoMT zero-shot PQ 0.298 | `ablations/01_zeroshot_lars/sources/pq_zeroshot_predremap_28011322.json` | original run [`txjoa9ef`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/txjoa9ef) (pre-fix value; see [WANDB_AUDIT.md](../WANDB_AUDIT.md)) |
| EoMT semantic mIoU 98.0 | `ablations/02_finetune_lars_semantic/sources/validate_27660680.log` | validate [`duguh1wo`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/duguh1wo); training [`mk2w7dr8`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/mk2w7dr8) |
| EoMT panoptic PQ 0.395 | `ablations/01_zeroshot_lars/sources/pq_native_ab03_check_28011322.json` | validate [`xcl7zvm8`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/xcl7zvm8); training [`67kk33hk`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/67kk33hk) |
| VidEoMT flicker / AP (temporal table, unchanged) | [ablation 07 README](../ablations/07_videomt_temporal/README.md) | [`g1bfz3nn`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/g1bfz3nn), [`5aoppsyb`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/5aoppsyb), `wzpe700z`, `r4ugnnd5` |
| YOLO LaRS box AP | `ablations/10_yolo_lars_boxes/sources/eval_lars_full_stageB_28132656.json`, `eval_lars_full_11s_stageB_28378811.json` | [lars_full_stageB](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_stageB_20261016_130702), [lars_full_11s_stageB](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_full_11s_stageB_20261020_170700) |
| Constant EoMT score | `water-debris-seg/scripts/eval_eomt_cansurf_mask2box.py` (`preds_to_detections`) | — |
| EoMT instance scoring used | EoMT source `training/mask_classification_instance.py` (`eval_step`), `/scratch-shared/rcampanella/water-debris-seg/third_party/eomt` | — |
| **New results** | ⏳ `fairness_fixes/sources/fix1_*.json` | ⏳ group `fix1_lars_metrics` |

Checkpoints: `data/water_debris/checkpoints/ab02_semantic_mk2w7dr8/epoch=30-step=2356.ckpt`, `ab03_panoptic_67kk33hk/epoch=11-step=7812.ckpt` (SHA-256 in `SHA256SUMS`).

## 7. Caveats

- **Static Obstacle** is stuff in EoMT and one box per segment in YOLO's labels; the 8-object-class AP (without it) is the cleaner comparison.
- **mIoU from a panoptic model** merges 9 classes into "obstacle"; it is comparable with ablation 02's 3-class mIoU, but the panoptic model was not trained for that objective.
- **Box AP from masks** inherits mask quality: a loose mask gives a loose box.
- **VidEoMT** stays in the temporal table: its per-keyframe AP is mask AP, not box AP.

**Cost:** fix 1 is inference only on 198 images × 3 checkpoints (zero-shot, panoptic, and semantic for the scorer check) plus the SAM control (< 1 GPU hour); fix 1b under 1 more GPU hour.
