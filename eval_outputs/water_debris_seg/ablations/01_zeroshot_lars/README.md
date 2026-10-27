# Ablation 01 — EoMT zero-shot on LaRS (+ box-set mask→box)

> **Role in the benchmark (2026-10-23).** **Main comparison table: EoMT zero-shot row.** LaRS PQ 0.298 stays (also to be reported with the tightened COCO → LaRS mapping, [COVERAGE_CLASSES.md § 6](../../fairness_fixes/COVERAGE_CLASSES.md#6-classes-under-the-agreed-decisions-2026-10-23)); LaRS mIoU and box AP are added by [fix 1](../../fairness_fixes/FIX1_LARS_SPLIT_METRICS.md). The zero-shot FloW / CANSURF values (ablations 04, 05) are superseded by fix 1b.

> **Re-run 2026-10-13 (job 28011322).** The original LaRS PQ ≈ 0.0014 came from a GT bridge (`taxonomy.py`) that pointed at the wrong COCO ids (boats → *spoon*, buoys/floats → *knife*, water → *food-other*, sky → *paper*), so correct predictions could not match. Ablation 01 is now scored with **prediction-side remapping**: GT stays native LaRS 11-class; frozen COCO-133 predictions are mapped many-to-one (`water-debris-seg/configs/coco133_to_lars_pred_map.json`). The script reproduces EoMT's own validate for ab03 (0.3946 vs 0.3945), so 01 and 03 are now in the **same label space**. Sections below that discuss 0.0014 are kept for the record and marked superseded.

**Question.** How well do **unfine-tuned** COCO/ADE EoMT weights transfer to maritime panoptic/semantic labels and to debris-like detection on box-only sets?

**Expected outcome.** Reasonable water/sky stuff from ADE-like priors; **low** PQ on debris things (Float/Other) and **low** mask→box mAP on FloW/CANSURF vs in-domain YOLO (CANSURF paper: ~12× gap).

**Not training.** No `fit` on any dataset in this ablation. Official class head is kept (`load_ckpt_class_head: true`).

---

## Motivation (why this ablation exists)

This is **rung 2 of the EoMT eval ladder** (after smoke): a frozen, off-the-shelf vision foundation model (VFM) evaluated as a maritime segmenter and as a detector. Later ablations only mean something relative to this floor:

| Later ablation | What it changes vs 01 | Why 01 is the baseline |
|---|---|---|
| [02 semantic FT](../02_finetune_lars_semantic/README.md) | Train 3-class obstacle/water/sky on LaRS | Shows whether dense FT recovers stuff mIoU |
| [03 panoptic FT](../03_finetune_lars_panoptic/README.md) | Train 11-class LaRS panoptic | Shows whether PQ on boats/buoys/floats needs in-domain things |
| [04–06 mask→box](../04_flow_mask2box/README.md) | Same weights, box datasets | Isolates domain/size/distractor gaps without extra training |
| [08 YOLO](../08_specialized_yolo/README.md) | In-domain box detector | Specialized ceiling on CANSURF/FloW |
| [09 transfer](../09_cross_dataset_transfer/README.md) | LaRS-FT weights on box sets | Separates “generic VFM” vs “LaRS-adapted VFM” |

**Benchmark relevance.** When this ablation was designed, the spec's hard rule was: train masks **only on LaRS**; treat FloW/CANSURF as **eval-only** detection (revised 2026-10-23: EoMT now also trains on FloW/CANSURF with box-prompted SAM masks, fix 3). Ablation 01 is the only run that uses **neither** LaRS training **nor** box-set training. If a generic COCO panoptic checkpoint already segmented water-surface debris well, LaRS fine-tuning (02/03) and specialized YOLO (08) would be unnecessary. The measured near-zero PQ and the ~6–12× CANSURF mAP gap vs YOLO justify those later stages.

### Inputs

| Input | What it is | Role |
|---|---|---|
| RGB stills | LaRS val keyframes (USV nearshore); optionally CANSURF/FloW val/test | Camera images only — no extra sensors |
| Frozen EoMT weights | Official `coco_panoptic_eomt_base_640_2x.bin` (COCO-133 panoptic, ViT-B/14 DINOv2) | Class head **not** reset; predicts COCO things/stuff |
| Panoptic GT (LaRS) | Instance-colored PNGs + COCO panoptic JSON | Remapped LaRS categories → COCO-133 so PQ is in the **same label space** as the head |
| Box GT (box sets) | YOLO/VOC/COCO axis-aligned boxes | Never used as mask GT; only scored after mask→box |

### Outputs

| Output | Produced from | Used as |
|---|---|---|
| Per-pixel panoptic map | EoMT queries → class + instance ids (things vs stuff) | LaRS PQ / SQ / RQ |
| Thing masks → boxes | Min/max foreground pixels of each non-stuff segment | COCO bbox mAP / AP_S on box-only sets |
| W&B + log | Lightning `validate` | Report Source column |

Nothing in this ablation updates weights. Lightning is invoked as `main.py validate`, not `fit`.

---

## Metrics

Two **incompatible** GT kinds force two protocols. Pixel masks exist only on LaRS; FloW/CANSURF have boxes. Headline mask IoU against filled rectangles is out of scope.

### LaRS panoptic: PQ, SQ, RQ

**Panoptic Quality** (Kirillov et al., CVPR 2019) scores a single panoptic map that mixes **things** (countable instances: boats, buoys, floats) and **stuff** (amorphous regions: water, sky, static obstacle).

For each class, predicted segments are matched to GT segments if **IoU > 0.5** (unique matching). Then:

| Symbol | Name | Meaning |
|---|---|---|
| TP / FP / FN | matched pairs / extra preds / missed GT | Recognition errors |
| **SQ** | Segmentation Quality | Mean IoU of matched pairs only — “how well are the true positives painted?” |
| **RQ** | Recognition Quality | F1 of that matching: \(\mathrm{RQ} = \frac{\mathrm{TP}}{\mathrm{TP} + \tfrac{1}{2}\mathrm{FP} + \tfrac{1}{2}\mathrm{FN}}\) |
| **PQ** | Panoptic Quality | \(\mathrm{PQ} = \mathrm{SQ} \times \mathrm{RQ} = \frac{\sum_{(p,g)\in\mathrm{TP}} \mathrm{IoU}(p,g)}{\mathrm{TP} + \tfrac{1}{2}\mathrm{FP} + \tfrac{1}{2}\mathrm{FN}}\) |

EoMT logs three slices (and per decoder block `-1/-2/-3`; headline = last block):

- **`val_pq_all`** — mean over classes that appear
- **`val_pq_things`** — boats/rowboats/paddleboards/buoys/swimmers/animals/floats/other (after the COCO bridge: mostly `boat`, `surfboard`, `person`, `bird`)
- **`val_pq_stuff`** — water/sky/static obstacle (bridged to `river`, `sky-other-merged`, `wall-stone`)

**[Superseded 2026-10-13: bridge bug; see banner.]** **How to read the 01 numbers.** PQ ≈ **0.0014** is not “slightly weak transfer”; it is essentially **no usable panoptic alignment**. Stuff PQ = **0** means water/sky were never matched at IoU > 0.5 under the COCO-133 bridge (class-name priors like “river”/“sky” do not imply maritime extent or ID). Things PQ ≈ **0.0024** with SQ ≈ **0.016** and RQ ≈ **0.004** means even the rare matches are poorly localized and almost never recognized. Compare ablation 03 (same val images, **native 11-class** head after LaRS FT): PQ ≈ **0.395**, stuff PQ ≈ **0.94**.

**[Superseded 2026-10-13: 01 and 03 now share the LaRS label space.]** **Caveat (apples-to-oranges with 03).** 01 scores PQ in **COCO-133 space** after `LARS_TO_COCO_PANOPTIC_CATEGORY`. Ablation 03 scores **LaRS 11-class** space. 01 is the correct zero-shot protocol (the COCO head cannot emit LaRS IDs). Do not treat 0.0014 vs 0.395 as a pure architecture gap; it mixes domain shift **and** label-space mismatch. The point of 01 is the COCO-head floor, not a like-for-like class-set comparison.

### LaRS semantic: mIoU (not run here)

**mIoU** = mean Intersection-over-Union over the three LaRS semantic PNG IDs: obstacle (0), water (1), sky (2); ignore 255. That is a **pixel** metric with no instance matching. Ablation 01 only ran the panoptic COCO-133 bridge, so semantic mIoU is blank; use [02](../02_finetune_lars_semantic/README.md) for the 3-class protocol (and a future 01 semantic job if a like-for-like zero-shot mIoU is needed).

### Box sets: mask→box mAP and AP_S

FloW/CANSURF have **no pixel GT**, so PQ/mIoU are undefined. Protocol:

1. Run the same frozen COCO (or later LaRS-FT) checkpoint — **no weight updates**.
2. Keep thing masks; drop stuff (COCO stuff ids 80–132).
3. Convert each mask to an axis-aligned box (min/max of foreground pixels).
4. Map predicted classes onto the dataset’s label (CANSURF: all remaining things → `can`; FloW: `waste`).
5. Score **COCO bbox AP** vs native boxes.

| Metric | Definition in this repo |
|---|---|
| **mAP** | Mean AP over IoU thresholds 0.50:0.95 (COCO-style; implementation in `eval_coco_bbox.py`) |
| **AP50 / AP75** | AP at IoU 0.50 and 0.75 |
| **AP_S** | AP restricted to GT area **&lt; 32²** (far-field debris; FloW/CANSURF are dominated by this bin) |

CANSURF zero-shot (same checkpoint, logged under [05](../05_cansurf_small_object/README.md)): mAP **0.095**, AR100 **0.140** vs YOLOv11n mAP **0.857** on the same aligned val GT (pycocotools, re-scored 2026-10-13; legacy values 0.147 / AP_S 0.083 superseded — see [`RESCORE.md`](../../RESCORE.md)). Score is fixed at 0.5 per EoMT box, so read AR100 alongside mAP.

Optional diagnostic `box_proxy_IoU` vs the GT rectangle is **not** a headline metric (would reward painting the box, not the object).

---

## Models and components

### Model (frozen)

**EoMT** ([Encoder-only Mask Transformer](https://arxiv.org/pdf/2503.19108), CVPR 2025) — general-purpose image panoptic segmentation. Not trained on maritime data.

| Piece | Ablation 01 setting |
|---|---|
| Backbone | `vit_base_patch14_reg4_dinov2` (ViT-B/14 DINOv2 + 4 register tokens) |
| Decoder | EoMT, **200** mask queries, **3** blocks |
| Task head | `MaskClassificationPanoptic`, **133** COCO panoptic classes |
| Stuff ids | `[80 … 132]` (COCO stuff); things are 0–79 in EoMT’s contiguous map |
| Resolution | 640×640 (pad/resize as in upstream EoMT) |
| Checkpoint | Hugging Face `coco_panoptic_eomt_base_640_2x.bin` |
| Class head | **Loaded as-is** (`load_ckpt_class_head: true`) |

**Not in this ablation:** VidEoMT (temporal; [07](../07_videomt_temporal/README.md)), YOLOv11 training ([08](../08_specialized_yolo/README.md)), or any LaRS `fit` ([02](../02_finetune_lars_semantic/README.md)/[03](../03_finetune_lars_panoptic/README.md)). YOLO appears only as the **external** specialized ceiling on box sets.

### Label bridge (required for a COCO head)

**[Superseded 2026-10-13.]** The table below is the original GT-side bridge; its cat ids were wrong (e.g. 50 is *spoon*, not boat; 196 is *food-other-merged*, not river). Current protocol maps **predictions** many-to-one into LaRS: water ← river/sea/water-other; sky ← sky-other-merged; static obstacle ← 29 land/structure stuff classes; boat ← boat; paddle board ← surfboard; swimmer ← person; animal ← 10 animal classes; other ← bottle, cup, frisbee, sports ball, kite, backpack, handbag, suitcase, umbrella. Full list with official ids: [`coco133_to_lars_pred_map.json`](../../../../water-debris-seg/configs/coco133_to_lars_pred_map.json).

The COCO classifier cannot emit LaRS category IDs. GT is projected into COCO panoptic ids, then into EoMT’s `CLASS_MAPPING` (`LarsPanopticCoco133.target_parser`).

| LaRS panoptic | Unified role | COCO proxy (cat id) |
|---|---|---|
| Static Obstacle (1) | static_context | wall-stone (199) |
| Water (3) | water | river (196) |
| Sky (5) | sky | sky-other-merged (195) |
| Boat/ship (11), Row boats (12), Paddle board (13) | dynamic_obstacle | boat (50) |
| Buoy (14) | dynamic_obstacle | surfboard (49) |
| Swimmer (15) | dynamic_obstacle | person (1) |
| Animal (16) | dynamic_obstacle | bird (16) |
| **Float (17), Other (19)** | **debris** | **surfboard (49)** |

Debris has **no COCO class**. Float/Other share the same weak proxy as Buoy (`surfboard`). That is an inherent zero-shot limitation: even a perfect COCO detector cannot name “floating waste.”

### Code path (LaRS PQ)

```
LaRS stills + panoptic JSON
  → pack_lars_eomt_zips.py --full   (COCO zip layout: val2017.zip + panoptic JSON)
  → LarsPanopticCoco133             (GT remap LaRS → COCO-133)
  → EoMT MaskClassificationPanoptic (frozen COCO ckpt)
  → Lightning validate              (PQ / SQ / RQ)
```

Config: `water-debris-seg/configs/ablation01_lars_zeroshot.yaml`  
Job: `water-debris-seg/scripts/snellius/ablation01_eomt_lars.sbatch`

### Code path (box-set mAP)

```
RGB val/test + native boxes
  → same frozen EoMT (or later LaRS-FT ckpt)
  → drop stuff masks
  → mask_to_xywh
  → eval_coco_bbox.py  (mAP, AP_S)
```

Shared scripts: `eval_eomt_cansurf_mask2box.py`, `eval_coco_bbox.py`, `masks_to_coco_boxes.py`. CANSURF numbers for this **same** zero-shot checkpoint live in ablation 05.

---

## Datasets

| Dataset | Labels | Split used | Status in 01 | Why it is in the zero-shot question |
|---|---|---|---|---|
| **LaRS** | Panoptic COCO + semantic PNG (obstacle/water/sky); 11 panoptic cats | **val** (~197 images in job 27600047; 50× batch 4) | **Done** (PQ) | Only set with **pixel** GT — native PQ/mIoU protocol; USV nearshore water/sky/obstacles |
| **CANSURF** | 1-class YOLO boxes (`can`); reflective far-field cans | val (~2.2k images / 11.5k boxes in 05 GT JSON) | Numbers in [05](../05_cansurf_small_object/README.md) (same ckpt) | Box-only debris; tests small-object AP_S |
| **FloW-Img** | Boxes (`waste`); many &lt;32²; inland water | official test (~800) | **Pending** (ORCA portal) | Inland floating waste vs LaRS maritime |

LaRS panoptic categories (ID → name): Static Obstacle (1), Water (3), Sky (5); things: Boat/ship (11), Row boats (12), Paddle board (13), Buoy (14), Swimmer (15), Animal (16), Float (17), Other (19).

**Hard rule reminder (rule until 2026-10-23).** LaRS was the only legal **mask-training** set in the benchmark, but this ablation does not train even there. FloW/CANSURF must not be used to `fit` EoMT (boxes are loose on water: glare, ripples, partial submersion).

Scale (from spec): LaRS 4000+ keyframes; FloW-Img 2000 images / 5271 boxes; CANSURF ~7.2k raw YOLO images.

---

## Setup

| Item | Value |
|---|---|
| Checkpoint | Official EoMT DINOv2 COCO panoptic B, 640, 2× schedule (`coco_panoptic_eomt_base_640_2x.bin`) |
| LaRS eval | val split — PQ / SQ / RQ in COCO-133 space |
| Box sets | FloW/CANSURF val/test — **mask→box mAP** only |
| Class map | `LARS_TO_COCO_PANOPTIC_CATEGORY` (see table above) |
| Batch / img | 4 / 640×640, 1× A100 |

## Results

| Split | mIoU / PQ | mAP | AP_S | Source |
|---|---|---|---|---|
| LaRS semantic | — | — | — | *not run (panoptic bridge only)* |
| LaRS panoptic (native LaRS space, pred-side remap) | **PQ 0.298** (things 0.073, stuff 0.899; SQ 0.434, RQ 0.341) | — | — | job **28011322**, [`pq_zeroshot_predremap_28011322.json`](sources/pq_zeroshot_predremap_28011322.json) |
| *Check:* ab03 FT, same script | PQ 0.3946 (things 0.192, stuff 0.936) = EoMT validate 0.3945 | — | — | [`pq_native_ab03_check_28011322.json`](sources/pq_native_ab03_check_28011322.json) vs `../03_finetune_lars_panoptic/sources/validate_27660922.log` |
| ~~LaRS panoptic (COCO-133 GT bridge)~~ | ~~PQ ≈ 0.0014~~ superseded (bridge bug) | — | — | `sources/validate_27600047.log` |

Per-class PQ, zero-shot → LaRS FT (ab03): Boat/ship 0.391 → 0.395 · Static obstacle 0.783 → 0.885 · Water 0.979 → 0.984 · Sky 0.934 → 0.937 · Swimmer 0.061 → 0.200 · Animal 0.131 → 0.035 · Row boats 0 → 0.348 · Paddle board 0 → 0.342 · Buoy 0 → 0.138 · Other 0 → 0.077 · Float 0 → 0. Row boats, Buoy and Float have no COCO counterpart (0 by construction for zero-shot); on the 8 mappable classes PQ is 0.410 → 0.482.
| FloW-Img | — | **0.034** | **0.012** | [04](../04_flow_mask2box/README.md) job **27767340** (pycocotools; legacy 0.086 / 0.168) |
| CANSURF | — | **0.095** | n/a (19 small GT) | ablation 05, same COCO ckpt ([`metrics`](../05_cansurf_small_object/sources/eomt_zeroshot_yolo_aligned_gt.metrics.json); legacy 0.147 / 0.083) |
CANSURF in-domain YOLO (not this ablation): mAP50-95 **0.857** (pycocotools; Ultralytics 0.861, mAP50 0.966) ([08](../08_specialized_yolo/README.md)).

## Verdict

Zero-shot COCO EoMT transfers **well on stuff and boats** (PQ 0.298; water 0.98, sky 0.93, boat 0.39 — boat is already on par with LaRS FT) and **poorly on small/rare things** (things PQ 0.073). LaRS fine-tuning adds +0.10 PQ, mostly from shoreline, swimmers and classes COCO lacks; the debris class *Float* stays at 0 either way. (The earlier "PQ ≈ 0.0014, no transfer" verdict was a bridge-bug artefact.) On box sets it is a weak detector (CANSURF mAP 0.095, FloW mAP 0.034 vs in-domain YOLO 0.857 / 0.390). For debris the conclusion holds: “generic VFM is not a maritime debris system” and is the floor for LaRS fine-tunes 02/03 and specialized detection 08. WSODD out of scope v1 ([06](../06_wsodd_distractors/README.md)).

## Resources copied here

| Original | Copy in this folder | Present |
|---|---|---|
| Slurm validate log | `sources/validate_27600047.log` | yes |
| Earlier validate log | `sources/validate_27598361.log` | yes |

## Code components

- Config: `water-debris-seg/configs/ablation01_lars_zeroshot.yaml`
- Slurm: `water-debris-seg/scripts/snellius/ablation01_eomt_lars.sbatch` (job submitted on Snellius)
- Data: full LaRS train/val COCO-layout zips (`pack_lars_eomt_zips.py --full`)
- Checkpoint: `coco_panoptic_eomt_base_640_2x.bin` (Hugging Face)
- W&B: project `water-debris-seg`, group `ablation01_zeroshot_lars` — see `water-debris-seg/docs/WANDB.md`
- GT bridge: `water_debris_seg/taxonomy.py` → `LARS_TO_COCO_PANOPTIC_CATEGORY`
- Adapter: `water_debris_seg/eomt_adapters/lars_panoptic_coco133.py`
- Box-set eval (same weights): `scripts/eval_eomt_cansurf_mask2box.py`, `scripts/eval_coco_bbox.py`

## Tracking

| Run | Slurm | W&B group | Log copy |
|---|---|---|---|
| LaRS val PQ | **27600047** COMPLETED | `ablation01_zeroshot_lars` | W&B `txjoa9ef` (synced), `sources/validate_27600047.log` |

<!-- wandb-links:start (generated by scripts/wandb_audit.py; do not edit) -->

## W&B runs

Online runs for this ablation (fetched 2026-10-14); value-by-value comparison with the reported results: [`WANDB_AUDIT.md`](../../WANDB_AUDIT.md).

| Run | State | Slurm | Data on W&B | Audit |
|---|---|---|---|---|
| [`eomt_base640_2x_lars_val_27600047`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/txjoa9ef) | finished | 27600047 | one logged step | ⚠️ online holds pre-fix (superseded) values → corrected: [`ab01_zeroshot_lars_pq_predremap`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/ab01_zeroshot_lars_pq_predremap) |
| [`ab01_zeroshot_lars_pq_predremap`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/ab01_zeroshot_lars_pq_predremap) | finished | — | one logged step + evidence files | ✅ online values match the reported results |

<!-- wandb-links:end -->
