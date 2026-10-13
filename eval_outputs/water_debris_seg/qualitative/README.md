# Qualitative error analysis

Inputs → outputs with images, organised by *why* predictions fail. Every figure and count here traces back to a scored prediction file (sha256 in each `errors/<run>/summary.json`) or to a checkpoint (`box_eval_registry.json` → `model`). Generated 2026-10-13.

## Browse

| What | Where |
|---|---|
| Box-error panels (GT vs every method, TP/FP/FN by type, zoom, provenance footer) | [`INDEX.md`](INDEX.md) → `panels/<dataset>/` (78 images) |
| Mask panels (the masks behind each EoMT box; LaRS pixel-GT comparisons) | [`MASKS.md`](MASKS.md) → `mask_panels/<dataset>/` |
| GT outcome breakdown charts | [`error_breakdown_flow_test.png`](error_breakdown_flow_test.png), [`error_breakdown_cansurf_val_aligned.png`](error_breakdown_cansurf_val_aligned.png) |
| Counts per run (for slides/charts) | [`error_summary.csv`](error_summary.csv) |
| Per-box tables | `errors/<run_id>/detections.csv`, `errors/<run_id>/gt.csv` |
| Why each image was chosen | [`selection.json`](selection.json) (rule + numbers per pick; 6 seeded random images per dataset as unbiased reference) |
| LaRS per-image / per-object tables | [`lars_val_per_image.csv`](lars_val_per_image.csv), [`lars_val_things.csv`](lars_val_things.csv) |
| Raw masks | `masks/<dataset>/<run_id>/*.npz` (+ `consistency.json`), `masks/lars_val/{semantic_lars,panoptic_lars}/*.npz` |

## Error taxonomy

Matching is pycocotools' own (IoU 0.5, max 100 dets/image), so TP counts reconcile with the metrics files (recall@0.5 identical to COCOeval, checked). Definitions in `water-debris-seg/water_debris_seg/qualitative/errors.py`.

| Detections | GT boxes |
|---|---|
| **fp_merge** box contains centres of ≥2 GT | **fn_merged** centre inside an fp_merge box |
| **fp_dup** IoU ≥ 0.5 with an already-matched GT | **fn_frag** ≥2 dets lie mostly inside it, none matched |
| **fp_loc** best IoU 0.1–0.5 | **fn_loc** best IoU 0.1–0.5 |
| **fp_bkg** best IoU < 0.1 | **fn_missed** best IoU < 0.1 |

Operating point: YOLO score ≥ 0.25 (Ultralytics default); EoMT all boxes (fixed score 0.5).

## Findings (what the images show)

**1. Two distinct EoMT failure modes, not one.** Share of GT boxes (IoU 0.5):

| Dataset | Method | found | merged | missed |
|---|---|---|---|---|
| FloW | EoMT zero-shot | 20% | 1% | 75% |
| FloW | EoMT semantic FT | **56%** | **24%** | 10% |
| FloW | EoMT panoptic FT | 12% | 5% | 80% |
| FloW | YOLOv11n | 81% | 2% | 12% |
| CANSURF | EoMT zero-shot | 20% | 5% | 73% |
| CANSURF | EoMT semantic FT | 29% | **65%** | 3% |
| CANSURF | EoMT panoptic FT | 9% | 4% | 86% |
| CANSURF | YOLOv11n | 94% | 1% | 4% |

- *Instance heads (zero-shot, panoptic FT) don't fire on debris* → recognition failure.
- *The semantic "obstacle" path sees almost all debris but cannot separate touching objects*; connected components fuse rows of bottles / piles of cans (mask panels make this explicit) → instance-separation failure.

**2. Semantic FT boxes are found but loose.** FloW recall@0.5 is 56% vs YOLO 81%, a much smaller gap than mAP (0.082 vs 0.390): mAP is dragged by merges, loose boxes (AP75 0.040) and the fixed score.

**3. Shoreline and skyline become one giant false box.** The semantic obstacle class includes static shore, so a single CC spanning the horizon produces an image-wide `fp_bkg` (e.g. FloW `000212.jpg`, mask panel 16).

**4. On LaRS itself, the panoptic model misses most small objects.** PQ 0.395 hides thing recall of **35%** (388/1107 at IoU > 0.5); by size: small **14%** (96/698), medium 64%, large 85%. Small objects are 63% of LaRS val things.

**5. LaRS barely contains debris.** LaRS val has **3 "Float" instances** (0 found), 73 "Other" (8% found), 125 buoys (15%). LaRS-only training cannot teach "debris"; it teaches "boat" (42%).

**6. Conditions (LaRS val, ab02 semantic obstacle IoU / ab03 thing recall).** All 0.936 / 0.35; dusk 0.790 / 0.32 (n = 9); fog 0.903 / 0.26 (n = 12); wakes 0.880 / 0.51 (n = 18); glitter 0.937 / 0.30 (n = 16). Strata are small, so treat these as hypotheses, not effects.

**7. CANSURF val is partly synthetic and repetitive.** Sources include sprite composites on white backgrounds and indoor scenes; Roboflow augmentations repeat a source up to 9× within val (492 distinct augmented sources for 1294 files). Train/val overlap is negligible (2 sources, 7 files).

## Design implications to discuss

- Instance separation on top of the semantic obstacle mask (e.g. a class-agnostic instance head trained on LaRS things, or watershed/distance-transform splitting) attacks the dominant semantic-path error.
- Remove static shore/sky-adjacent regions before CC→box (water-boundary mask, horizon prior).
- Export real mask scores for EoMT so AP and PR curves are meaningful.
- Debris needs debris-bearing supervision; LaRS alone has ~no floats. Revisit the hard rule's consequences, or add pixel labels for a FloW/CANSURF subset.
- Small objects: higher input resolution / tiling (also for YOLO on FloW at 640 px).

## Regenerate

```bash
cd water-debris-seg
python scripts/rescore_box_predictions.py          # metrics (RESCORE.md)
python scripts/qual_error_analysis.py              # errors/, error_summary.csv, selection.json
python scripts/qual_render_panels.py               # panels/, INDEX.md, breakdown charts
sbatch scripts/snellius/qual_dump_masks.sbatch     # masks/ (GPU, ~6 min)
python scripts/qual_render_masks.py all            # mask_panels/, MASKS.md, lars_val_*.csv
```
