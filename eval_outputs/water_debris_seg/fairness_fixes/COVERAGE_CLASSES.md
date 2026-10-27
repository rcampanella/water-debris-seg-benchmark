# Data and class coverage: which model is trained on what, tested on what, with which classes

> **Results (2026-10-26):** all runs of this fix are done; values with links to their JSON are in [RESULTS.md](RESULTS.md). ⏳ cells below are the plan as written before the runs.

**Status: reference for audit (updated 2026-10-23).** Back to the summary: [README.md](README.md). The proposals in § 4 were agreed on 2026-10-23; § 6 lists the classes under those decisions. §§ 1 and 3 describe the state before the fixes.

## 1. Coverage matrix: training data × test dataset

Each row is one training source; each cell says whether that model has been (or will be) tested on that dataset, and where.

| Model | Trained on | Tested on **LaRS** | Tested on **FloW** | Tested on **CANSURF** |
|---|---|---|---|---|
| EoMT | **COCO only** (zero-shot) | ✅ done ([01](../ablations/01_zeroshot_lars/README.md)); ⏳ mIoU + box AP ([fix 1](FIX1_LARS_SPLIT_METRICS.md)) | ✅ done ([04](../ablations/04_flow_mask2box/README.md)); ⏳ re-score ([1b](FIX1_LARS_SPLIT_METRICS.md#5-consequence-for-flow-and-cansurf-ablations-04-05-09)) | ✅ done ([05](../ablations/05_cansurf_small_object/README.md)); ⏳ re-score (1b) |
| EoMT | **LaRS** panoptic | ✅ done ([03](../ablations/03_finetune_lars_panoptic/README.md)); ⏳ mIoU + box AP + SAM control (fix 1) | ✅ cross-domain ([04](../ablations/04_flow_mask2box/README.md), [09](../ablations/09_cross_dataset_transfer/README.md)); ⏳ re-score (1b) | ✅ cross-domain ([05](../ablations/05_cansurf_small_object/README.md), [09](../ablations/09_cross_dataset_transfer/README.md)); ⏳ re-score (1b) |
| EoMT | **LaRS** semantic | ✅ done ([02](../ablations/02_finetune_lars_semantic/README.md)); **dropped from the main table** (2026-10-23), kept as reference | ✅ cross-domain ([09](../ablations/09_cross_dataset_transfer/README.md)) | ✅ cross-domain ([09](../ablations/09_cross_dataset_transfer/README.md)) |
| EoMT | **FloW** (SAM masks) | not needed | ⏳ [fix 3](FIX3_SAM_BOX_DATASETS.md) | not needed |
| EoMT | **CANSURF** (SAM masks) | not needed | not needed | ⏳ [fix 3](FIX3_SAM_BOX_DATASETS.md) |
| YOLO | **COCO only** (zero-shot, `yolo11n.pt` / `yolo11s.pt` as released) | ❌ never tested | ❌ never tested | ❌ never tested |
| YOLO | **LaRS** (9 classes; YOLOv11s in the comparison, YOLOv11n reference) | ✅ box AP done ([10](../ablations/10_yolo_lars_boxes/README.md)); ⏳ YOLO + SAM 11 classes for PQ/mIoU ([fix 2](FIX2_YOLO_ON_LARS.md)) | ⏳ cross-domain ([fix 2](FIX2_YOLO_ON_LARS.md#open-item-b--lars-trained-yolo-on-flow-and-cansurf-cross-domain-row)) | ⏳ cross-domain (fix 2) |
| YOLO | **FloW** (YOLOv11n done; YOLOv11s planned) | not needed | ✅ YOLOv11n done ([08](../ablations/08_specialized_yolo/README.md)); ⏳ YOLOv11s (fix 3) | not needed |
| YOLO | **CANSURF** (YOLOv11n done; YOLOv11s planned) | not needed | not needed | ✅ YOLOv11n done ([08](../ablations/08_specialized_yolo/README.md)); ⏳ YOLOv11s (fix 3) |

**Gaps this exposes**

- **EoMT and YOLO were never trained on the same data until ablation 10** (YOLO on LaRS); fix 3 adds the FloW / CANSURF pairs.
- **YOLO zero-shot (COCO only) has never been tested anywhere**, while EoMT zero-shot has been tested on all three datasets. Without it there is no zero-shot YOLO-vs-EoMT row. It would be inference only (three datasets, < 1 h) and would hit the same 80-class problem as EoMT zero-shot (§ 4).
- **Cross-domain is only one-way**: LaRS → FloW / CANSURF. FloW ↔ CANSURF and FloW / CANSURF → LaRS are marked not needed.
- **YOLO's size differs from the CANSURF paper**: the paper trains YOLOv11s ([`train.py`](https://github.com/ZaidAljundiHW2/CANSURF), `yolo11s.pt`, 150 epochs, batch 64); ablation 08 used YOLOv11n (50 epochs, batch 16). Decided 2026-10-23: YOLOv11s is the YOLO in the comparison.

## 2. Classes each training source can predict (complete lists)

| Training source | # classes | Classes |
|---|---|---|
| **COCO (EoMT zero-shot, panoptic)** — objects | 80 | person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush |
| **COCO (EoMT zero-shot, panoptic)** — regions | 53 | banner, blanket, bridge, cardboard, counter, curtain, door-stuff, floor-wood, flower, fruit, gravel, house, light, mirror-stuff, net, pillow, platform, playingfield, railroad, river, road, roof, sand, sea, shelf, snow, stairs, tent, towel, wall-brick, wall-stone, wall-tile, wall-wood, water-other, window-blind, window-other, tree-merged, fence-merged, ceiling-merged, sky-other-merged, cabinet-merged, table-merged, floor-other-merged, pavement-merged, mountain-merged, grass-merged, dirt-merged, paper-merged, food-other-merged, building-other-merged, rock-merged, wall-other-merged, rug-merged |
| **COCO (YOLO zero-shot)** | 80 | the same 80 COCO object classes (no regions) |
| **LaRS panoptic (EoMT)** | 11 | regions: Static Obstacle, Water, Sky; objects: Boat/ship, Row boats, Paddle board, Buoy, Swimmer, Animal, Float, Other |
| **LaRS semantic (EoMT)** | 3 | obstacle, water, sky |
| **LaRS boxes (YOLO, ablation 10)** | 9 | Static Obstacle, Boat/ship, Row boats, Paddle board, Buoy, Swimmer, Animal, Float, Other |
| **LaRS boxes + water/sky (YOLO + SAM, fix 2)** | 11 | the 9 above + Water, Sky |
| **FloW (YOLO, EoMT fix 3)** | 1 | waste (FloW GT name `waste`) |
| **CANSURF (YOLO, EoMT fix 3)** | 1 | can (CANSURF GT name `object`; `data.yaml` name `debris`) |

Sources: COCO-80 names from Ultralytics `cfg/datasets/coco.yaml`; COCO-panoptic regions = the standard 53 COCO-panoptic 2017 stuff categories predicted by EoMT's `coco_panoptic_eomt_base_640_2x.bin`; LaRS from `lars/val/panoptic_annotations.json`; FloW and CANSURF from the GT files in `box_eval_registry.json`.

## 3. Classes counted in each evaluation (current rule)

| Model trained on → tested on | Ground-truth classes | Predicted classes that count today | Source of the rule |
|---|---|---|---|
| EoMT COCO → **LaRS** | LaRS 11 | 55 COCO classes mapped to LaRS (below); the other 78 COCO classes are ignored | `configs/coco133_to_lars_pred_map.json` |
| EoMT COCO → **FloW / CANSURF** | 1 (waste / can) | **all 80 COCO object classes**, every one scored as debris | `eval_eomt_cansurf_mask2box.py` (`preds_to_detections`, `category_id=0`) |
| EoMT LaRS panoptic → **LaRS** | LaRS 11 | LaRS 11, per class | EoMT validation; fix 1 |
| EoMT LaRS panoptic → **FloW / CANSURF** | 1 | **all 8 LaRS object classes** (Boat/ship, Row boats, Paddle board, Buoy, Swimmer, Animal, Float, Other) as debris | same script |
| EoMT LaRS semantic → **LaRS** | 3 | 3 | EoMT validation |
| EoMT LaRS semantic → **FloW / CANSURF** | 1 | **every obstacle region** (connected component), shoreline included, as debris | `eval_eomt_cansurf_semantic_mask2box.py` |
| EoMT FloW / CANSURF (SAM) → own dataset | 1 | 1 | fix 3 |
| YOLO LaRS → **LaRS** | 9 (8 objects + Static Obstacle) | 9, per class | `eval_yolo_lars.py` |
| YOLO LaRS → **FloW / CANSURF** | 1 | planned: all 8 object classes (fix 2) — see § 4 | fix 2 |
| YOLO FloW / CANSURF → own dataset | 1 | 1 | ablation 08 |

**COCO → LaRS mapping used on LaRS** (55 COCO classes; all other predictions ignored):

| LaRS class | COCO classes mapped to it |
|---|---|
| Static Obstacle | bridge, house, building-other-merged, tree-merged, fence-merged, mountain-merged, rock-merged, wall-brick, wall-stone, wall-tile, wall-wood, wall-other-merged, roof, platform, pavement-merged, road, railroad, sand, grass-merged, dirt-merged, gravel, playingfield, snow, tent, stairs, door-stuff, window-other, window-blind, net |
| Water | river, sea, water-other |
| Sky | sky-other-merged |
| Boat/ship | boat |
| Paddle board | surfboard |
| Swimmer | person |
| Animal | bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe |
| Other | bottle, cup, frisbee, sports ball, kite, backpack, handbag, suitcase, umbrella |
| Row boats, Buoy, Float | — (no COCO counterpart) |

## 4. Classes that do not belong in an evaluation, and proposals

| Where | Problem | Proposal |
|---|---|---|
| EoMT zero-shot on FloW/CANSURF | All 80 COCO object classes count as debris: cars, trains, people, giraffes, toasters | Count only debris-like COCO classes (bottle, cup, sports ball, frisbee, … the existing "Other" mapping) |
| EoMT LaRS models on FloW/CANSURF | Boats, swimmers, paddle boards and animals count as debris. They're maritime, but not debris | Count only **Float + Other**, the taxonomy's debris classes |
| EoMT semantic on FloW/CANSURF | Every obstacle region counts, shoreline included | Drop with the semantic model (proposed) |
| YOLO LaRS → FloW/CANSURF (planned, fix 2) | Planned "all 8 object classes", which would copy the problem above | Same debris-only rule: Float + Other |
| YOLO zero-shot (if added) | Same 80-class problem as EoMT zero-shot | Same debris-like COCO subset |
| EoMT zero-shot on LaRS | Mapping includes land classes: Animal ← cat, cow, sheep, horse, elephant, bear, zebra, giraffe; Static Obstacle ← road, railroad, playingfield, … | Tighten to maritime-plausible classes (e.g. Animal ← bird, dog), and report the score change |
| LaRS's own classes (Static Obstacle, Animal, …) | Maritime, but not debris | **Keep**: LaRS's own task. Report per class, plus a separate debris line (Float + Other) |

**Scoring rule proposed for every FloW / CANSURF evaluation** (EoMT fix 1b, YOLO fix 2): report **debris-only** (main; only the debris classes above count) and **class-agnostic** (reference; any object counts). The gap between them shows whether a model finds debris but names it wrong (class-agnostic higher) or whether non-debris predictions cost precision (debris-only higher). Every saved prediction keeps its **original class**; today's EoMT FloW / CANSURF files store only category 0, so the source class of e.g. the 996 zero-shot FloW detections is lost.

## 5. Common and unique classes

| | Classes |
|---|---|
| **Common to all three datasets** | No class name is shared. Only the **concept "floating debris"** is: FloW `waste`, CANSURF cans, LaRS **Float + Other** (benchmark taxonomy, [REPORT.md](../REPORT.md)) |
| **LaRS only** | Static Obstacle, Water, Sky, Boat/ship, Row boats, Paddle board, Buoy, Swimmer, Animal (obstacle-avoidance classes, not debris) |
| **FloW only** | waste (floating waste on inland rivers: bottles, plastics) |
| **CANSURF only** | cans (aluminium cans on the water surface; a subset of debris) |

LaRS val has only 3 Float and 73 Other segments, so debris is barely measurable on LaRS; the debris question rests on FloW and CANSURF.

## 6. Classes under the agreed decisions (2026-10-23)

Classes predicted by each model of the [main comparison table](README.md#main-comparison-table-decisions-enforced), and classes scored in each evaluation.

| Model | Predicts | LaRS: scored | FloW / CANSURF: scored (main, debris-only) | FloW / CANSURF: reference |
|---|---|---|---|---|
| EoMT zero-shot (COCO) | 133 COCO (80 objects + 53 regions) | PQ: 11 LaRS classes via the tightened map; mIoU: obstacle / water / sky; box AP: 8 objects / 9 classes via the map | 9 debris-like COCO classes: bottle, cup, frisbee, sports ball, kite, backpack, handbag, suitcase, umbrella | class-agnostic (all 80 COCO objects) |
| EoMT panoptic (LaRS) | 11 LaRS | PQ: 11; mIoU: 3 (obstacle = Static Obstacle + 8 objects); box AP: 8 / 9 | Float, Other *(cross)* | class-agnostic (8 LaRS objects) |
| EoMT panoptic + SAM (control) | 11 LaRS | PQ: 11; mIoU: 3 | not tested | — |
| EoMT, FloW (SAM masks) | waste | not tested | waste | — |
| EoMT, CANSURF (SAM masks) | can | not tested | can | — |
| YOLOv11s (LaRS) | 9: Static Obstacle + 8 objects | box AP: 8 / 9 | Float, Other *(cross)* | class-agnostic (8 LaRS objects) |
| YOLOv11s + SAM (LaRS) | 11: the 9 + Water, Sky | PQ: 11; mIoU: 3; box AP: 8 / 9 (water / sky boxes excluded) | not tested | — |
| YOLOv11s, FloW | waste | not tested | waste | — |
| YOLOv11s, CANSURF | can | not tested | can | — |
| *VidEoMT (temporal table)* | 8 LaRS objects | per-keyframe mask AP on 8 objects; flicker | not tested | — |

**Tightened COCO → LaRS mapping** (EoMT zero-shot on LaRS; replaces the mapping in § 3):

| LaRS class | COCO classes kept | Dropped (not maritime) |
|---|---|---|
| Boat/ship | boat | — |
| Paddle board | surfboard | — |
| Swimmer | person | — |
| Animal | bird, dog | cat, horse, sheep, cow, elephant, bear, zebra, giraffe |
| Other | bottle, cup, frisbee, sports ball, kite, backpack, handbag, suitcase, umbrella | — |
| Static Obstacle | bridge, house, building-other-merged, tree-merged, fence-merged, mountain-merged, rock-merged, wall-brick, wall-stone, wall-tile, wall-wood, wall-other-merged, roof, platform, pavement-merged, road, sand, grass-merged, dirt-merged, gravel, snow, tent, stairs, door-stuff, window-other, window-blind, net | railroad, playingfield (decision 2026-10-23: not maritime) |
| Water | river, sea, water-other | — |
| Sky | sky-other-merged | — |
| Row boats, Buoy, Float | — (no COCO counterpart; zero-shot scores 0 by construction) | — |

The score change from the tightened mapping is reported next to the zero-shot PQ (0.298 with the current mapping).

**Benchmark class groups** (taxonomy in [REPORT.md](../REPORT.md)):

| Group | Classes | Scored on |
|---|---|---|
| Debris (target) | LaRS Float + Other; FloW waste; CANSURF can | all three; measured mainly on FloW / CANSURF |
| Dynamic obstacles | Boat/ship, Row boats, Paddle board, Buoy, Swimmer, Animal | LaRS |
| Static context | Static Obstacle | LaRS |
| Water, Sky | Water, Sky | LaRS (PQ, mIoU) |
