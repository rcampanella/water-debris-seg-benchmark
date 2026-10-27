# Fix 3 — Train EoMT on FloW and CANSURF with SAM masks; train YOLOv11s with the same protocol

> **Results (2026-10-26):** all runs of this fix are done; values with links to their JSON are in [RESULTS.md](RESULTS.md). ⏳ cells below are the plan as written before the runs.

**Status: plan, not started. Decisions of 2026-10-23: the REPORT's "hard rule" is revised (EoMT may train on FloW / CANSURF with box-prompted SAM masks, after the quality check); YOLOv11s is the YOLO in the comparison.** Back to the summary: [README.md](README.md).

## Motivation

On FloW and CANSURF, YOLO was trained in-domain (ablation 08) while EoMT was only tested there after LaRS training (ablations 04, 05, 09). The gap (FloW: YOLO 0.390 vs best EoMT 0.082 mAP; CANSURF: 0.857 vs 0.095) mixes two effects, **model** and **training data**, so it does not answer the research question. Both families must be trained on the same images and tested on the same ground truth.

FloW and CANSURF only have boxes. EoMT needs masks to train, so masks are produced by **SAM prompted with each ground-truth box** (pseudo-labels). Testing stays on the **native boxes** (detection only).

## Change to a recorded design decision

[REPORT.md](../REPORT.md) states: *"Train EoMT/VidEoMT masks only on LaRS … Training on rectangle or SAM-filled 'masks' would optimize painting the box, not the object."* This fix reverses that rule. The concern is addressed by:

- prompting SAM with the box (it segments the object inside it), **not** filling the rectangle;
- a **quality check before any training** (below), with a stop rule;
- testing only on native boxes, never on the pseudo-masks;
- a decision-log entry recording the change and its reason.

## Method

### Step 1 — SAM pseudo-masks (training splits only)

| Item | Value |
|---|---|
| Model | ⏳ to confirm: SAM 2.1 (large) through Ultralytics' SAM interface, box prompts; weights and SHA-256 recorded. The **same frozen SAM** is used by fix 2 (YOLO + SAM on LaRS) |
| Images | FloW `training/` (1,200 images) and CANSURF `train/` (4,950 images). Test images get **no** masks |
| Output | COCO instance json with RLE masks, one per GT box; `iscrowd` copied |
| Cost | ~6,150 images; ⏳ about 1 GPU hour (to be measured in the pilot) |

### Step 2 — Quality check (pilot, before any training)

| Check | Measure | Stop rule (proposed) |
|---|---|---|
| Mask fits its box | IoU between each SAM mask's min/max box and its GT box; distribution by object size | **Reported as the SAM-box ceiling, not a stop rule** (revised 2026-10-23: GT boxes are looser than the objects; correct masks give ~0.8 on FloW) |
| Empty or tiny masks | share of boxes with an empty mask, or a mask < 10% of the box | > 5%: stop and report |
| Visual check | 20 images, fixed selection (smallest objects, most objects, random seed 0), mask over image | rendered for your review |

The mask-box IoU also sets a **ceiling** for EoMT on these datasets: a perfectly learned SAM mask still gives that box, so AP at strict IoU thresholds is capped. Reported next to the results.

### Step 3 — Training (both models, same splits and protocol)

| | FloW | CANSURF |
|---|---|---|
| Train / held-out | `training/` 1,200 → 1,080 train / 120 held-out (the existing seeded split, [`flow_val_split.txt`](../ablations/08_specialized_yolo/sources/flow_val_split.txt)) | `train/` 4,950 → 90% / 10% held-out, ⏳ grouped by source sequence if CANSURF frames come from videos (checked first) |
| Test (scored once) | official test, 800 images | val, aligned GT, 2,209 images |
| Protocol | stage A on train90 → epoch count; stage B on the full training split, last checkpoint | same |

| Model | Training labels | Starting weights | Notes |
|---|---|---|---|
| **EoMT** | SAM masks, instance (things only; one class per dataset) | ⏳ to confirm: the same COCO-pretrained EoMT used to start ablations 02/03 | Same EoMT code path as ablation 03; config recorded |
| **YOLOv11s** | **native GT boxes** (not SAM boxes, which would only add SAM's errors) | `yolo11s.pt` | The CANSURF paper's model ([`train.py`](https://github.com/ZaidAljundiHW2/CANSURF): `yolo11s.pt`, imgsz 640, batch 64, 150 epochs, early-stopping patience 15). We keep imgsz 640 and batch 64, but replace the paper's early stopping on its validation split with our held-out stage A/B protocol. Replaces ablation 08's YOLOv11n (50 epochs, batch 16) and **fixes ablation 08's CANSURF checkpoint selection on the scored split** (≈ +0.006 mAP inflation) |
| YOLOv11n | — | — | Not trained (decision 2026-10-23); ablation 08's YOLOv11n stays as a reference |
| EoMT LoRA | SAM masks | same as EoMT | Second arm, trained alongside: [fix 4](FIX4_EOMT_LORA.md). Both arms scale EoMT's attention-mask annealing to the run length |
| VidEoMT | — | — | Not trained: FloW-Img is still images; CANSURF frames ⏳ to check |

### Step 4 — Scoring

Both models, test split, `eval_coco_bbox.py` (pycocotools): mAP@[.50:.95], AP50, AP75, AP_small, AR100. EoMT masks → min/max boxes, **with real confidences from EoMT's instance inference** (class probability × mean mask probability, top 100), not the constant 0.5 used in ablations 04/05/09 (see [fix 1b](FIX1_LARS_SPLIT_METRICS.md#5-consequence-for-flow-and-cansurf-ablations-04-05-09)). Same GT files as those ablations.

## Expected outputs

| Output | Path |
|---|---|
| SAM masks + QC report | `fairness_fixes/sam_masks/{flow,cansurf}_train_sam.json`, `fairness_fixes/SAM_QC.md`, `fairness_fixes/sam_qc/*.png` |
| Training configs and run specs | `fairness_fixes/runs/<model>_<dataset>_stage{A,B}/` |
| Predictions and metrics | `fairness_fixes/sources/fix3_<model>_<dataset>_test.json`, `.metrics.json` |
| Results page | `fairness_fixes/FIX3_RESULTS.md` |
| Decision-log entry | `docs/water_debris_seg_decision_log.md` |

**Target table (test splits, pycocotools mAP):**

| Model | Trained on | FloW mAP | FloW AP_small | FloW AR100 | CANSURF mAP | CANSURF AR100 | W&B |
|---|---|---|---|---|---|---|---|
| YOLOv11s (two checkpoints: FloW-trained, CANSURF-trained) | native boxes, held-out protocol | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ group `fix3_sam_box_datasets` |
| **EoMT** (two checkpoints: FloW-trained, CANSURF-trained) | **SAM masks** | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| SAM-mask box ceiling | — | ⏳ (AP of SAM boxes vs GT on held-out) | — | — | ⏳ | — | — |
| *Reference (not in the main table):* YOLOv11n (ablation 08) | native boxes | 0.390 | 0.249 | 0.491 | 0.857 | 0.870 | [rescore_flow_yolo11n](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/rescore_flow_yolo11n), [rescore_cansurf_yolo11n](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/rescore_cansurf_yolo11n) |
| *Previous (constant score, superseded by fix 1b; semantic dropped from the main table):* EoMT LaRS semantic (cross-domain) | LaRS | 0.082 | 0.044 | 0.281 | 0.063 | 0.173 | [rescore_flow_eomt_semantic](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/rescore_flow_eomt_semantic), [rescore_cansurf_eomt_semantic](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/rescore_cansurf_eomt_semantic) |
| *Previous (constant score, superseded by fix 1b):* EoMT COCO zero-shot | COCO | 0.034 | 0.012 | 0.093 | 0.095 | 0.140 | [rescore_flow_eomt_zeroshot](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/rescore_flow_eomt_zeroshot), [rescore_cansurf_eomt_zeroshot](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/rescore_cansurf_eomt_zeroshot) |

## References (existing)

| What | Where |
|---|---|
| YOLO in-domain results and training | [ablation 08 README](../ablations/08_specialized_yolo/README.md); training runs [cansurf_train_20261006_104548](https://wandb.ai/riccardo-campanella/-home-rcampanella-projects-water-debris-seg-benchmark-eval_outputs-water_debris_seg-ablations-08_specialized_yolo/runs/cansurf_train_20261006_104548), [flow_train_20261012_153258](https://wandb.ai/riccardo-campanella/-home-rcampanella-projects-water-debris-seg-benchmark-eval_outputs-water_debris_seg-ablations-08_specialized_yolo/runs/flow_train_20261012_153258) (older auto-named W&B project); FloW test result [ab08_yolo_flow_test_split](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/ab08_yolo_flow_test_split) |
| EoMT cross-domain results | [ablation 04](../ablations/04_flow_mask2box/README.md), [05](../ablations/05_cansurf_small_object/README.md), [09](../ablations/09_cross_dataset_transfer/README.md); rescoring audit [RESCORE.md](../RESCORE.md) |
| GT files and scorer | `box_eval_registry.json` (GT paths and hashes), `water-debris-seg/scripts/eval_coco_bbox.py` |
| Data | `/scratch-shared/rcampanella/water-debris-seg/data/{flow,cansurf}/`; FloW access [FLOW_DATA.md](../../../water-debris-seg/docs/FLOW_DATA.md) |

## Risks and caveats

- **Pseudo-labels.** EoMT learns SAM's idea of the object. The box ceiling quantifies the cost at the evaluation step; it does not catch SAM errors that still fit the box.
- **Small objects.** FloW bottles and distant cans are small; SAM is weakest there. The QC by object size decides whether small objects stay in EoMT training (option: drop masks below a size, keep boxes for YOLO).
- **Different supervision.** YOLO trains on exact boxes, EoMT on masks whose boxes are approximate. Documented, not hidden: the target table shows the SAM box ceiling next to EoMT.
- **Compute** (scaled from measured jobs: EoMT panoptic on LaRS 41 min for 12 epochs, YOLOv11s on LaRS 54 + 29 min). SAM ~0.5–1 GPU h; EoMT ~1–1.5 h (FloW) and ~4–6 h (CANSURF) for stage A + B; YOLOv11s ~0.7 h (FloW) and ~2.7 h (CANSURF).
- **Scratch purge.** FloW and CANSURF live on scratch; any new SAM masks and checkpoints are written to home.
