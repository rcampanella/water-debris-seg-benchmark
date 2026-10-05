# Water-surface debris segmentation — benchmark report

**Status:** scaffold (no eval numbers yet).  
**Generated:** 2026-10-05 (Stage 0).

## Hard rule

Train EoMT/VidEoMT masks **only on LaRS**. On FloW, CANSURF, WSODD: **eval only** via mask→box mAP.

## Headline results (pending)

| Method | LaRS PQ / mIoU | FloW mAP | CANSURF mAP | WSODD rubbish mAP | Source |
|---|---|---|---|---|---|
| EoMT zero-shot (COCO/ADE) | — | — | — | — | ablations/01_zeroshot_lars |
| EoMT LaRS semantic | — | — | — | — | ablations/02_finetune_lars_semantic |
| EoMT LaRS panoptic | — | — | — | — | ablations/03_finetune_lars_panoptic |
| EoMT LaRS → box sets | — | — | — | — | ablations/09_cross_dataset_transfer |
| VidEoMT LaRS | — | — | — | — | ablations/07_videomt_temporal |
| YOLOv11 specialized | — | — | — | — | ablations/08_specialized_yolo |

Do not fill with invented values.

## Experiment catalog

| ID | Folder | Question (short) | Expected outcome | Status |
|---|---|---|---|---|
| 01 | [01_zeroshot_lars](ablations/01_zeroshot_lars/README.md) | Generic VFM transfer to maritime masks? | Low PQ on debris things; water/sky reasonable | not run |
| 02 | [02_finetune_lars_semantic](ablations/02_finetune_lars_semantic/README.md) | Does LaRS semantic FT help obstacle/water/sky? | mIoU ↑ vs zero-shot | not run |
| 03 | [03_finetune_lars_panoptic](ablations/03_finetune_lars_panoptic/README.md) | Panoptic things for USV obstacles? | PQ ↑ on buoys/boats/floats | not run |
| 04 | [04_flow_mask2box](ablations/04_flow_mask2box/README.md) | Segmenter as detector on inland waste? | mAP &lt; YOLO; AP_S hard | not run |
| 05 | [05_cansurf_small_object](ablations/05_cansurf_small_object/README.md) | Far-field reflective cans? | Recall gap vs YOLO+SAHI | not run |
| 06 | [06_wsodd_distractors](ablations/06_wsodd_distractors/README.md) | Rubbish vs buoys/boats? | High FP on distractors if untuned | not run |
| 07 | [07_videomt_temporal](ablations/07_videomt_temporal/README.md) | Temporal vs still EoMT on LaRS? | Stable thing IDs; similar PQ | not run |
| 08 | [08_specialized_yolo](ablations/08_specialized_yolo/README.md) | In-domain detector ceiling? | Beats zero-shot EoMT on box sets | not run |
| 09 | [09_cross_dataset_transfer](ablations/09_cross_dataset_transfer/README.md) | LaRS-trained → other domains? | Domain gap visible on cans/waste | not run |

## Protocol references

- Spec: [`docs/water_debris_seg_benchmark.md`](../../../docs/water_debris_seg_benchmark.md)
- Disk: [`data/water_debris/DISK_BUDGET.md`](../../../data/water_debris/DISK_BUDGET.md)
- GPU: [GPU_INVENTORY.md](GPU_INVENTORY.md)

## Caveats

- FloW/WSODD downloads may be blocked — mark ablations **not run** in decision log.
- Current disk (~19G avail on umbrella) insufficient for full data + training without cleanup or extra quota.
