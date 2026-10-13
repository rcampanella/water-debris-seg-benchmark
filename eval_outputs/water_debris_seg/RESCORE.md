# Box-metric re-score (pycocotools)

Generated 2026-10-13T12:45:56 by `water-debris-seg/scripts/rescore_box_predictions.py` from [`box_eval_registry.json`](box_eval_registry.json). Machine-readable: [`RESCORE.json`](RESCORE.json) (includes sha256 of every prediction and GT file).

**Why:** the pre-2026-10-13 scorer (`eval_coco_bbox_legacy.py`) averaged only AP@.50/.75, never counted false positives in AP_small, and skipped images without predictions. See [decision log](../../docs/water_debris_seg_decision_log.md) 2026-10-13. Legacy metrics files are kept beside the new ones as `*.legacy.json`.

| Run | Dataset | Method | mAP legacy → **new** | AP50 | AP75 | AP_small legacy → **new** | AR100 | Images with preds | Preds | Metrics file |
|---|---|---|---|---|---|---|---|---|---|---|
| flow_eomt_zeroshot ★ | flow_test | EoMT zero-shot | 0.086 → **0.034** | 0.103 | 0.015 | 0.168 → **0.012** | 0.093 | 478/800 | 996 | [eomt_zeroshot_flow_preds_coco.metrics.json](ablations/04_flow_mask2box/sources/eomt_zeroshot_flow_preds_coco.metrics.json) |
| flow_eomt_semantic ★ | flow_test | EoMT semantic FT | 0.133 → **0.082** | 0.229 | 0.040 | 0.425 → **0.044** | 0.281 | 800/800 | 2911 | [eomt_lars_semantic_flow_preds_coco.metrics.json](ablations/09_cross_dataset_transfer/sources/eomt_lars_semantic_flow_preds_coco.metrics.json) |
| flow_eomt_panoptic ★ | flow_test | EoMT panoptic FT | 0.040 → **0.014** | 0.044 | 0.006 | 0.130 → **0.007** | 0.051 | 419/800 | 682 | [eomt_lars_panoptic_flow_preds_coco.metrics.json](ablations/04_flow_mask2box/sources/eomt_lars_panoptic_flow_preds_coco.metrics.json) |
| flow_yolo11n ★ | flow_test | YOLOv11n (FloW-trained) | 0.566 → **0.390** | 0.815 | 0.315 | 0.858 → **0.249** | 0.491 | 800/800 | 18029 | [yolo11n_flow_test_preds_coco.metrics.json](ablations/08_specialized_yolo/sources/yolo11n_flow_test_preds_coco.metrics.json) |
| cansurf_yolo11n ★ | cansurf_val_aligned | YOLOv11n (CANSURF-trained) | 0.917 → **0.857** | 0.960 | 0.865 | 0.368 → **0.060** | 0.870 | 2209/2209 | 30206 | [yolo11n_cansurf_val_preds_coco.metrics.json](ablations/08_specialized_yolo/sources/yolo11n_cansurf_val_preds_coco.metrics.json) |
| cansurf_eomt_zeroshot ★ | cansurf_val_aligned | EoMT zero-shot | 0.148 → **0.095** | 0.148 | 0.090 | 0.083 → **0.000** | 0.140 | 1502/2209 | 3766 | [eomt_zeroshot_yolo_aligned_gt.metrics.json](ablations/05_cansurf_small_object/sources/eomt_zeroshot_yolo_aligned_gt.metrics.json) |
| cansurf_eomt_zeroshot_origgt | cansurf_val_original | EoMT zero-shot | 0.147 → **0.095** | 0.148 | 0.090 | 0.083 → **0.000** | 0.140 | 1512/2221 | 3782 | [eomt_zeroshot_val_preds_coco.metrics.json](ablations/05_cansurf_small_object/sources/eomt_zeroshot_val_preds_coco.metrics.json) |
| cansurf_eomt_semantic | cansurf_val_original | EoMT semantic FT | 0.085 → **0.064** | 0.132 | 0.049 | 0.632 → **0.001** | 0.173 | 2221/2221 | 8833 | [eomt_lars_semantic_cansurf_preds_coco.metrics.json](ablations/09_cross_dataset_transfer/sources/eomt_lars_semantic_cansurf_preds_coco.metrics.json) |
| cansurf_eomt_semantic_aligned ★ | cansurf_val_aligned | EoMT semantic FT | 0.085 → **0.063** | 0.128 | 0.049 | 0.632 → **0.001** | 0.173 | 2209/2209 | 8790 | [eomt_lars_semantic_cansurf_aligned_gt.metrics.json](ablations/09_cross_dataset_transfer/sources/eomt_lars_semantic_cansurf_aligned_gt.metrics.json) |
| cansurf_eomt_semantic_min256 ★ | cansurf_val_aligned | EoMT semantic FT (min-area 256) | 0.108 → **0.078** | 0.160 | 0.061 | 0.632 → **0.004** | 0.173 | 2209/2209 | 6583 | [eomt_lars_semantic_cansurf_min256_preds.metrics.json](ablations/09_cross_dataset_transfer/sources/eomt_lars_semantic_cansurf_min256_preds.metrics.json) |
| cansurf_eomt_panoptic ★ | cansurf_val_aligned | EoMT panoptic FT | 0.120 → **0.034** | 0.069 | 0.021 | 0.133 → **0.001** | 0.047 | 1005/2209 | 1395 | [eomt_lars_panoptic_yolo_aligned_gt.metrics.json](ablations/05_cansurf_small_object/sources/eomt_lars_panoptic_yolo_aligned_gt.metrics.json) |
| cansurf_eomt_panoptic_origgt | cansurf_val_original | EoMT panoptic FT | 0.120 → **0.033** | 0.069 | 0.021 | 0.133 → **0.001** | 0.047 | 1014/2221 | 1406 | [eomt_lars_panoptic_ft_val_preds_coco.metrics.json](ablations/05_cansurf_small_object/sources/eomt_lars_panoptic_ft_val_preds_coco.metrics.json) |

★ = headline row quoted in REPORT.md. mAP = AP@[.50:.95]; AP_small = AP@[.50:.95] on GT area < 32².
