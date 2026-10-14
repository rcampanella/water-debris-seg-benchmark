# W&B audit

Generated 2026-10-14T09:17:59+00:00 by `water-debris-seg/scripts/wandb_audit.py` (read-only) from [`wandb_result_map.json`](wandb_result_map.json), [`box_eval_registry.json`](box_eval_registry.json) and [`RESCORE.json`](RESCORE.json). Every benchmark run with its summary: [`wandb_runs.json`](wandb_runs.json). Entity `riccardo-campanella`; projects: `-home-rcampanella-projects-water-debris-…`, `videomt`, `water-debris-seg`.

Verdict meaning: **matches current** = online value equals the value now reported; **superseded** = online value is the pre-2026-10-13 value that was later restated (decision log 2026-10-13); **differs** = see note.

## 1. Reported result → W&B run

| Abl. | Result | Run | Metric | Online | Now reported | Pre-fix | Verdict |
|---|---|---|---|---|---|---|---|
| 01 | LaRS val PQ, zero-shot EoMT | [`txjoa9ef`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/txjoa9ef) | PQ (val_pq_all) | 0.0014 | 0.2982 | 0.0014 | ⚠️ online = superseded (legacy) value |
| 02 | LaRS val mIoU, semantic FT (validate run) | [`duguh1wo`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/duguh1wo) | mIoU (val_iou_all) | 0.9805 | 0.9805 | — | ✅ matches current |
| 02 | LaRS semantic FT training run (final val mIoU) | [`mk2w7dr8`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/mk2w7dr8) | mIoU (val_iou_all) | 0.9805 | 0.9805 | — | ✅ matches current |
| 03 | LaRS val PQ, panoptic FT (validate runs) | [`xcl7zvm8`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/xcl7zvm8) | PQ (val_pq_all) | 0.3945 | 0.3946 | — | ✅ matches current — current = independent re-computation (job 28011322); EoMT's own validate log reads 0.3945 |
| 03 | LaRS val PQ, panoptic FT (first validate run) | [`qmi79nsh`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/qmi79nsh) | PQ (val_pq_all) | 0.3945 | 0.3946 | — | ✅ matches current |
| 03 | LaRS panoptic FT training run (final val PQ) | [`67kk33hk`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/67kk33hk) | PQ (val_pq_all) | 0.3945 | 0.3946 | — | ✅ matches current |
| 08 | YOLOv11n CANSURF training run (final val) | [`cansurf_train_20261006_1`](https://wandb.ai/riccardo-campanella/-home-rcampanella-projects-water-debris-seg-benchmark-eval_outputs-water_debris_seg-ablations-08_specialized_yolo/runs/cansurf_train_20261006_104548) | mAP50-95 (Ultralytics) | 0.8610 | 0.8567 | — | ✅ matches current — Ultralytics and pycocotools AP differ by ~0.004 (interpolation); same split (CANSURF val) |
| 08 | YOLOv11n CANSURF val replay logged to W&B | [`n3vuqqbi`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/n3vuqqbi) | mAP50-95 (Ultralytics) | 0.8615 | 0.8567 | — | ✅ matches current |
| 08 | YOLOv11n FloW training run: online value is the 120-image val split, NOT the test split | [`flow_train_20261012_1532`](https://wandb.ai/riccardo-campanella/-home-rcampanella-projects-water-debris-seg-benchmark-eval_outputs-water_debris_seg-ablations-08_specialized_yolo/runs/flow_train_20261012_153258) | mAP50-95 (val split, last epoch) | 0.4655 | 0.3904 | — | ❌ differs |
| 07 | Flicker proxy, EoMT semantic FT | [`g1bfz3nn`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/g1bfz3nn) | flicker (mean proxy) | 0.0406 | 0.0406 | — | ✅ matches current |
| 07 | Flicker proxy, EoMT panoptic FT | [`5aoppsyb`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/5aoppsyb) | flicker (mean proxy) | 0.2291 | 0.2291 | — | ✅ matches current |
| 07 | Flicker proxy, VidEoMT 4k | [`wzpe700z`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/wzpe700z) | flicker (mean proxy) | 0.6120 | 0.6120 | — | ✅ matches current |
| 07 | Flicker proxy, VidEoMT 16k | [`r4ugnnd5`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/r4ugnnd5) | flicker (mean proxy) | 0.3108 | 0.3108 | — | ✅ matches current |
| 04 | FloW test, EoMT zero-shot mask→box | [`u45u5rne`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/u45u5rne) | mAP | 0.0856 | 0.0342 | 0.0856 | ⚠️ online = superseded (legacy) value |
| 04 | FloW test, EoMT zero-shot mask→box | [`u45u5rne`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/u45u5rne) | AP50 | 0.1495 | 0.1026 | 0.1495 | ⚠️ online = superseded (legacy) value |
| 04 | FloW test, EoMT zero-shot mask→box | [`u45u5rne`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/u45u5rne) | AP_small | 0.1679 | 0.0120 | 0.1679 | ⚠️ online = superseded (legacy) value |
| 04 | FloW test, EoMT panoptic FT mask→box | [`0c4pnnn0`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/0c4pnnn0) | mAP | 0.0400 | 0.0143 | 0.0400 | ⚠️ online = superseded (legacy) value |
| 04 | FloW test, EoMT panoptic FT mask→box | [`0c4pnnn0`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/0c4pnnn0) | AP50 | 0.0718 | 0.0441 | 0.0718 | ⚠️ online = superseded (legacy) value |
| 04 | FloW test, EoMT panoptic FT mask→box | [`0c4pnnn0`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/0c4pnnn0) | AP_small | 0.1296 | 0.0073 | 0.1296 | ⚠️ online = superseded (legacy) value |
| 09 | FloW test, EoMT semantic FT obstacle→box | [`hd5bs8yz`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/hd5bs8yz) | mAP | 0.1333 | 0.0824 | 0.1333 | ⚠️ online = superseded (legacy) value |
| 09 | FloW test, EoMT semantic FT obstacle→box | [`hd5bs8yz`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/hd5bs8yz) | AP50 | 0.2279 | 0.2294 | 0.2279 | ⚠️ online = superseded (legacy) value |
| 09 | FloW test, EoMT semantic FT obstacle→box | [`hd5bs8yz`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/hd5bs8yz) | AP_small | 0.4245 | 0.0440 | 0.4245 | ⚠️ online = superseded (legacy) value |
| 05 | CANSURF val (original 2221-image GT), EoMT zero-shot | [`lhnbhuej`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lhnbhuej) | mAP | 0.1473 | 0.0947 | 0.1473 | ⚠️ online = superseded (legacy) value |
| 05 | CANSURF val (original 2221-image GT), EoMT zero-shot | [`lhnbhuej`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lhnbhuej) | AP50 | 0.1844 | 0.1480 | 0.1844 | ⚠️ online = superseded (legacy) value |
| 05 | CANSURF val (original 2221-image GT), EoMT zero-shot | [`lhnbhuej`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lhnbhuej) | AP_small | 0.0833 | 0.0001 | 0.0833 | ⚠️ online = superseded (legacy) value |
| 05 | CANSURF val (original GT), EoMT panoptic FT | [`qi6ymm68`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/qi6ymm68) | mAP | 0.1203 | 0.0335 | 0.1203 | ⚠️ online = superseded (legacy) value |
| 05 | CANSURF val (original GT), EoMT panoptic FT | [`qi6ymm68`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/qi6ymm68) | AP50 | 0.1950 | 0.0688 | 0.1950 | ⚠️ online = superseded (legacy) value |
| 05 | CANSURF val (original GT), EoMT panoptic FT | [`qi6ymm68`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/qi6ymm68) | AP_small | 0.1333 | 0.0012 | 0.1333 | ⚠️ online = superseded (legacy) value |
| 09 | CANSURF val (original GT), EoMT semantic FT | [`vquavnrm`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/vquavnrm) | mAP | 0.0854 | 0.0635 | 0.0854 | ⚠️ online = superseded (legacy) value |
| 09 | CANSURF val (original GT), EoMT semantic FT | [`vquavnrm`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/vquavnrm) | AP50 | 0.1234 | 0.1316 | 0.1234 | ⚠️ online = superseded (legacy) value |
| 09 | CANSURF val (original GT), EoMT semantic FT | [`vquavnrm`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/vquavnrm) | AP_small | 0.6316 | 0.0007 | 0.6316 | ⚠️ online = superseded (legacy) value |
| 09 | CANSURF val (aligned GT), EoMT semantic FT, min-area 256 | [`hxuvlvcn`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/hxuvlvcn) | mAP | 0.1079 | 0.0780 | 0.1079 | ⚠️ online = superseded (legacy) value |
| 09 | CANSURF val (aligned GT), EoMT semantic FT, min-area 256 | [`hxuvlvcn`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/hxuvlvcn) | AP50 | 0.1567 | 0.1597 | 0.1567 | ⚠️ online = superseded (legacy) value |
| 09 | CANSURF val (aligned GT), EoMT semantic FT, min-area 256 | [`hxuvlvcn`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/hxuvlvcn) | AP_small | 0.6316 | 0.0045 | 0.6316 | ⚠️ online = superseded (legacy) value |

## 2. Results with no W&B record

| What | Slurm / script | Where it lives locally |
|---|---|---|
| Corrected ablation 01 LaRS PQ 0.298 (prediction-side remap) and the ab03 reproduction 0.3946 | 28011322 | `ablations/01_zeroshot_lars/sources/pq_*_28011322.json` |
| All pycocotools box metrics (RESCORE): the restated mAP / AP_small for ablations 04, 05, 08, 09 | local script rescore_box_predictions.py | `RESCORE.json, RESCORE.md` |
| YOLOv11n FloW test-split result (0.391 Ultralytics, 0.390 pycocotools) | 27942882 (post-training steps) | `ablations/08_specialized_yolo/sources/flow_test_val_27942882.log, yolo11n_flow_test_preds_coco.metrics.json` |
| YOLOv11n CANSURF pycocotools predictions and metrics | 28000242 | `ablations/08_specialized_yolo/sources/yolo11n_cansurf_val_preds_coco.*` |
| VidEoMT per-keyframe AP (0.016 at 4k, 0.009 at 16k): the eval runs have empty W&B summaries | 27775358, 27806956 | `ablations/07_videomt_temporal/sources/videomt_rescore_keyframe_*.json` |
| Qualitative error tables, saved masks and panels | 28001374 | `qualitative/` |

## 3. All benchmark runs on W&B

| Created | Group | Run | State | Slurm | Recorded git commit | Cited result |
|---|---|---|---|---|---|---|
| 2026-10-05T11:18 | ablation01_zeroshot_lars | [`eomt_base640_2x_lars_val_27600047`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/txjoa9ef) | finished | 27600047 | `0f1e1ccbc0` | yes (value compared in §1) |
| 2026-10-05T11:35 | ablation03_finetune_lars_panoptic | [`eomt_lars_panoptic_ft_27607195`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/67kk33hk) | finished | 27607195 | `0f1e1ccbc0` | yes (value compared in §1) |
| 2026-10-06T08:36 | ablation03_finetune_lars_panoptic | [`eomt_lars_panoptic_val_27648289`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/qmi79nsh) | finished | 27648289 | `0f1e1ccbc0` | yes (value compared in §1) |
| 2026-10-06T08:39 | ablation05_eomt_cansurf_zeroshot | [`eomt_cansurf_mask2box_eval_27648392`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/glnvdl1f) | **failed** | 27648392 | `none` | failed attempt (not documented individually) |
| 2026-10-06T08:44 | ablation05_eomt_cansurf_zeroshot | [`eomt_cansurf_mask2box_eval_27648692`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lhnbhuej) | finished | 27648692 | `none` | yes (value compared in §1) |
| 2026-10-06T08:45 | smoke_yolo | [`microrun`](https://wandb.ai/riccardo-campanella/-home-rcampanella-projects-water-debris-seg-benchmark-eval_outputs-water_debris_seg-smoke-yolo/runs/microrun_20261006_104546) | finished | — | `5983e7fd31` | YOLO smoke test |
| 2026-10-06T08:45 | ablation08_yolo_cansurf | [`cansurf_train`](https://wandb.ai/riccardo-campanella/-home-rcampanella-projects-water-debris-seg-benchmark-eval_outputs-water_debris_seg-ablations-08_specialized_yolo/runs/cansurf_train_20261006_104548) | finished | — | `5983e7fd31` | yes (value compared in §1) |
| 2026-10-06T08:49 | smoke_eomt | [`eomt_smoke_lars_27648817`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/g0xb8i9q) | finished | 27648817 | `0f1e1ccbc0` | smoke test |
| 2026-10-06T11:47 | ablation02_finetune_lars_semantic | [`eomt_lars_semantic_ft_27658486`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/mk2w7dr8) | finished | 27658486 | `0f1e1ccbc0` | yes (value compared in §1) |
| 2026-10-06T12:30 | ablation05_eomt_cansurf_larsft | [`eomt_lars_panoptic_cansurf_27660681`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/qi6ymm68) | finished | 27660681 | `5983e7fd31` | yes (value compared in §1) |
| 2026-10-06T12:31 | ablation02_finetune_lars_semantic | [`eomt_lars_semantic_val_27660680`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/duguh1wo) | finished | 27660680 | `0f1e1ccbc0` | yes (value compared in §1) |
| 2026-10-06T12:36 | ablation09_semantic_cansurf | [`eomt_lars_semantic_cansurf_27660924`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/vquavnrm) | finished | 27660924 | `5983e7fd31` | yes (value compared in §1) |
| 2026-10-06T12:36 | ablation03_finetune_lars_panoptic | [`eomt_lars_panoptic_val_27660922`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/xcl7zvm8) | finished | 27660922 | `0f1e1ccbc0` | yes (value compared in §1) |
| 2026-10-06T12:37 | ablation08_yolo_cansurf | [`yolo11n_cansurf_val_wandb_27660923`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/n3vuqqbi) | finished | 27660923 | `5983e7fd31` | yes (value compared in §1) |
| 2026-10-07T11:01 | ablation09_semantic_cansurf | [`eomt_semantic_cansurf_min256_27711596`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/hxuvlvcn) | finished | 27711596 | `5983e7fd31` | yes (value compared in §1) |
| 2026-10-07T11:07 | ablation07_videomt_temporal | [`eomt_seq_flicker_27711595`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/g1bfz3nn) | finished | 27711595 | `5983e7fd31` | yes (value compared in §1) |
| 2026-10-07T13:17 | ablation07_videomt_temporal | [`eomt_panoptic_seq_flicker_27715321`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/5aoppsyb) | finished | 27715321 | `none` | yes (value compared in §1) |
| 2026-10-07T13:36 | ablation07_videomt_temporal | [`videomt_lars_vitb_27716592`](https://wandb.ai/riccardo-campanella/videomt/runs/lars_vitb_27716592) | **failed** | 27716592 | `f351ccbfb2` | failed attempt (not documented individually) |
| 2026-10-07T13:57 | ablation07_videomt_temporal | [`videomt_lars_vitb_27718883`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_27718883) | **failed** | 27718883 | `f351ccbfb2` | failed attempt (not documented individually) |
| 2026-10-08T09:44 | ablation07_videomt_temporal | [`videomt_lars_vitb_27760828`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_27760828) | finished | 27760828 | `f351ccbfb2` | ab07 documented bad run (NUM_CLASSES mismatch, AP 0.000) |
| 2026-10-08T11:22 | ablation04_eomt_flow_larsft | [`eomt_lars_panoptic_flow_27767341`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/0c4pnnn0) | finished | 27767341 | `none` | yes (value compared in §1) |
| 2026-10-08T11:22 | ablation04_eomt_flow_zeroshot | [`eomt_flow_mask2box_27767340`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/u45u5rne) | finished | 27767340 | `none` | yes (value compared in §1) |
| 2026-10-08T11:33 | ablation09_semantic_flow | [`eomt_lars_semantic_flow_27767594`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/hd5bs8yz) | finished | 27767594 | `none` | yes (value compared in §1) |
| 2026-10-08T12:14 | ablation07_videomt_temporal | [`videomt_lars_vitb_27769294`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_27769294) | **crashed** | 27769294 | `f351ccbfb2` | crashed attempt (not documented individually) |
| 2026-10-08T12:23 | ablation07_videomt_temporal | [`videomt_lars_vitb_27769704`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_27769704) | **crashed** | 27769704 | `f351ccbfb2` | crashed attempt (not documented individually) |
| 2026-10-08T12:25 | ablation07_videomt_temporal | [`videomt_lars_vitb_27769861`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_27769861) | finished | 27769861 | `f351ccbfb2` | ab07 documented bad run (no mask supervision, AP 0.000) |
| 2026-10-08T13:32 | ablation07_videomt_temporal | [`videomt_lars_vitb_27772987`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_27772987) | finished | 27772987 | `f351ccbfb2` | ab07 VidEoMT 4k training run behind AP 0.016 / flicker 0.612 |
| 2026-10-08T14:32 | ablation07_videomt_temporal | [`videomt_lars_eval_27775358`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/eval_27775358) | finished | 27775358 | `f351ccbfb2` | ab07 eval of 4k ckpt (AP 0.016); W&B summary empty |
| 2026-10-08T15:04 | ablation07_videomt_temporal | [`videomt_lars_vitb_16k_27776810`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/lars_vitb_16k_27776810) | finished | 27776810 | `f351ccbfb2` | ab07 VidEoMT 16k training run (in-train AP 0.010) |
| 2026-10-08T15:05 | ablation07_videomt_temporal | [`videomt_seq_flicker_27776809`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/wzpe700z) | finished | 27776809 | `5983e7fd31` | yes (value compared in §1) |
| 2026-10-09T11:17 | ablation07_videomt_temporal | [`videomt_lars_eval_27806956`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/eval_27806956) | finished | 27806956 | `f351ccbfb2` | ab07 eval of 16k ckpt (AP 0.009); W&B summary empty |
| 2026-10-09T11:35 | ablation07_videomt_temporal | [`videomt_seq_flicker_27807251`](https://wandb.ai/riccardo-campanella/water-debris-seg/runs/r4ugnnd5) | finished | 27807251 | `none` | yes (value compared in §1) |
| 2026-10-12T13:33 | ablation08_yolo_flow | [`flow_train`](https://wandb.ai/riccardo-campanella/-home-rcampanella-projects-water-debris-seg-benchmark-eval_outputs-water_debris_seg-ablations-08_specialized_yolo/runs/flow_train_20261012_153258) | finished | — | `5983e7fd31` | yes (value compared in §1) |

## 4. Limitations of what W&B can prove

- **Code version:** the commit W&B recorded is the upstream repo's (`0f1e1ccb` = tue-mps/eomt, `f351ccbf` = tue-mps/videomt), this project's scaffold commit `5983e7f` (which contains no code), or none. This project's own code was untracked when the runs executed, so no run pins it. Commit `9bc24c7` contains the code as of 2026-10-09.
- **Project names:** Ultralytics sets the W&B project from its `project=` path, so YOLO runs sit in path-named projects, not `water-debris-seg`.
- **Box-set metrics online are pre-fix:** EoMT mask→box runs logged the legacy scorer's values; corrected values exist only in git.
- **20 of 33 runs** have a value compared in §1; 8 more back a documented result or invalid run without a comparable logged value; 5 are failed or crashed attempts; 0 support no result.
