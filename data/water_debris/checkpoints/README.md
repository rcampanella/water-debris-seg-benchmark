# Fine-tuned EoMT checkpoints (persistent copies)

Copied from scratch on 2026-10-13 so scratch's 14-day purge cannot delete them. SHA256 in [`SHA256SUMS`](SHA256SUMS) matches the scratch originals.

| Folder | Ablation | W&B fit run | Original path (scratch) |
|---|---|---|---|
| `ab02_semantic_mk2w7dr8/epoch=30-step=2356.ckpt` | 02 LaRS semantic FT (mIoU 98.0) | `mk2w7dr8` | `/scratch-shared/rcampanella/water-debris-seg/artifacts/wandb/water-debris-seg/mk2w7dr8/checkpoints/` |
| `ab03_panoptic_67kk33hk/epoch=11-step=7812.ckpt` | 03 LaRS panoptic FT (PQ 0.395) | `67kk33hk` | `/scratch-shared/rcampanella/water-debris-seg/artifacts/wandb/water-debris-seg/67kk33hk/checkpoints/` |

Pretrained weights used by the fairness fixes (added to `SHA256SUMS` on 2026-10-27):

| File | Used by | Source |
|---|---|---|
| `eomt_coco/coco_panoptic_eomt_base_640_2x.bin` | starting weights of every EoMT fine-tuning run in `fairness_fixes/` (and EoMT zero-shot) | EoMT COCO-panoptic ViT-B 640 release |
| `sam/sam2.1_l.pt` | SAM pseudo-masks (fix 3), YOLOv11s + SAM and EoMT + SAM (fixes 1–2) | Ultralytics SAM 2.1 large weights |
| `yolo/yolo11s.pt` | YOLOv11s COCO zero-shot on CANSURF (paper protocol, 2026-10-27) | Ultralytics COCO weights, `ultralytics/assets` release v8.3.0 |

Fine-tuned checkpoints from `fairness_fixes/` stay in their run folders; their SHA-256 is recorded in each scored JSON in `fairness_fixes/sources/`.

`*.ckpt`, `*.pt` and `*.bin` files are git-ignored.
