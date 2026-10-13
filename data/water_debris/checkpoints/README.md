# Fine-tuned EoMT checkpoints (persistent copies)

Copied from scratch on 2026-10-13 so scratch's 14-day purge cannot delete them. SHA256 in [`SHA256SUMS`](SHA256SUMS) matches the scratch originals.

| Folder | Ablation | W&B fit run | Original path (scratch) |
|---|---|---|---|
| `ab02_semantic_mk2w7dr8/epoch=30-step=2356.ckpt` | 02 LaRS semantic FT (mIoU 98.0) | `mk2w7dr8` | `/scratch-shared/rcampanella/water-debris-seg/artifacts/wandb/water-debris-seg/mk2w7dr8/checkpoints/` |
| `ab03_panoptic_67kk33hk/epoch=11-step=7812.ckpt` | 03 LaRS panoptic FT (PQ 0.395) | `67kk33hk` | `/scratch-shared/rcampanella/water-debris-seg/artifacts/wandb/water-debris-seg/67kk33hk/checkpoints/` |

`*.ckpt` files are git-ignored.
