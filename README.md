# Water-surface debris segmentation benchmark — code

Code for a nearshore USV camera perception benchmark:

- **Segmentation and detection** of debris on the water surface with **EoMT**, **VidEoMT**, **YOLOv11** and
  **YOLO26** / **YOLO26-seg** on **LaRS**, **FloW** and **CANSURF**.
- **Monocular depth** with **YOLO26-depth** on **Pohang Canal** and **MULTIAQUA**, scored against LiDAR.

The repository contains code only (adapters, training, scoring, latency and plotting scripts, Slurm wrappers). It
contains no datasets, checkpoints or results.

## Layout

```
water-debris-seg/
  water_debris_seg/          # package: dataset loaders, EoMT adapters, LoRA, YOLO focal loss, taxonomy, VidEoMT patches
  scripts/                   # data download/verification, training, scoring, latency, tables and plots
  scripts/depth/             # depth benchmark: GT build, splits, training, scoring, latency, tables, plots
  scripts/snellius/          # Slurm job scripts (Snellius, SURF)
  configs/                   # EoMT / VidEoMT training configs, COCO→LaRS class maps
  envs/                      # pinned package lists of the virtual environments and third-party commits
  tests/                     # unit tests
data/water_debris/
  sources/                   # dataset checksums and third-party pins
  checkpoints/SHA256SUMS     # checksums of the model checkpoints used
```

Setup, paths and quick commands: [`water-debris-seg/README.md`](water-debris-seg/README.md).

## Requirements

Python ≥ 3.11 (`water-debris-seg/pyproject.toml`). Third-party models (EoMT, VidEoMT, Ultralytics) are cloned
separately at the commits in `data/water_debris/sources/third_party_pins.json` and
`water-debris-seg/envs/third_party_commits.txt`. Datasets are downloaded by the scripts in
`water-debris-seg/scripts/` and checked against the checksums in `data/water_debris/sources/`.
