# Water-surface debris segmentation benchmark

Benchmark and agent execution plan for **nearshore USV camera** perception: segment and detect debris on the water surface using **EoMT**, **VidEoMT**, and **YOLOv11** baselines on **LaRS**, **FloW**, **CANSURF**, and **WSODD**.

## Start here

| Doc | Purpose |
|---|---|
| [`docs/water_debris_seg_benchmark.md`](docs/water_debris_seg_benchmark.md) | Spec, hard rule (LaRS-only mask training), metrics |
| [`docs/water_debris_seg_swarm.md`](docs/water_debris_seg_swarm.md) | Agent waves, timeline, Slurm/GPU gates |
| [`docs/water_debris_seg_decision_log.md`](docs/water_debris_seg_decision_log.md) | Append-only benchmark decisions |
| [`eval_outputs/water_debris_seg/REPORT.md`](eval_outputs/water_debris_seg/REPORT.md) | Experiment catalog and results (scaffold) |

## Hard rule (summary)

Train **EoMT / VidEoMT** masks **only on LaRS**. On **FloW, CANSURF, WSODD**: **eval only** via masks → boxes → COCO mAP. **YOLO** may train on box datasets.

## Layout

```
docs/                          # benchmark + swarm
eval_outputs/water_debris_seg/ # ablations 01–09, REPORT, gates
data/water_debris/             # dataset mount points + DISK_BUDGET.md
```

Implementation code (adapters, eval scripts) will live under `water-debris-seg/` in a later stage.

## Status

Stage 0 complete (docs + scaffold). Stage 1+ blocked on disk quota and GPU node probe — see `data/water_debris/DISK_BUDGET.md` and `eval_outputs/water_debris_seg/GPU_INVENTORY.md`.
