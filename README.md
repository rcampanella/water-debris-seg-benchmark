# Water-surface debris segmentation benchmark

Benchmark and execution plan for **nearshore USV camera** perception: segment and detect debris on the water surface using **EoMT**, **VidEoMT**, and **YOLOv11** on **LaRS**, **FloW**, and **CANSURF** (WSODD excluded v1 — see decision log).

## Start here

| Doc | Purpose |
|---|---|
| **[`eval_outputs/water_debris_seg/REPORT.md`](eval_outputs/water_debris_seg/REPORT.md)** | **Main report** — metrics, all ablations (motivation, inputs/outputs, relevance), headline results |
| [`docs/water_debris_seg_benchmark.md`](docs/water_debris_seg_benchmark.md) | Full spec, hard rule (LaRS-only mask training), citations |
| [`docs/water_debris_seg_swarm.md`](docs/water_debris_seg_swarm.md) | Agent waves, timeline, Slurm/GPU gates |
| [`docs/water_debris_seg_decision_log.md`](docs/water_debris_seg_decision_log.md) | Append-only benchmark decisions |
| [`eval_outputs/water_debris_seg/ablations/01_zeroshot_lars/README.md`](eval_outputs/water_debris_seg/ablations/01_zeroshot_lars/README.md) | Deep dive: PQ, models, datasets, zero-shot protocol |

## What we are measuring (summary)

- **LaRS (masks):** semantic **mIoU** (obstacle/water/sky); panoptic **PQ / SQ / RQ** (things vs stuff); video **flicker** and VidEoMT metrics ([07](eval_outputs/water_debris_seg/ablations/07_videomt_temporal/README.md)).
- **FloW / CANSURF (boxes only):** EoMT/VidEoMT **eval-only** — **mask→box → mAP and AP_S** (small objects &lt; 32²). No mask training on these sets.
- **YOLO:** train + eval on box datasets — **in-domain detection ceiling** ([08](eval_outputs/water_debris_seg/ablations/08_specialized_yolo/README.md)).

Nine ablations ([01](eval_outputs/water_debris_seg/ablations/01_zeroshot_lars/README.md)–[09](eval_outputs/water_debris_seg/ablations/09_cross_dataset_transfer/README.md)) form an **eval ladder**: zero-shot floor → LaRS fine-tune → cross-domain detection → specialized YOLO → temporal video. Catalog with I/O and relevance: [REPORT.md § Experiment catalog](eval_outputs/water_debris_seg/REPORT.md#experiment-catalog-motivation-io-relevance).

## Hard rule (summary)

Train **EoMT / VidEoMT** masks **only on LaRS**. On **FloW** and **CANSURF**: **eval only** via masks → boxes → COCO mAP. **YOLO** may train on box datasets.

## Layout

```
docs/                          # benchmark + swarm
eval_outputs/water_debris_seg/ # REPORT, ablations 01–09, gates
water-debris-seg/              # adapters, configs, Slurm, eval scripts
data/water_debris/             # dataset mount points + DISK_BUDGET.md
```

## Status (2026-10-08)

**Cluster:** Snellius (SURF).

| Area | Status |
|---|---|
| Ablations 01–05, 09 | LaRS + CANSURF + FloW mask→box complete — see [REPORT.md](eval_outputs/water_debris_seg/REPORT.md) |
| Ablation 08 | CANSURF YOLO done; FloW YOLO pending |
| Ablation 07 | VidEoMT train done; VIS eval needs category fix + retrain |
| Ablation 06 / WSODD | **Out of scope v1** |
| W&B | `water-debris-seg` — [`water-debris-seg/docs/WANDB.md`](water-debris-seg/docs/WANDB.md) |

Storage: datasets on **scratch-shared** (14-day purge); repo and reports on **home** (~200 GiB quota) — [SURF filesystem docs](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/85295828/Snellius+filesystems).
