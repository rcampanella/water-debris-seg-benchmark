# Water-surface debris segmentation benchmark

Benchmark and execution plan for **nearshore USV camera** perception: segment and detect debris on the water surface using **EoMT**, **VidEoMT**, and **YOLOv11s** on **LaRS**, **FloW**, and **CANSURF** (WSODD excluded v1 — see decision log). **2026-10-23:** comparison redesigned so both model families are trained and scored on the same data: [`eval_outputs/water_debris_seg/fairness_fixes/README.md`](eval_outputs/water_debris_seg/fairness_fixes/README.md).

## Start here

| Doc | Purpose |
|---|---|
| **[`eval_outputs/water_debris_seg/REPORT.md`](eval_outputs/water_debris_seg/REPORT.md)** | **Main report** — metrics, all ablations (motivation, inputs/outputs, relevance), main comparison table |
| [`eval_outputs/water_debris_seg/fairness_fixes/README.md`](eval_outputs/water_debris_seg/fairness_fixes/README.md) | Fair-comparison plan: decisions, main comparison table (decisions enforced), temporal table, fixes 1–3 |
| [`docs/water_debris_seg_benchmark.md`](docs/water_debris_seg_benchmark.md) | Full spec, training rule (revised 2026-10-23), citations |
| [`docs/water_debris_seg_swarm.md`](docs/water_debris_seg_swarm.md) | Agent waves, timeline, Slurm/GPU gates |
| [`docs/water_debris_seg_decision_log.md`](docs/water_debris_seg_decision_log.md) | Append-only benchmark decisions |
| [`eval_outputs/water_debris_seg/ablations/01_zeroshot_lars/README.md`](eval_outputs/water_debris_seg/ablations/01_zeroshot_lars/README.md) | Deep dive: PQ, models, datasets, zero-shot protocol |

## What we are measuring (summary)

- **LaRS:** **PQ**, **mIoU** (obstacle / water / sky) and **box AP**, each in its own column, for EoMT panoptic and YOLOv11s (+ SAM for its masks). VidEoMT in a separate temporal table (per-keyframe mask AP, flicker; [07](eval_outputs/water_debris_seg/ablations/07_videomt_temporal/README.md)).
- **FloW / CANSURF (boxes only):** **box AP**, debris classes only (class-agnostic as reference). YOLOv11s trains on the native boxes; EoMT on box-prompted SAM masks; EoMT boxes = min/max of its masks.
- **YOLO:** YOLOv11s (the CANSURF paper's model); YOLOv11n results kept as references ([08](eval_outputs/water_debris_seg/ablations/08_specialized_yolo/README.md), [10](eval_outputs/water_debris_seg/ablations/10_yolo_lars_boxes/README.md)).

Ten ablations ([01](eval_outputs/water_debris_seg/ablations/01_zeroshot_lars/README.md)–[10](eval_outputs/water_debris_seg/ablations/10_yolo_lars_boxes/README.md)) plus the fairness fixes form an **eval ladder**: zero-shot floor → LaRS fine-tune → cross-domain detection → specialized YOLO → temporal video. Catalog with I/O and relevance: [REPORT.md § Experiment catalog](eval_outputs/water_debris_seg/REPORT.md#experiment-catalog-motivation-io-relevance).

## Training rule (summary, revised 2026-10-23)

Both model families are trained on each dataset they are compared on, with the same held-out protocol: **EoMT** on LaRS masks and on box-prompted **SAM masks** for FloW / CANSURF; **YOLOv11s** on LaRS traced boxes and on the native FloW / CANSURF boxes. VidEoMT trains on LaRS sequences only. Superseded rule: EoMT/VidEoMT trained only on LaRS (2026-10-05 to 2026-10-23).

## Layout

```
docs/                          # benchmark + swarm
eval_outputs/water_debris_seg/ # REPORT, ablations 01–10, fairness_fixes/, gates
water-debris-seg/              # adapters, configs, Slurm, eval scripts
data/water_debris/             # dataset mount points + DISK_BUDGET.md
```

## Status (2026-10-23)

**Cluster:** Snellius (SURF).

| Area | Status |
|---|---|
| Ablations 01–05, 07–10 | Done; FloW / CANSURF EoMT values superseded (constant confidence) — see [REPORT.md](eval_outputs/water_debris_seg/REPORT.md) |
| Fairness fixes 1, 1b, 2, 3 | Planned; main comparison table mostly pending — [fairness_fixes](eval_outputs/water_debris_seg/fairness_fixes/README.md) |
| Ablation 06 / WSODD | **Out of scope v1** |
| W&B | `water-debris-seg` — [`water-debris-seg/docs/WANDB.md`](water-debris-seg/docs/WANDB.md) |

Storage: datasets on **scratch-shared** (14-day purge); repo and reports on **home** (~200 GiB quota) — [SURF filesystem docs](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/85295828/Snellius+filesystems).
