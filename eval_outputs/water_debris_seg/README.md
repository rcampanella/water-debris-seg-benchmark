# Water-surface debris segmentation benchmark outputs

Start here: **[REPORT.md](REPORT.md)** — benchmark motivation, **metrics glossary**, per-ablation **inputs/outputs/relevance**, the main comparison table, and status.

Fair-comparison plan (2026-10-23): **[fairness_fixes/README.md](fairness_fixes/README.md)**.

Spec: [`docs/water_debris_seg_benchmark.md`](../../docs/water_debris_seg_benchmark.md)  
Swarm: [`docs/water_debris_seg_swarm.md`](../../docs/water_debris_seg_swarm.md)

- `ablations/` — one experiment per folder (OpenFly-style README + `sources/`); [01](ablations/01_zeroshot_lars/README.md) is the most detailed template for metrics and motivation.
- `GPU_INVENTORY.md` / `data/water_debris/DISK_BUDGET.md` — gates before GPU/download stages
- `metrics.json` / `MANIFEST.json` — machine-readable index (filled after runs)
