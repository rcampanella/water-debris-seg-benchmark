# Ablation 06 — WSODD rubbish vs distractors *(out of scope)*

**Status:** **Excluded from benchmark v1** (2026-10-07). WSODD is hosted on [Baidu Pan](https://github.com/sunjiaen/WSODD#dataset_wsodd) only; automated or Snellius-side download is not practical for this project.

**Original question.** On a **14-class harbor** dataset, does a segmenter-as-detector find **rubbish** without confusing **boats, buoys, balls, animals**?

**Replacement in reduced benchmark.** Debris vs obstacles is covered by **LaRS panoptic** (things vs stuff) and **FloW/CANSURF** mask→box mAP. Harbor-specific **rubbish AP with distractors** is not scored.

**If reinstated later** (mirror outside Baidu): per-frame mask→box on rubbish class only; no EoMT/VidEoMT training on WSODD.

## Results

| Weights | rubbish AP | Source |
|---|---|---|
| — | — | *not in scope* |

## Verdict

**Cancelled** — not a benchmark failure; scope decision logged in [`docs/water_debris_seg_decision_log.md`](../../../../docs/water_debris_seg_decision_log.md).

## Code components (dormant)

- `water-debris-seg/scripts/download_wsodd.sh` (documents Baidu link; do not run for v1)
