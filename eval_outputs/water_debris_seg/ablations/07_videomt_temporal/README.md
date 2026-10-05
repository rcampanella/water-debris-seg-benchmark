# Ablation 07 — VidEoMT temporal on LaRS sequences

**Question.** Does **online** VidEoMT beat per-frame EoMT on LaRS keyframes for PQ/mIoU and **temporal stability** (mask flicker, thing ID switches)?

**Expected outcome.** Similar or slightly better PQ vs EoMT still; clear win on ID stability for boats/buoys over 9-frame context.

**Hard rule.** VidEoMT **training** (if any) uses **LaRS sequences only**. Box sets: eval-only mask→box on stills (optional side table).

## Setup

| Item | Value |
|---|---|
| Data | LaRS sequence images + panoptic keyframes |
| Compare | EoMT still vs VidEoMT online |
| Metrics | PQ, flicker score, ID switch rate |
| Speed | FPS via `benchmark.py` |

## Results

| Method | PQ | Flicker | ID switches | FPS | Source |
|---|---|---|---|---|---|
| EoMT per-frame | — | — | — | — | *pending* |
| VidEoMT online | — | — | — | — | *pending* |

## Verdict

*Pending.* Requires LaRS sequences (~13.6 GB).

## Resources copied here

| Original | Copy in this folder | Present |
|---|---|---|
| — | sources/ | — |

## Code components

- [VidEoMT](https://github.com/tue-mps/videomt) `train_net_video.py`
