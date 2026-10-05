# Disk budget — water debris benchmark

Update **before every stage**. Require: `avail ≥ stage_increment × 1.2`.

## Current host snapshot (Stage 0)

| When | Filesystem | Size | Used | Avail | Use% | Mount |
|---|---|---|---|---|---|---|
| 2026-10-05 | vast-storage.umbrella:/umbrella | 200G | 182G | **19G** | 91% | /vast.mnt |

**Stage 0 GO** for docs/scaffold only. **Stage 1 NO-GO** for full benchmark (~80–120 GB working set) until additional storage or cleanup — 19G fits minimum path (~15–25 GB) only with tight pruning and no WSODD/FloW videos.

## Planned increments (from spec)

| Stage | Add (conservative) | Cumulative note |
|---|---|---|
| 0 docs | ~1 MB | OK |
| 1 code + weights | 8–15 GB | Needs ~10–18 GB avail |
| 1 LaRS stills+ann | +1 GB | |
| 1 LaRS sequences | +14 GB | Defer |
| 1 CANSURF raw | +3–8 GB | |
| 1 FloW-Img | +5–15 GB | |
| 1 WSODD | +10–25 GB | Last |
| 4–6 artifacts | +20–40 GB | Keep last ckpt |

## Gate log

| Stage | Date | Avail at gate | Needed | Decision | Notes |
|---|---|---|---|---|---|
| 0 | 2026-10-05 | 19G | 1 MB | GO | Scaffold only |
| 1 | — | — | ≥25G min path | PENDING | Free space before downloads |
