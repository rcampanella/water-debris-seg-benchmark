# Disk budget — water debris benchmark

**Host:** Snellius (`int6.local.snellius.surf.nl`, login 2026-10-05).  
Update **before every stage**. Require: `avail ≥ stage_increment × 1.2` on the filesystem tier used for that stage.

Official filesystem reference: [Snellius filesystems (SURF)](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/85295828/Snellius+filesystems).

---

## Filesystem map (this cluster)

| Tier | Mount | Retention / permanence | Backup | Benchmark use |
|---|---|---|---|---|
| **Home** | `/home/$USER` → `/gpfs/home1/$USER` | **Persistent** while login is tied to an active project; removed ~15 weeks after project/login expiry ([FAQ](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/30660265/Snellius+FAQ)) | Nightly incremental | Git repo, docs, manifests, `eval_outputs/` summaries, **last** checkpoints only |
| **Scratch-shared** | `/scratch-shared/` | **Not permanent** — files unchanged **14 days** are auto-deleted (no warning) | None | Primary **datasets**, conda/env caches, train/eval **I/O**, intermediate checkpoints |
| **Scratch-local** | `/scratch-local/` (often `$TMPDIR`) | **Not permanent** — **6-day** inactivity purge; node-local view | None | Per-job temp; copy results off before job ends |
| **Scratch-node** | `/scratch-node/` (`#SBATCH --constraint=scratch-node`) | **Deleted when the job ends** | None | Burst I/O inside long jobs only |
| **Project** | `/projects/<name>/` | Project duration; not backed up | None | Use if allocated; treat like durable project disk, still not backup |
| **Archive** | `/archive/` | Project / service terms | Nightly | Long-term cold storage (not for active training) |

**Do not** store the canonical dataset tree only on scratch without a home/project copy of manifests and a plan to refresh or archive before the 14-day window.

---

## Canonical paths (Snellius)

| Role | Path |
|---|---|
| Repository (this benchmark) | `/gpfs/home1/rcampanella/projects/water-debris-seg-benchmark` |
| Heavy data root (create on Stage 1) | `/scratch-shared/rcampanella/water-debris-seg/` |
| Suggested layout under scratch | `data/{lars,cansurf,flow,wsodd}/`, `third_party/`, `artifacts/` |
| Env variable (future code) | `WATER_DEBRIS_DATA_ROOT=/scratch-shared/rcampanella/water-debris-seg/data` |

Symlink from `data/water_debris/<dataset>/` → scratch paths once downloads start (keeps repo paths stable).

---

## Current snapshot (Stage 0 → 1 re-gate)

| When | Location | Quota / size | Used (approx) | Avail for plan | Notes |
|---|---|---|---|---|---|
| 2026-10-05 | Home `/gpfs/home1/rcampanella` | **200 GiB** default ([SURF](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/85295828/Snellius+filesystems)) | **~5 GiB** (`du -sh`) | **~195 GiB** under default quota | Permanent + backed up; OK for repo + gates; avoid heavy parallel I/O here |
| 2026-10-05 | Scratch-shared `/scratch-shared` | **8 TiB** per user (shared counter with scratch-local) | not yet used | **≫ full benchmark (~80–120 GiB)** | Ephemeral — datasets + training live here |
| 2026-10-05 | GPFS pool `home1` | 121 TiB filesystem | 62% used cluster-wide | 47 TiB pool free | Not user quota; informational only |

Run `quota -s` before each gate (often **~2 min**, can print **no lines** on this login); if empty, use `du -sh ~` and assume **200 GiB** home / **8 TiB** scratch unless your project says otherwise.

**Legacy note:** Stage 0 on a previous host used `vast-storage.umbrella:/umbrella` at `/vast.mnt` with **19 GiB** free — that gate **does not apply** on Snellius.

---

## Planned increments (from spec)

| Stage | Add (conservative) | Preferred tier | Cumulative note |
|---|---|---|---|
| 0 docs | ~1 MB | Home | OK |
| 1 code + weights | 8–15 GB | Scratch (+ pin small manifests in repo) | Needs scratch headroom |
| 1 LaRS stills+ann | +1 GB | Scratch | Required first |
| 1 LaRS sequences | +14 GB | Scratch | Defer until VidEoMT |
| 1 CANSURF raw | +3–8 GB | Scratch | |
| 1 FloW-Img | +5–15 GB | Scratch | |
| 4–6 artifacts | +20–40 GB | Scratch during run; **last ckpt → home** | Prune intermediates on scratch |

WSODD (~10–25 GB) **not planned** — excluded from benchmark v1 (decision log 2026-10-07).

Full working set ~80–120 GB fits comfortably in scratch quota; home holds repo + exported results.

---

## Gate log

| Stage | Date | Avail at gate | Needed | Decision | Notes |
|---|---|---|---|---|---|
| 0 | 2026-10-05 (umbrella host) | 19G | 1 MB | GO | Scaffold only; superseded on Snellius |
| 0 | 2026-10-05 (Snellius) | ~195G home; scratch unused | 1 MB | **GO** | Docs + repo on home |
| 1 | 2026-10-05 (Snellius) | scratch ≫120G; home ≫25G | ≥25G min on **scratch** | **GO** | Create `/scratch-shared/rcampanella/water-debris-seg/`; GPU gate separate |
| 1 LaRS stills | 2026-10-05 | ~1G on scratch | ~1G | **DONE** | `train/`, `val/`, `test/` + zips; sha256 in `data/water_debris/sources/lars_v1.0.0_sha256.txt` |
| 1 LaRS verify | 2026-10-05 | — | panoptic train/val | **DONE** | `verify_lars.py` — test split images-only (no public panoptic GT) |
| 1 CANSURF | 2026-10-05 | ~2.1G on scratch | ~2.1G | **DONE** | Zenodo v2; `train/`+`val/` YOLO; sha256 in `sources/cansurf_v2_sha256.txt` |
| 1 full | — | — | ~120G scratch + sync policy | PENDING | FloW optional; WSODD out of scope |

---

## Stage 1 checklist (Agent 1)

1. `mkdir -p /scratch-shared/rcampanella/water-debris-seg/{data,artifacts,third_party}`
2. `df -h /home/$USER /scratch-shared` and `quota -s` → paste into gate log
3. Downloads only under scratch; sha256 manifests under `data/water_debris/*/sources/` in repo (home)
4. After each Slurm train job: rsync `last.ckpt` / metrics to `eval_outputs/` on home; delete old scratch checkpoints
