# Dataset mount points

On Snellius, **binary data lives on scratch**, not in git.

| Dataset | Scratch path | Repo stub |
|---|---|---|
| LaRS | `/scratch-shared/$USER/water-debris-seg/data/lars` | `lars/` (`.gitkeep` only) |
| CANSURF | `.../data/cansurf` | `cansurf/` |
| FloW | `.../data/flow` | `flow/` |
| ~~WSODD~~ | *(out of scope v1)* | `wsodd/` stub unused |

Optional symlink from repo (keeps relative paths in docs):

```bash
ln -sfn "/scratch-shared/${USER}/water-debris-seg/data/lars" \
  "/gpfs/home1/${USER}/projects/water-debris-seg-benchmark/data/water_debris/lars"
```

Download: `bash water-debris-seg/scripts/download_lars.sh --stills-only`
