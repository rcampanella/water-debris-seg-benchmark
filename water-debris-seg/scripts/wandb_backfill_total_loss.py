#!/usr/bin/env python3
"""Add train/total_loss and val/total_loss to finished ablation-10 YOLO runs on W&B (from their results.csv).

Runs trained before train_yolo_lars.py logged the total (it now does, via add_total_loss) only have the three
terms. W&B cannot write to past steps of a finished run, so the totals are appended with their own x-axis
`epoch` (define_metric), equal to the original W&B step (Ultralytics logs epoch e at step e). Runs that log the
total natively also log `epoch`, so one panel with x-axis `epoch` compares all of them.

  total = box_loss + cls_loss + dfl_loss   (each already multiplied by its gain: box 7.5, cls 0.5, dfl 1.5)

Usage: python wandb_backfill_total_loss.py RUN_ID=path/to/results.csv [...]   (source wandb_env.sh first)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

TERMS = ("box_loss", "cls_loss", "dfl_loss")


def totals(results_csv: Path) -> list[dict]:
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(results_csv.open())]
    return [{"epoch": int(float(r["epoch"])),
             "train/total_loss": round(sum(float(r[f"train/{t}"]) for t in TERMS), 5),
             "val/total_loss": round(sum(float(r[f"val/{t}"]) for t in TERMS), 5)} for r in rows]


def main() -> None:
    import os

    import wandb

    pairs = [a.split("=", 1) for a in sys.argv[1:]]
    if not pairs or any(len(p) != 2 for p in pairs):
        sys.exit(__doc__)
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY") or api.default_entity
    project = os.environ.get("WANDB_PROJECT", "water-debris-seg")
    for run_id, path in pairs:
        rows = totals(Path(path))
        existing = api.run(f"{entity}/{project}/{run_id}")
        if existing.state == "running":
            sys.exit(f"{run_id} is still running; backfill after it finishes")
        if "val/total_loss" in existing.summary:
            print(f"{run_id}: already has val/total_loss, skipped")
            continue
        run = wandb.init(project=project, entity=entity, id=run_id, resume="must")
        run.define_metric("epoch")
        for k in ("train/total_loss", "val/total_loss"):
            run.define_metric(k, step_metric="epoch")
        for r in rows:
            run.log(r)
        run.summary["total_loss_source"] = f"backfilled from {path} (sum of box, cls, dfl terms)"
        run.finish()
        print(f"{run_id}: {len(rows)} epochs, last train/val total {rows[-1]['train/total_loss']} / {rows[-1]['val/total_loss']}")


if __name__ == "__main__":
    main()
