#!/usr/bin/env python3
"""Run Ultralytics val on an existing detector and log metrics to W&B."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--group", default="ablation08_yolo_cansurf")
    parser.add_argument("--name", default=None)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "water-debris-seg"))
    args = parser.parse_args()

    import wandb
    from ultralytics import YOLO

    run_name = args.name or f"yolo_val_{os.environ.get('SLURM_JOB_ID', 'local')}"
    wandb.init(project=args.project, group=args.group, name=run_name, job_type="val")

    model = YOLO(args.model)
    results = model.val(data=args.data, imgsz=640, device=0, verbose=True)
    box = results.box
    metrics = {
        "metrics/mAP50-95": float(box.map),
        "metrics/mAP50": float(box.map50),
        "metrics/precision": float(box.mp),
        "metrics/recall": float(box.mr),
    }
    if hasattr(box, "maps") and box.maps is not None:
        metrics["metrics/mAP50-95_per_class"] = [float(x) for x in box.maps]
    wandb.log(metrics)
    wandb.finish()
    print(metrics)


if __name__ == "__main__":
    main()
