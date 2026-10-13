#!/usr/bin/env python3
"""COCO bbox JSON → standard COCO detection metrics via pycocotools.

Headline keys keep their pre-2026-10-13 names, now with standard meaning:
  mAP       AP@[.50:.95]             (COCOeval stats[0])
  AP50/AP75 AP@.50 / AP@.75           (stats[1], stats[2])
  AP_small  AP@[.50:.95], area < 32²  (stats[3])
Every image in the GT counts, including images a method left empty. Predictions on
image ids absent from the GT are dropped and counted in `n_pred_dropped`.

The previous hand-rolled scorer is kept as eval_coco_bbox_legacy.py for auditing.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STAT_KEYS = (
    "mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
)


def run_cocoeval(gt_path: Path, pred_path: Path | list[dict]) -> tuple[COCOeval | None, dict]:
    """Run COCOeval; return the evaluator (None if no usable predictions) and run counts.

    `pred_path` may also be an already-loaded prediction list. The evaluator's evalImgs keep
    per-detection matches, which error-analysis code reuses.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_path))
    if isinstance(pred_path, list):
        preds = pred_path
    else:
        preds = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    gt_ids = set(coco_gt.getImgIds())
    kept = [p for p in preds if p["image_id"] in gt_ids]
    counts = {
        "n_gt_images": len(gt_ids),
        "n_gt_boxes": len(coco_gt.getAnnIds()),
        "n_pred": len(kept),
        "n_pred_dropped": len(preds) - len(kept),
        "n_images_with_pred": len({p["image_id"] for p in kept}),
    }
    if not kept:
        return None, counts
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(kept)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev, counts


def evaluate_bbox_map(gt_path: Path, pred_path: Path) -> dict[str, float]:
    ev, counts = run_cocoeval(gt_path, pred_path)
    stats = [0.0] * len(STAT_KEYS) if ev is None else [float(s) for s in ev.stats]
    # COCOeval reports -1 when an area range has no GT; keep that sentinel visible.
    metrics: dict = dict(zip(STAT_KEYS, stats))
    metrics.update(counts)
    metrics["scorer"] = "pycocotools"
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt-coco", type=Path, required=True)
    parser.add_argument("--pred-coco", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-group", default="eval_coco_bbox")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    metrics = evaluate_bbox_map(args.gt_coco, args.pred_coco)
    print(json.dumps(metrics, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.wandb:
        import wandb

        wandb.init(
            project="water-debris-seg",
            group=args.wandb_group,
            name=args.wandb_name,
            config={"gt": str(args.gt_coco), "pred": str(args.pred_coco), "scorer": "pycocotools"},
        )
        wandb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        wandb.finish()


if __name__ == "__main__":
    main()
