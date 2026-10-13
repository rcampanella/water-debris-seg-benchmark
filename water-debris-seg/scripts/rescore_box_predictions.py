#!/usr/bin/env python3
"""Re-score every run in box_eval_registry.json with the pycocotools scorer.

For each run:
  - a pre-existing metrics file is preserved once as <name>.legacy.json (never overwritten)
  - the new pycocotools metrics are written to the registry's `metrics` path
  - an audit row (sha256 of pred + GT, legacy vs new metrics) goes to the summary

Outputs (under --eval-root):
  RESCORE.json   machine-readable audit table
  RESCORE.md     human-readable table, linked from REPORT.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

from eval_coco_bbox import evaluate_bbox_map
from eval_coco_bbox_legacy import evaluate_bbox_map as evaluate_bbox_map_legacy


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def legacy_path(metrics: Path) -> Path:
    return metrics.with_name(metrics.name.removesuffix(".json") + ".legacy.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = Path(__file__).resolve().parents[2] / "eval_outputs" / "water_debris_seg"
    parser.add_argument("--eval-root", type=Path, default=default_root)
    args = parser.parse_args()
    root = args.eval_root
    registry = json.loads((root / "box_eval_registry.json").read_text())

    rows = []
    for run in registry["runs"]:
        gt = root / registry["ground_truth"][run["dataset"]]["path"]
        pred = root / run["pred"]
        metrics_path = root / run["metrics"]
        legacy = legacy_path(metrics_path)

        if metrics_path.is_file() and not legacy.is_file():
            old = json.loads(metrics_path.read_text())
            if old.get("scorer") != "pycocotools":
                shutil.copy2(metrics_path, legacy)
        if legacy.is_file():
            legacy_metrics = json.loads(legacy.read_text())
            legacy_source = str(legacy.relative_to(root))
        else:  # no stored legacy file for this pairing: recompute with the legacy scorer
            legacy_metrics = evaluate_bbox_map_legacy(gt, pred)
            legacy_source = "recomputed with eval_coco_bbox_legacy.py"

        new = evaluate_bbox_map(gt, pred)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(new, indent=2))

        rows.append({
            **{k: run[k] for k in ("id", "ablation", "dataset", "method", "headline")},
            "pred": run["pred"], "pred_sha256": sha256(pred),
            "gt": registry["ground_truth"][run["dataset"]]["path"], "gt_sha256": sha256(gt),
            "metrics": run["metrics"], "legacy_metrics_source": legacy_source,
            "legacy": {k: legacy_metrics.get(k) for k in ("mAP", "AP50", "AP75", "AP_small")},
            "new": new,
        })
        print(f"{run['id']:32} legacy mAP {legacy_metrics['mAP']:.3f} -> {new['mAP']:.3f}  "
              f"AP_small {legacy_metrics['AP_small']:.3f} -> {new['AP_small']:.3f}")

    stamp = dt.datetime.now().isoformat(timespec="seconds")
    (root / "RESCORE.json").write_text(json.dumps({"generated": stamp, "scorer": "pycocotools", "runs": rows}, indent=2))

    def f(x):
        return "—" if x is None else ("n/a" if x < 0 else f"{x:.3f}")

    lines = [
        "# Box-metric re-score (pycocotools)",
        "",
        f"Generated {stamp} by `water-debris-seg/scripts/rescore_box_predictions.py` from "
        "[`box_eval_registry.json`](box_eval_registry.json). Machine-readable: [`RESCORE.json`](RESCORE.json) "
        "(includes sha256 of every prediction and GT file).",
        "",
        "**Why:** the pre-2026-10-13 scorer (`eval_coco_bbox_legacy.py`) averaged only AP@.50/.75, never "
        "counted false positives in AP_small, and skipped images without predictions. See "
        "[decision log](../../docs/water_debris_seg_decision_log.md) 2026-10-13. Legacy metrics files are kept "
        "beside the new ones as `*.legacy.json`.",
        "",
        "| Run | Dataset | Method | mAP legacy → **new** | AP50 | AP75 | AP_small legacy → **new** | AR100 | Images with preds | Preds | Metrics file |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        n, lg = r["new"], r["legacy"]
        lines.append(
            f"| {r['id']}{' ★' if r['headline'] else ''} | {r['dataset']} | {r['method']} | "
            f"{f(lg['mAP'])} → **{f(n['mAP'])}** | {f(n['AP50'])} | {f(n['AP75'])} | "
            f"{f(lg['AP_small'])} → **{f(n['AP_small'])}** | {f(n['AR100'])} | "
            f"{n['n_images_with_pred']}/{n['n_gt_images']} | {n['n_pred']} | [{Path(r['metrics']).name}]({r['metrics']}) |"
        )
    lines += ["", "★ = headline row quoted in REPORT.md. mAP = AP@[.50:.95]; AP_small = AP@[.50:.95] on GT area < 32²."]
    (root / "RESCORE.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {root / 'RESCORE.md'}")


if __name__ == "__main__":
    main()
