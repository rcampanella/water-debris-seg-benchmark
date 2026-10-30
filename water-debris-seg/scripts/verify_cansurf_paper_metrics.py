#!/usr/bin/env python3
"""Check cansurf_paper_metrics.py against the CANSURF paper's own script on the same YOLO weights and images.

Runs `benchmark_yolo` from the pinned yolov11_benchmark.py (ZaidAljundiHW2/CANSURF @ 449ede6, unmodified; seaborn is
stubbed because it only draws the confusion matrix after the metrics) on a symlinked copy of the 2,209 scored val
images, and compares its TP / FP / FN / P / R / F1 / mean IoU with our JSON for the same weights. Writes
fairness_fixes/sources/cansurf_paper_check_<method>_<job>.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

from cansurf_paper_metrics import GT_JSON, PAPER_SCRIPT, REPO, SRC, VAL


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--ours", type=Path, required=True, help="cansurf_paper_<method>_<job>.json for the same weights")
    ap.add_argument("--variant", default="class_agnostic", help="our variant matching the paper's class 0 row")
    args = ap.parse_args()

    sys.modules.setdefault("seaborn", types.SimpleNamespace(heatmap=lambda *a, **k: None))
    import matplotlib

    matplotlib.use("Agg")
    spec = importlib.util.spec_from_file_location("paper_bench", PAPER_SCRIPT)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)

    names = [im["file_name"] for im in json.loads(GT_JSON.read_text())["images"]]
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        root = Path(tmp) / "val"
        for sub in ("images", "labels"):
            (root / sub).mkdir(parents=True)
        for n in names:
            (root / "images" / n).symlink_to(VAL / "images" / n)
            (root / "labels" / f"{Path(n).stem}.txt").symlink_to(VAL / "labels" / f"{Path(n).stem}.txt")
        cwd = os.getcwd()
        os.chdir(tmp)                                    # their script saves its confusion-matrix PNG to the cwd
        try:
            rows = bench.benchmark_yolo(str(args.weights), str(root), conf_threshold=0.25)
        finally:
            os.chdir(cwd)
    cname, tp, fp, fn, p, r, f1, miou = rows[0]          # class id 0 (can / COCO person)
    paper = {"class": cname, "TP": tp, "FP": fp, "FN": fn, "precision": p, "recall": r, "f1": f1, "mean_iou": float(miou)}
    ours = json.loads(args.ours.read_text())[args.variant]["paper_protocol"]
    diff = {k: ours[k] - paper[k] for k in ("TP", "FP", "FN", "precision", "recall", "f1", "mean_iou")}
    job = os.environ.get("SLURM_JOB_ID", "local")
    out = {"weights": str(args.weights.resolve().relative_to(REPO)), "ours": str(args.ours.resolve().relative_to(REPO)),
           "variant": args.variant, "paper_script": str(PAPER_SCRIPT.relative_to(REPO)), "n_images": len(names),
           "paper_script_result": paper, "ours_result": ours, "ours_minus_paper": diff}
    dst = SRC / f"cansurf_paper_check_{args.ours.stem.removeprefix('cansurf_paper_')}.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps({"paper": paper, "diff": diff}), "->", dst, f"(job {job})")


if __name__ == "__main__":
    main()
