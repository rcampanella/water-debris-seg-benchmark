#!/usr/bin/env python3
"""Score exported LaRS val panoptic PNGs with the official LaRS evaluator (lojzezust/lars_evaluator).

Runs `evaluate.py <config> <method>` on data/water_debris/lars_eval/preds/<method>/ and writes
eval_outputs/water_debris_seg/fairness_fixes/sources/lars_official_<method>_<job>.json with the evaluator's summary
(PQ / RQ / SQ overall, things, stuff; obstacle-class-agnostic aPQ; semantic metrics, all in %), the evaluator commit,
the SHA-256 of the prediction set and of the GT panoptic masks, and the scorer JSON the PNGs came from. The evaluator's
own result files are copied to fairness_fixes/lars_official/<method>/.

Run with the evaluator's env (~/envs/lars-eval: numpy 1.26, the evaluator overflows uint8 under numpy 2).
Differences from our PQ (EoMT's metric): object predictions lying mostly on Static Obstacle are ignored, not false
positives; Static Obstacle GT segments mostly covered by void/water/sky predictions are not false negatives
(lars_eval/panoptic.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "data/water_debris/third_party/lars_evaluator"
ROOT = REPO / "data/water_debris/lars_eval"
SRC = REPO / "eval_outputs/water_debris_seg/fairness_fixes/sources"
KEEP = REPO / "eval_outputs/water_debris_seg/fairness_fixes/lars_official"


def tree_sha(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.glob("*.png")):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, help="prediction dir name under lars_eval/preds")
    ap.add_argument("--scorer-json", type=Path, help="our scorer's JSON for the same predictions (provenance)")
    args = ap.parse_args()

    cfg = ROOT / "val_panoptic.yaml"
    preds = ROOT / "preds" / args.method
    images = (ROOT / "val/image_list.txt").read_text().split()
    missing = [s for s in images if not (preds / f"{s}.png").exists()]
    if missing:
        sys.exit(f"{len(missing)} prediction PNGs missing, e.g. {missing[:3]}")
    subprocess.run([sys.executable, "evaluate.py", str(cfg), args.method], cwd=EVAL, check=True,
                   stdout=subprocess.DEVNULL)
    res = ROOT / "results" / args.method
    summary = json.loads((res / "summary.json").read_text())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=EVAL, capture_output=True, text=True).stdout.strip()
    out = {"method": args.method, "evaluator": "lojzezust/lars_evaluator", "evaluator_commit": commit,
           "config": str(cfg.resolve().relative_to(REPO)), "split": "LaRS v1.0.0 val", "n_images": len(images),
           "predictions_sha256": tree_sha(preds), "gt_panoptic_masks_sha256": tree_sha(ROOT / "val/panoptic_masks"),
           "scorer_json": str(args.scorer_json.resolve().relative_to(REPO)) if args.scorer_json else None,
           "units": "percent", "summary": summary}
    job = os.environ.get("SLURM_JOB_ID", "local")
    dst = SRC / f"lars_official_{args.method}_{job}.json"
    dst.write_text(json.dumps(out, indent=2))
    keep = KEEP / args.method
    keep.mkdir(parents=True, exist_ok=True)
    for f in res.iterdir():
        shutil.copy2(f, keep / f.name)
    print(json.dumps({k: round(summary[k], 2) for k in ("PQ", "PQ_th", "PQ_st", "aPQ", "aPQ_th", "sem_mIoU")}), "->", dst)


if __name__ == "__main__":
    main()
