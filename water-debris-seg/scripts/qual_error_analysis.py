#!/usr/bin/env python3
"""Classify every detection / GT box of the headline box runs and pick examples to render.

Reads eval_outputs/water_debris_seg/box_eval_registry.json (headline runs only) and writes
under eval_outputs/water_debris_seg/qualitative/:

  errors/<run_id>/detections.csv   one row per detection kept at the operating point
  errors/<run_id>/gt.csv           one row per GT box with its error type
  errors/<run_id>/summary.json     counts by error type × size + provenance (sha256, thresholds)
  error_summary.csv                one row per run, all counts (feeds charts / slides)
  selection.json                   images chosen for panels, each with the rule that chose it

Error definitions: water_debris_seg/qualitative/errors.py. Operating point: YOLO runs keep
score >= --yolo-min-score (default 0.25, Ultralytics' predict default); EoMT mask→box runs
keep everything (fixed score 0.5).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from water_debris_seg.qualitative.errors import (  # noqa: E402
    DET_TYPES, GT_TYPES, IOA_FRAGMENT, IOU_MATCH, IOU_NEAR, classify_run, sha256, summarize,
)

PER_CATEGORY = 3
N_RANDOM = 6


def write_csvs(out: Path, records) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "detections.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file_name", "det_id", "x", "y", "w", "h", "score", "area", "max_iou", "status"])
        for r in records.values():
            for d in r.dets:
                w.writerow([r.image_id, r.file_name, d["det_id"], *[round(v, 1) for v in d["bbox"]],
                            round(d["score"], 4), round(d["area"], 1), round(d["max_iou"], 3), d["status"]])
    with (out / "gt.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file_name", "gt_id", "x", "y", "w", "h", "area", "size", "best_iou", "status"])
        for r in records.values():
            for g in r.gts:
                w.writerow([r.image_id, r.file_name, g["gt_id"], *[round(v, 1) for v in g["bbox"]],
                            round(g["area"], 1), g["size"], round(g["best_iou"], 3), g["status"]])


def select(dataset: str, runs: list[dict], recs: dict[str, dict], seed: int) -> list[dict]:
    """Deterministic, rule-based picks. Every pick names its rule and the numbers behind it."""
    eomt = [r for r in runs if r["method"].startswith("EoMT")]
    yolo = [r for r in runs if r["method"].startswith("YOLO")]
    any_rec = recs[runs[0]["id"]]
    ids_with_gt = sorted(i for i, r in any_rec.items() if r.n_gt)
    picks: list[dict] = []
    used: set[int] = set()

    def add(img_id, category, rule, stats):
        if img_id in used:
            return False
        used.add(img_id)
        picks.append({"dataset": dataset, "image_id": img_id, "file_name": any_rec[img_id].file_name,
                      "category": category, "rule": rule, "stats": stats})
        return True

    def top(category, rule, key, run_id, minimum=1):
        rec = recs[run_id]
        ranked = sorted(ids_with_gt, key=lambda i: (-key(rec[i]), i))
        n = 0
        for i in ranked:
            v = key(rec[i])
            if v < minimum or n >= PER_CATEGORY:
                break
            n += add(i, category, rule, {"run": run_id, "value": v, "n_gt": rec[i].n_gt})

    rng = random.Random(seed)
    for i in sorted(rng.sample(ids_with_gt, min(N_RANDOM, len(ids_with_gt)))):
        add(i, "random", f"uniform random among images with GT (seed {seed})", {})

    for r in eomt:
        rid = r["id"]
        top(f"merge__{rid}", f"most GT boxes merged into one detection ({rid})",
            lambda x: x.count("gt", "fn_merged"), rid, minimum=2)
        top(f"missed_small__{rid}", f"most small GT never detected ({rid})",
            lambda x: sum(g["status"] == "fn_missed" and g["size"] == "small" for g in x.gts), rid, minimum=2)
        top(f"bkg_fp__{rid}", f"most background false positives ({rid})",
            lambda x: x.count("det", "fp_bkg"), rid, minimum=2)
        top(f"fragment__{rid}", f"most fragmented GT boxes ({rid})",
            lambda x: x.count("gt", "fn_frag"), rid, minimum=1)

    if yolo and eomt:
        y = recs[yolo[0]["id"]]

        def gap(i):
            best_eomt = max((recs[r["id"]][i].recall for r in eomt), default=0.0)
            return y[i].recall - best_eomt if y[i].n_gt >= 3 else -1

        ranked = sorted(ids_with_gt, key=lambda i: (-gap(i), i))
        n = 0
        for i in ranked:
            if gap(i) < 0.5 or n >= PER_CATEGORY:
                break
            n += add(i, "yolo_finds_eomt_misses",
                     "largest recall gap: YOLO minus best EoMT (>= 3 GT, gap >= 0.5)",
                     {"yolo_recall": round(y[i].recall, 3), "gap": round(gap(i), 3)})
        ranked = sorted((i for i in ids_with_gt if y[i].n_gt >= 2), key=lambda i: (y[i].recall, -y[i].n_gt, i))
        for i in ranked[:PER_CATEGORY]:
            add(i, "yolo_failures", "lowest YOLO recall at operating point (>= 2 GT)",
                {"yolo_recall": round(y[i].recall, 3), "n_gt": y[i].n_gt})
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = Path(__file__).resolve().parents[2] / "eval_outputs" / "water_debris_seg"
    parser.add_argument("--eval-root", type=Path, default=default_root)
    parser.add_argument("--yolo-min-score", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = args.eval_root
    registry = json.loads((root / "box_eval_registry.json").read_text())
    qroot = root / "qualitative"
    runs = [r for r in registry["runs"] if r["headline"] and r.get("qualitative", True)]

    recs, rows = {}, []
    for run in runs:
        gt_entry = registry["ground_truth"][run["dataset"]]
        gt, pred = root / gt_entry["path"], root / run["pred"]
        min_score = args.yolo_min_score if run["method"].startswith("YOLO") else 0.0
        rec = classify_run(gt, pred, min_score=min_score)
        recs[run["id"]] = rec
        out = qroot / "errors" / run["id"]
        write_csvs(out, rec)
        summ = summarize(rec)
        summ["provenance"] = {
            "run_id": run["id"], "method": run["method"], "dataset": run["dataset"],
            "pred": run["pred"], "pred_sha256": sha256(pred), "gt": gt_entry["path"], "gt_sha256": sha256(gt),
            "metrics": run["metrics"], "min_score": min_score, "iou_match": IOU_MATCH, "iou_near": IOU_NEAR,
            "ioa_fragment": IOA_FRAGMENT, "max_dets": 100, "matching": "pycocotools COCOeval evalImgs, areaRng=all",
        }
        (out / "summary.json").write_text(json.dumps(summ, indent=2))
        row = {"run_id": run["id"], "dataset": run["dataset"], "method": run["method"], "min_score": min_score,
               "n_gt": summ["n_gt"], "n_det": summ["n_det_top100"], "recall_iou50": round(summ["recall_at_iou50"], 4)}
        row.update({f"det_{k}": v for k, v in summ["det"].items()})
        for k, by_size in summ["gt"].items():
            row.update({f"gt_{k}_{z}": v for z, v in by_size.items()})
        rows.append(row)
        print(f"{run['id']:30} recall@.5 {summ['recall_at_iou50']:.3f}  det {summ['det']}  "
              f"gt { {k: sum(v.values()) for k, v in summ['gt'].items()} }")

    with (qroot / "error_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    selection = []
    for ds in sorted({r["dataset"] for r in runs}):
        ds_runs = [r for r in runs if r["dataset"] == ds]
        selection += select(ds, ds_runs, {r["id"]: recs[r["id"]] for r in ds_runs}, args.seed)
    meta = {"seed": args.seed, "yolo_min_score": args.yolo_min_score, "per_category": PER_CATEGORY,
            "n_random": N_RANDOM, "det_types": DET_TYPES, "gt_types": GT_TYPES,
            "runs": {r["id"]: {"dataset": r["dataset"], "method": r["method"], "pred": r["pred"]} for r in runs}}
    (qroot / "selection.json").write_text(json.dumps({"meta": meta, "picks": selection}, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"{len(selection)} images selected -> {qroot / 'selection.json'}")


if __name__ == "__main__":
    main()
