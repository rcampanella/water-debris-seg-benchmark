#!/usr/bin/env python3
"""CANSURF val scored with the CANSURF paper's detection protocol (arXiv 2605.16774, Table II), separate from our box AP.

Protocol, reimplemented from the paper's `yolov11_benchmark.py` (ZaidAljundiHW2/CANSURF @ 449ede6):
  predictions with confidence > 0.25; per image, predictions in descending score order are matched greedily to the
  best-IoU unmatched GT box of the same class; at each IoU threshold 0.50:0.05:0.95 a prediction is a TP if that IoU
  reaches the threshold, else an FP; unmatched GT boxes are FNs. TP / FP / FN are summed over the 10 thresholds, then
  P = TP / (TP + FP), R = TP / (TP + FN), F1. Mean IoU = mean IoU of all TPs. GT = the YOLO label files, converted to
  pixels as the paper's script does. Their FN check looks a GT box up with list.index (an identical duplicate box takes
  the first copy's match); kept as-is so the numbers follow their code.
Also reported (references, not the paper's numbers): the same counts at IoU 0.50 only, and the best F1 over
confidence thresholds 0.05:0.05:0.95 (EoMT's scores are calibrated differently from YOLO's, so a fixed 0.25 can
favour one model).

Images: the 2,209 of our scorer's GT (cansurf_val_gt_coco_yolo_aligned.json: released val minus 12 images whose labels
fall outside the frame). The paper's Table II used a 900-image curated test set that is not released, so these numbers
are not comparable to Table II, only to each other.

Input: a predictions file (COCO detections, bbox xywh, `score`, `src_class`, optional `src_class_id`) written by our
scorers, or `--weights` to run a YOLO model first (conf 0.001, then filtered > 0.25; same boxes as predicting at 0.25).
Variants: class_agnostic (every prediction is a can; our main-table setting), debris_only (src_class in the scorer
JSON's debris_classes), and for raw YOLO COCO weights paper_class_id0 (only COCO class id 0 matches the can class id 0,
what the paper's script does with an untrained COCO model).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GT_JSON = REPO / "eval_outputs/water_debris_seg/ablations/05_cansurf_small_object/sources/cansurf_val_gt_coco_yolo_aligned.json"
VAL = REPO / "data/water_debris/cansurf_home/val"
SRC = REPO / "eval_outputs/water_debris_seg/fairness_fixes/sources"
PAPER_SCRIPT = REPO / "data/water_debris/third_party/cansurf_benchmark/yolov11_benchmark.py"
IOUS = np.arange(0.5, 1.0, 0.05)
CONF = 0.25
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def iou(a, b) -> float:  # paper's calculate_iou (xyxy, no +1)
    xa, ya, xb, yb = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return 0.0 if union == 0 else inter / union


def load_gt():
    from PIL import Image

    doc = json.loads(GT_JSON.read_text())
    gt = {}
    for im in doc["images"]:
        w, h = Image.open(VAL / "images" / im["file_name"]).size
        boxes = []
        for line in (VAL / "labels" / f"{Path(im['file_name']).stem}.txt").read_text().splitlines():
            p = list(map(float, line.strip().split()))
            if len(p) != 5:
                continue
            xc, yc, bw, bh = p[1:]
            boxes.append({"bbox": [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h],
                          "class_id": int(p[0])})
        gt[im["id"]] = boxes
    return doc, gt


def match(preds, gts, iou_t):
    """Paper's greedy loop for one image and one threshold; returns per prediction (is_tp, gt_idx, iou)."""
    matched, out = set(), []
    for pb, pc in preds:
        best, best_j = 0, -1
        for j, g in enumerate(gts):
            if g["class_id"] != pc or j in matched:
                continue
            v = iou(pb, g["bbox"])
            if v > best:
                best, best_j = v, j
        if best >= iou_t:
            matched.add(best_j)
            out.append((True, best_j, best))
        else:
            out.append((False, -1, 0.0))
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "TP": tp, "FP": fp, "FN": fn}


def score(gt, dets, cls_of, gt_cls=0):
    """dets: list of COCO detections; cls_of(d) -> class id compared with the GT class id (0)."""
    by_img = defaultdict(list)
    for d in dets:
        if d["score"] > GRID[0] - 1e-9:
            by_img[d["image_id"]].append(d)
    # rec[iou_idx] -> (score, is GT class, is_tp, matched GT index, iou, image); the predictions above any confidence
    # are a prefix of the score-sorted list, so one greedy pass serves every threshold
    rec = {k: [] for k in range(len(IOUS))}
    n_gt = 0
    for img, gts in gt.items():
        n_gt += sum(g["class_id"] == gt_cls for g in gts)
        ds = sorted(by_img.get(img, []), key=lambda d: -d["score"])
        preds = [([d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2], d["bbox"][1] + d["bbox"][3]], cls_of(d))
                 for d in ds]
        for k, t in enumerate(IOUS):
            for d, (pb, pc), (tp, j, v) in zip(ds, preds, match(preds, gts, t)):
                rec[k].append((d["score"], pc == gt_cls, tp, j, v, img))

    def counts(conf, ks):
        tp = fp = fn = 0
        ious = []
        for k in ks:
            found = defaultdict(set)
            for s, is_gt_cls, is_tp, j, v, img in rec[k]:
                if s <= conf or not is_gt_cls:
                    continue
                if is_tp:
                    tp += 1
                    found[img].add(j)
                    ious.append(v)
                else:
                    fp += 1
            for img, gts in gt.items():  # paper's FN check: gts.index(g) (first identical copy) not matched
                fn += sum(1 for g in gts if g["class_id"] == gt_cls and gts.index(g) not in found[img])
        return tp, fp, fn, ious

    tp, fp, fn, ious = counts(CONF, range(len(IOUS)))
    paper = {**prf(tp, fp, fn), "mean_iou": float(np.mean(ious)) if ious else 0.0}
    tp, fp, fn, _ = counts(CONF, [0])
    iou50 = prf(tp, fp, fn)
    sweep = {}
    for c in GRID:
        tp, fp, fn, _ = counts(c, range(len(IOUS)))
        sweep[f"{c:.2f}"] = prf(tp, fp, fn)
    best = max(sweep, key=lambda c: sweep[c]["f1"])
    return {"paper_protocol": paper, "iou50_conf025": iou50, "best_f1_conf": {"conf": float(best), **sweep[best]},
            "conf_sweep": sweep, "n_dets_scored": sum(len(v) for v in by_img.values()), "n_gt": n_gt}


def yolo_predict(weights: Path, doc) -> list:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    dets = []
    ims = doc["images"]
    for im in ims:  # one image per call, as the paper's script (batches letterbox mixed image sizes differently)
        for r in model.predict(str(VAL / "images" / im["file_name"]), imgsz=640, conf=0.001, max_det=300, verbose=False):
            for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                dets.append({"image_id": im["id"], "category_id": 0, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s,
                             "src_class": model.names[int(c)], "src_class_id": int(c)})
    return dets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, help="name of the output JSON")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preds", type=Path, help="our scorer's .preds.json")
    g.add_argument("--weights", type=Path, help="YOLO weights: predict CANSURF val first")
    ap.add_argument("--scorer-json", type=Path, help="our scorer's JSON for the same predictions (debris classes, provenance)")
    ap.add_argument("--debris-classes", help="comma-separated source classes for debris_only (overrides the scorer JSON)")
    args = ap.parse_args()

    doc, gt = load_gt()
    job = os.environ.get("SLURM_JOB_ID", "local")
    out = {"method": args.method, "split": "CANSURF val (2,209 images, our aligned GT image list; YOLO label files as GT)",
           "protocol": "CANSURF paper yolov11_benchmark.py @ 449ede6 (reimplemented): conf > 0.25, greedy class-aware "
                       "matching, TP/FP/FN summed over IoU 0.50:0.05:0.95",
           "paper_script_sha256": sha(PAPER_SCRIPT), "gt_image_list": str(GT_JSON.relative_to(REPO)),
           "gt_image_list_sha256": sha(GT_JSON), "n_images": len(gt)}
    if args.weights:
        dets = yolo_predict(args.weights, doc)
        preds_path = SRC / f"cansurf_paper_{args.method}_{job}.preds.json"
        preds_path.write_text(json.dumps(dets))
        out.update(weights=str(args.weights.resolve().relative_to(REPO)), weights_sha256=sha(args.weights))
    else:
        preds_path = args.preds
        dets = json.loads(preds_path.read_text())
    out.update(preds=str(preds_path.resolve().relative_to(REPO)), preds_sha256=sha(preds_path))
    debris = None
    if args.scorer_json:
        sj = json.loads(args.scorer_json.read_text())
        out["scorer_json"] = str(args.scorer_json.resolve().relative_to(REPO))
        debris = sj.get("debris_classes") or sj.get("box_AP", {}).get("debris_classes")
    if args.debris_classes:
        debris = args.debris_classes.split(",")
    out["debris_classes"] = debris
    out["class_agnostic"] = score(gt, dets, lambda d: 0)
    if debris:
        out["debris_only"] = score(gt, [d for d in dets if d["src_class"] in debris], lambda d: 0)
    if dets and "src_class_id" in dets[0] and len({d["src_class"] for d in dets}) > 1:
        out["paper_class_id0"] = score(gt, dets, lambda d: d["src_class_id"])
    dst = SRC / f"cansurf_paper_{args.method}_{job}.json"
    dst.write_text(json.dumps(out, indent=2))
    show = {k: {m: round(v, 3) for m, v in out[k]["paper_protocol"].items() if m in ("precision", "recall", "f1")}
            for k in ("class_agnostic", "debris_only", "paper_class_id0") if k in out}
    print(json.dumps(show), "->", dst)


if __name__ == "__main__":
    main()
