#!/usr/bin/env python3
"""LaRS-trained YOLO scored on FloW test / CANSURF val (fairness fix 2, cross-domain row; mirrors ablation 09 for EoMT).

Predictions at conf 0.001 (max 300 per image) are mapped onto the dataset's single GT class:
  debris-only (main)        Float (7) and Other (8)
  class-agnostic (reference) all 8 object classes (Static Obstacle, class 0, excluded)
Every prediction keeps its source class. Scorer: pycocotools (same GT files as ablations 04, 05, 08, 09).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "water-debris-seg"))
SETS = {
    "flow": (REPO / "eval_outputs/water_debris_seg/ablations/04_flow_mask2box/sources/flow_test_gt_coco.json",
             REPO / "data/water_debris/flow/test/images"),
    "cansurf": (REPO / "eval_outputs/water_debris_seg/ablations/05_cansurf_small_object/sources/cansurf_val_gt_coco_yolo_aligned.json",
                REPO / "data/water_debris/cansurf_home/val/images"),
}
NAMES = ["Static Obstacle", "Boat/ship", "Row boats", "Paddle board", "Buoy", "Swimmer", "Animal", "Float", "Other"]
STATS = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def score(gt, dets):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not dets:
        return {k: 0.0 for k in STATS}
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO(str(gt))
        ev = COCOeval(g, g.loadRes(dets), "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return dict(zip(STATS, map(float, ev.stats)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--dataset", choices=sorted(SETS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    from ultralytics import YOLO

    gt_path, img_dir = SETS[args.dataset]
    gt = json.loads(gt_path.read_text())
    model = YOLO(str(args.weights))
    debris, agnostic = [], []
    ims = gt["images"]
    for i in range(0, len(ims), 16):
        chunk = ims[i:i + 16]
        for im, r in zip(chunk, model.predict([str(img_dir / m["file_name"]) for m in chunk], imgsz=args.imgsz, conf=0.001,
                                             max_det=300, verbose=False)):
            for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                c = int(c)
                if c == 0 or c >= len(NAMES):
                    continue
                d = {"image_id": im["id"], "category_id": 0, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s, "src_class": NAMES[c]}
                agnostic.append(d)
                if c in (7, 8):
                    debris.append(d)
    res = {"weights": str(args.weights), "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
           "dataset": args.dataset, "gt": str(gt_path), "gt_sha256": hashlib.sha256(gt_path.read_bytes()).hexdigest(),
           "debris_classes": ["Float", "Other"], "debris_only": score(gt_path, debris), "class_agnostic": score(gt_path, agnostic),
           "n_dets_debris": len(debris), "n_dets_agnostic": len(agnostic)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2))
    args.out.with_suffix(".preds.json").write_text(json.dumps(agnostic))
    print(json.dumps({k: res[k] for k in ("debris_only", "class_agnostic")}, indent=1))


if __name__ == "__main__":
    main()
