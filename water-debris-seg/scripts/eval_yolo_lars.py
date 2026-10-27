#!/usr/bin/env python3
"""Score a LaRS-trained YOLO checkpoint on LaRS val (ablation 10).

* Ultralytics val on eval_val.yaml (its own metrics; crowd boxes are absent from YOLO labels)
* pycocotools on lars_val_gt_coco_boxes.json (crowd kept as iscrowd=1 and ignored) for:
  all 9 classes, the 8 object classes (Static Obstacle excluded), and each class separately.
* Focal-loss checkpoints load like the others (the loss only matters in training).
* For a LoRA checkpoint, also writes and scores a merged plain checkpoint (`*_merged.pt`, loads
  without water_debris_seg) and records that the two give the same numbers.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "water-debris-seg"))
import water_debris_seg.yolo_focal  # noqa: E402,F401  (needed to unpickle focal-loss checkpoints)
import water_debris_seg.yolo_lora  # noqa: E402,F401  (needed to unpickle LoRA checkpoints)

DATA = Path("/scratch-shared/rcampanella/water-debris-seg/data/lars")
STATS = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def predict(weights: Path, gt: dict, imgsz: int, images: Path) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    preds = []
    entries = [(im["id"], images / im["file_name"]) for im in gt["images"]]
    for i in range(0, len(entries), 16):
        chunk = entries[i:i + 16]
        for (iid, _), res in zip(chunk, model.predict([str(p) for _, p in chunk], imgsz=imgsz, conf=0.001, max_det=300, verbose=False)):
            for (x1, y1, x2, y2), sc, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist()):
                preds.append({"image_id": iid, "category_id": int(c), "bbox": [x1, y1, x2 - x1, y2 - y1], "score": sc})
    return preds


def cocoeval(gt_path: Path, preds: list[dict], cat_ids=None) -> tuple[dict, object]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO(str(gt_path))
        d = g.loadRes(preds) if preds else None
        if d is None:
            return {k: 0.0 for k in STATS}, None
        ev = COCOeval(g, d, "bbox")
        if cat_ids is not None:
            ev.params.catIds = cat_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return dict(zip(STATS, [float(s) for s in ev.stats])), ev


def per_class(ev, names: dict) -> dict:
    """AP@[.5:.95], AP50 and AR100 per category from COCOeval's accumulated arrays (area 'all')."""
    out = {}
    p, r = ev.eval["precision"], ev.eval["recall"]   # p: [T, R, K, A, M], r: [T, K, A, M]
    for k, cid in enumerate(ev.params.catIds):
        pk, rk = p[:, :, k, 0, -1], r[:, k, 0, -1]
        valid = pk[pk > -1]
        out[names[cid]] = {"AP": float(valid.mean()) if valid.size else -1.0,
                           "AP50": float(pk[0][pk[0] > -1].mean()) if (pk[0] > -1).any() else -1.0,
                           "AR100": float(rk[rk > -1].mean()) if (rk > -1).any() else -1.0}
    return out


def score(weights: Path, gt_path: Path, data_yaml: Path, imgsz: int, out_dir: Path, images: Path) -> dict:
    from ultralytics import YOLO

    gt = json.loads(gt_path.read_text())
    names = {c["id"]: c["name"] for c in gt["categories"]}
    m = YOLO(str(weights)).val(data=str(data_yaml), split="val", imgsz=imgsz, batch=16, conf=0.001, plots=False,
                               project=str(out_dir), name=f"val_{weights.stem}", exist_ok=True, verbose=False)
    ultra = {"mAP50-95": float(m.box.map), "mAP50": float(m.box.map50), "precision": float(m.box.mp), "recall": float(m.box.mr),
             "per_class_mAP50-95": {names.get(i, m.names.get(i, str(i)) if hasattr(m, "names") else str(i)): float(v)
                                    for i, v in enumerate(m.box.maps)}}   # model may have extra classes (11-class: Water, Sky)
    preds = predict(weights, gt, imgsz, images)
    allc, ev = cocoeval(gt_path, preds)
    things, _ = cocoeval(gt_path, preds, cat_ids=[c for c in names if names[c] != "Static Obstacle"])
    gt_counts = {names[c]: sum(1 for a in gt["annotations"] if a["category_id"] == c and not a["iscrowd"]) for c in names}
    return {"weights": str(weights), "weights_sha256": sha(weights), "gt": str(gt_path), "gt_sha256": sha(gt_path),
            "imgsz": imgsz, "n_pred": len(preds), "ultralytics": ultra, "pycocotools_all_9_classes": allc,
            "pycocotools_objects_8_classes": things, "per_class": per_class(ev, names) if ev else {}, "gt_boxes_per_class": gt_counts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=DATA / "yolo_boxes/eval_val.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--images", type=Path, default=DATA / "val/images", help="folder with the GT file_names (default LaRS val)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from water_debris_seg.yolo_focal import to_plain_detection_model
    from water_debris_seg.yolo_lora import LoRAConv2d, merge_lora

    res = {"checkpoint": score(args.weights, args.gt, args.data, args.imgsz, args.out.parent, args.images)}
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    if any(isinstance(mm, LoRAConv2d) for mm in ckpt["model"].modules()):
        ckpt["model"] = to_plain_detection_model(merge_lora(ckpt["model"].float())).half()
        if ckpt.get("ema") is not None:
            ckpt["ema"] = None
        merged = args.weights.with_name(args.weights.stem + "_merged.pt")
        torch.save(ckpt, merged)
        res["merged_checkpoint"] = score(merged, args.gt, args.data, args.imgsz, args.out.parent, args.images)
        a, b = res["checkpoint"]["pycocotools_all_9_classes"]["mAP"], res["merged_checkpoint"]["pycocotools_all_9_classes"]["mAP"]
        res["merged_matches"] = bool(np.isclose(a, b, atol=2e-3))
    args.out.write_text(json.dumps(res, indent=2))
    c = res["checkpoint"]
    print(json.dumps({"ultralytics_mAP50-95": c["ultralytics"]["mAP50-95"], "coco_all_mAP": c["pycocotools_all_9_classes"]["mAP"],
                      "coco_objects_mAP": c["pycocotools_objects_8_classes"]["mAP"], "merged_matches": res.get("merged_matches")}, indent=1))


if __name__ == "__main__":
    main()
