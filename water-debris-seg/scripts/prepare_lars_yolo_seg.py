#!/usr/bin/env python3
"""Build YOLO segmentation labels for LaRS from the panoptic instance masks (ablation 11, YOLO26-seg).

Same 9 classes, same instances and same crowd handling as the ablation 10 box labels (prepare_lars_yolo.py): Static
Obstacle + the 8 LaRS object classes, Water and Sky dropped, crowd segments left out. Each instance mask becomes one
polygon (cv2 outer contours, CHAIN_APPROX_NONE); an instance in several pieces is joined into one polygon with
Ultralytics' merge_multi_segment, as its COCO converter does. Holes are lost (the YOLO format has none).

Ultralytics scores mask AP against masks rasterised from these polygons, not against the original masks; the
polygon-vs-mask IoU of every val instance is written to the stats JSON so that gap is on record.

Splits: train_all = official LaRS train, val = official LaRS val (the test split here; LaRS test labels are not public).

--with-water-sky (LaRS PQ needs region predictions): also Water (id 9) and Sky (id 10), one polygon per region segment,
ids 0-8 unchanged, as prepare_lars_yolo.py --with-water-sky for boxes. A region's holes (objects inside the water) are
filled; at PQ time objects are merged first, so they take those pixels back. Outputs go to yolo_seg11 and the stats
file gets the prefix lars11_.

Outputs
  <out>/{train_all,val}/{images (symlinks), labels}, <out>/train_val.yaml
  <eval>/sources/lars_yolo_seg_label_stats.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from prepare_lars_yolo import KEEP, NAMES, YOLO_ID


def polygon(mask: np.ndarray):
    """One polygon (N x 2, pixels) for a binary mask, or None if no contour has 3+ points."""
    from ultralytics.data.converter import merge_multi_segment

    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c.reshape(-1, 2) for c in cs if len(c) >= 3]
    if not cs:
        return None, 0
    if len(cs) == 1:
        return cs[0].astype(np.float64), 1
    return np.concatenate(merge_multi_segment([c.reshape(-1).tolist() for c in cs]), axis=0).astype(np.float64), len(cs)


def mask_iou(poly: np.ndarray, mask: np.ndarray) -> float:
    r = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(r, [np.round(poly).astype(np.int32)], 1)
    inter, union = np.logical_and(r, mask).sum(), np.logical_or(r, mask).sum()
    return float(inter / union) if union else 0.0


def build(lars: Path, split: str, name: str, out: Path, with_iou: bool):
    pan = json.loads((lars / split / "panoptic_annotations.json").read_text())
    images = {im["id"]: im for im in pan["images"]}
    (out / name / "images").mkdir(parents=True, exist_ok=True)
    (out / name / "labels").mkdir(parents=True, exist_ok=True)
    cnt, crowd, multi, empty, ious = Counter(), Counter(), Counter(), 0, defaultdict(list)
    for a in sorted(pan["annotations"], key=lambda a: a["file_name"]):
        im = images[a["image_id"]]
        W, H = im["width"], im["height"]
        stem = Path(a["file_name"]).stem
        rgb = np.asarray(Image.open(lars / split / "panoptic_masks" / a["file_name"]).convert("RGB")).astype(np.int64)
        ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
        lines = []
        for s in a["segments_info"]:
            if s["category_id"] not in KEEP:
                continue
            cls = KEEP[s["category_id"]]
            if s.get("iscrowd", 0):
                crowd[cls] += 1
                continue
            m = ids == s["id"]
            if not m.any():
                continue
            p, parts = polygon(m)
            if p is None:
                empty += 1
                continue
            cnt[cls] += 1
            multi[cls] += parts > 1
            if with_iou:
                ious[cls].append(mask_iou(p, m))
            xy = " ".join(f"{x / W:.6f} {y / H:.6f}" for x, y in np.clip(p, 0, [W - 1, H - 1]))
            lines.append(f"{YOLO_ID[s['category_id']]} {xy}")
        link = out / name / "images" / f"{stem}.jpg"
        if not link.exists():
            link.symlink_to(lars / split / "images" / f"{stem}.jpg")
        (out / name / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    st = {"images": len(pan["annotations"]), "instances_per_class": dict(cnt), "crowd_excluded": dict(crowd),
          "instances_in_several_pieces_merged": dict(multi), "instances_without_polygon": empty}
    if with_iou:
        st["polygon_vs_mask_iou"] = {c: {"n": len(v), "mean": float(np.mean(v)), "p05": float(np.percentile(v, 5)),
                                         "min": float(np.min(v))} for c, v in ious.items()}
        allv = [x for v in ious.values() for x in v]
        st["polygon_vs_mask_iou"]["all"] = {"n": len(allv), "mean": float(np.mean(allv)),
                                            "p05": float(np.percentile(allv, 5)), "min": float(np.min(allv))}
    return st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lars", type=Path, default=Path("/scratch-shared/rcampanella/water-debris-seg/data/lars"))
    ap.add_argument("--out", type=Path, default=Path("/scratch-shared/rcampanella/water-debris-seg/data/lars/yolo_seg"))
    ap.add_argument("--with-water-sky", action="store_true", help="11 classes: + Water (9), Sky (10)")
    ap.add_argument("--eval-dir", type=Path, default=Path(__file__).resolve().parents[2] / "eval_outputs/water_debris_seg/ablations/11_yolo26_family")
    args = ap.parse_args()
    global YOLO_ID, NAMES
    pre = ""
    if args.with_water_sky:
        KEEP.update({3: "Water", 5: "Sky"})
        YOLO_ID = {cat: i for i, cat in enumerate(KEEP)}
        NAMES = list(KEEP.values())
        pre = "lars11_"
        if args.out == ap.get_default("out"):
            args.out = args.out.with_name("yolo_seg11")
    stats = {"train_all": build(args.lars, "train", "train_all", args.out, with_iou=False),
             "val": build(args.lars, "val", "val", args.out, with_iou=True),
             "class_index": dict(enumerate(NAMES))}
    (args.out / "train_val.yaml").write_text(f"path: {args.out}\ntrain: train_all/images\nval: val/images\nnc: {len(NAMES)}\nnames: {json.dumps(NAMES)}\n")
    (args.eval_dir / "sources").mkdir(parents=True, exist_ok=True)
    (args.eval_dir / "sources" / f"{pre}lars_yolo_seg_label_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
