#!/usr/bin/env python3
"""Convert instance masks to COCO detection boxes (mask→box eval helper)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def mask_to_xywh(mask: np.ndarray) -> list[float] | None:
    """Tight axis-aligned box for a binary mask; COCO format [x, y, w, h]."""
    if mask.ndim != 2:
        raise ValueError(f"expected H×W mask, got shape {mask.shape}")
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def masks_to_detections(
    masks: Iterable[np.ndarray],
    *,
    category_id: int,
    image_id: int,
    scores: Iterable[float] | None = None,
    ann_id_start: int = 1,
) -> tuple[list[dict], int]:
    """Build COCO ``annotations`` entries from binary masks."""
    anns: list[dict] = []
    ann_id = ann_id_start
    score_list = list(scores) if scores is not None else None
    for i, mask in enumerate(masks):
        box = mask_to_xywh(np.asarray(mask, dtype=bool))
        if box is None:
            continue
        area = float(box[2] * box[3])
        entry: dict = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": box,
            "area": area,
            "iscrowd": 0,
        }
        if score_list is not None:
            entry["score"] = float(score_list[i])
        anns.append(entry)
        ann_id += 1
    return anns, ann_id


def yolo_label_to_coco_bbox(
    cx: float,
    cy: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> list[float]:
    """YOLO normalized center box → COCO [x, y, w, h] pixels."""
    bw, bh = w * img_w, h * img_h
    x = cx * img_w - bw / 2
    y = cy * img_h - bh / 2
    return [float(x), float(y), float(bw), float(bh)]


def load_voc_split_to_coco(
    images_dir: Path,
    annotations_dir: Path,
    *,
    category_id: int = 0,
    class_names: set[str] | None = None,
) -> dict:
    """Minimal COCO detection JSON from Pascal VOC XML (single-class eval friendly)."""
    import xml.etree.ElementTree as ET

    if class_names is None:
        class_names = {"bottle", "waste", "floating waste", "object"}

    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 1
    ann_id = 1
    exts = {".jpg", ".jpeg", ".png", ".JPG"}

    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix not in exts:
            continue
        xml_path = annotations_dir / f"{img_path.stem}.xml"
        if not xml_path.is_file():
            continue
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        if size is not None:
            img_w = int(size.findtext("width", "0"))
            img_h = int(size.findtext("height", "0"))
        else:
            from PIL import Image

            with Image.open(img_path) as im:
                img_w, img_h = im.size
        images.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
            }
        )
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if class_names and name and name not in class_names:
                continue
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = float(bb.findtext("xmin") or 0)
            ymin = float(bb.findtext("ymin") or 0)
            xmax = float(bb.findtext("xmax") or 0)
            ymax = float(bb.findtext("ymax") or 0)
            w, h = xmax - xmin, ymax - ymin
            if w <= 0 or h <= 0:
                continue
            bbox = [xmin, ymin, w, h]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        image_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": category_id, "name": "waste"}],
    }


def load_yolo_split_to_coco(
    images_dir: Path,
    labels_dir: Path,
    *,
    category_id: int = 0,
) -> dict:
    """Minimal COCO detection JSON from a YOLO val/train split (single class)."""
    from PIL import Image

    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 1
    ann_id = 1
    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        with Image.open(img_path) as im:
            img_w, img_h = im.size
        images.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
            }
        )
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _cls, cx, cy, w, h = map(float, parts[:5])
                bbox = yolo_label_to_coco_bbox(cx, cy, w, h, img_w, img_h)
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        image_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": category_id, "name": "object"}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    yolo = sub.add_parser("yolo-split", help="Export YOLO split to COCO detection JSON")
    yolo.add_argument("--images", type=Path, required=True)
    yolo.add_argument("--labels", type=Path, required=True)
    yolo.add_argument("--out", type=Path, required=True)
    yolo.add_argument("--category-id", type=int, default=0)

    voc = sub.add_parser("voc-split", help="Export Pascal VOC XML split to COCO detection JSON")
    voc.add_argument("--images", type=Path, required=True)
    voc.add_argument("--annotations", type=Path, required=True)
    voc.add_argument("--out", type=Path, required=True)
    voc.add_argument("--category-id", type=int, default=0)

    args = parser.parse_args()
    if args.cmd == "voc-split":
        coco = load_voc_split_to_coco(args.images, args.annotations, category_id=args.category_id)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(coco), encoding="utf-8")
        print(f"wrote {args.out} ({len(coco['images'])} images, {len(coco['annotations'])} boxes)")
    elif args.cmd == "yolo-split":
        coco = load_yolo_split_to_coco(
            args.images, args.labels, category_id=args.category_id
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(coco), encoding="utf-8")
        print(f"wrote {args.out} ({len(coco['images'])} images, {len(coco['annotations'])} boxes)")


if __name__ == "__main__":
    main()
