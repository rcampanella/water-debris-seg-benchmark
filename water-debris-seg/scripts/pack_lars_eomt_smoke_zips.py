#!/usr/bin/env python3
"""Build tiny COCO-layout zips from LaRS for EoMT validate smoke (8 train + 8 val)."""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from water_debris_seg.datasets.lars import verify_lars_layout


def _pick_images(split_dir: Path, n: int) -> list[str]:
    imgs = sorted(p.name for p in (split_dir / "images").glob("*.jpg"))[:n]
    if len(imgs) < n:
        raise RuntimeError(f"not enough images in {split_dir}")
    return imgs


def _filter_panoptic_json(src_json: Path, keep: set[str], out_json: Path) -> None:
    data = json.loads(src_json.read_text(encoding="utf-8"))
    images = [im for im in data["images"] if im["file_name"] in keep]
    image_ids = {im["id"] for im in images}
    annotations = [a for a in data["annotations"] if a["image_id"] in image_ids]
    out = {**data, "images": images, "annotations": annotations}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out), encoding="utf-8")


def _zip_dir(files: list[tuple[Path, str]], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for src, arc in files:
            zf.write(src, arc)


def _nested_panoptic_zip(
    names: list[str],
    split: str,
    layout_root: Path,
    folder: str,
) -> bytes:
    """COCO layout: annotations/panoptic_{split}2017.zip contains {folder}/*.png."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as inner:
        for n in names:
            png = layout_root / split / "panoptic_masks" / n.replace(".jpg", ".png")
            inner.write(png, f"{folder}/{n.replace('.jpg', '.png')}")
    return buf.getvalue()


def main() -> int:
    layout = verify_lars_layout()
    out_root = layout.root / "eomt_smoke_zips"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    train_names = _pick_images(layout.root / "train", 8)
    val_names = _pick_images(layout.root / "val", 8)

    for split, names, zip_name, folder in (
        ("train", train_names, "train2017.zip", "train2017"),
        ("val", val_names, "val2017.zip", "val2017"),
    ):
        files = [
            (layout.root / split / "images" / n, f"{folder}/{n}") for n in names
        ]
        _zip_dir(files, out_root / zip_name)

    ann_dir = out_root / "_panoptic_build"
    ann_dir.mkdir()
    _filter_panoptic_json(
        layout.root / "train" / "panoptic_annotations.json",
        set(train_names),
        ann_dir / "panoptic_train2017.json",
    )
    _filter_panoptic_json(
        layout.root / "val" / "panoptic_annotations.json",
        set(val_names),
        ann_dir / "panoptic_val2017.json",
    )

    pan_zip = out_root / "panoptic_annotations_trainval2017.zip"
    with zipfile.ZipFile(pan_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for split, names, folder in (
            ("train", train_names, "panoptic_train2017"),
            ("val", val_names, "panoptic_val2017"),
        ):
            zf.write(
                ann_dir / f"panoptic_{split}2017.json",
                f"annotations/panoptic_{split}2017.json",
            )
            nested = _nested_panoptic_zip(names, split, layout.root, folder)
            zf.writestr(f"annotations/panoptic_{split}2017.zip", nested)

    shutil.rmtree(ann_dir)
    print("eomt_smoke_data_path", out_root)
    print("train2017.zip", out_root / "train2017.zip")
    print("val2017.zip", out_root / "val2017.zip")
    print("panoptic_annotations_trainval2017.zip", pan_zip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
