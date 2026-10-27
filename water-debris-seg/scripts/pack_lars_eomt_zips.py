#!/usr/bin/env python3
"""Build COCO-layout zips from LaRS for EoMT (smoke subset or full train/val)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from water_debris_seg.datasets.lars import verify_lars_layout


def _pick_images(split_dir: Path, n: int | None) -> list[str]:
    imgs = sorted(p.name for p in (split_dir / "images").glob("*.jpg"))
    if n is not None:
        imgs = imgs[:n]
    if not imgs:
        raise RuntimeError(f"no images in {split_dir}")
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
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as inner:
        for n in names:
            png = layout_root / split / "panoptic_masks" / n.replace(".jpg", ".png")
            inner.write(png, f"{folder}/{n.replace('.jpg', '.png')}")
    return buf.getvalue()


def pack(
    layout_root: Path,
    out_root: Path,
    smoke_n: int | None,
    train: tuple[str, list[str]] | None = None,
    val: tuple[str, list[str]] | None = None,
) -> None:
    """train / val: (LaRS source split, image names); default = LaRS train / LaRS val (optionally the first smoke_n)."""
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    train_src, train_names = train or ("train", _pick_images(layout_root / "train", smoke_n))
    val_src, val_names = val or ("val", _pick_images(layout_root / "val", smoke_n))

    for split, names, zip_name, folder in (
        (train_src, train_names, "train2017.zip", "train2017"),
        (val_src, val_names, "val2017.zip", "val2017"),
    ):
        files = [
            (layout_root / split / "images" / n, f"{folder}/{n}") for n in names
        ]
        _zip_dir(files, out_root / zip_name)

    ann_dir = out_root / "_panoptic_build"
    ann_dir.mkdir()
    _filter_panoptic_json(
        layout_root / train_src / "panoptic_annotations.json",
        set(train_names),
        ann_dir / "panoptic_train2017.json",
    )
    _filter_panoptic_json(
        layout_root / val_src / "panoptic_annotations.json",
        set(val_names),
        ann_dir / "panoptic_val2017.json",
    )

    pan_zip = out_root / "panoptic_annotations_trainval2017.zip"
    with zipfile.ZipFile(pan_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for split, src, names, folder in (
            ("train", train_src, train_names, "panoptic_train2017"),
            ("val", val_src, val_names, "panoptic_val2017"),
        ):
            zf.write(
                ann_dir / f"panoptic_{split}2017.json",
                f"annotations/panoptic_{split}2017.json",
            )
            nested = _nested_panoptic_zip(names, src, layout_root, folder)
            zf.writestr(f"annotations/panoptic_{split}2017.zip", nested)

    shutil.rmtree(ann_dir)
    print("eomt_data_path", out_root)
    print("train_images", len(train_names), "val_images", len(val_names))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Pack all LaRS train/val images (for ablation 01).",
    )
    parser.add_argument(
        "--smoke",
        type=int,
        default=8,
        help="Images per split for smoke zips (default 8).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: .../eomt_smoke_zips or .../eomt_full_zips).",
    )
    parser.add_argument(
        "--heldout-stage",
        choices=["A", "B"],
        default=None,
        help="Held-out protocol (fairness fixes 3/4): A = LaRS train minus held10 / held10; "
        "B = all LaRS train / held10 (monitored only). held10 = ablation 10's 245-image slice.",
    )
    parser.add_argument(
        "--held10",
        type=Path,
        default=ROOT.parent / "data/water_debris/lars_backup/yolo_boxes/held10/labels",
        help="Folder whose *.txt stems name the held-out images.",
    )
    args = parser.parse_args()
    layout = verify_lars_layout()
    if args.heldout_stage:
        held = sorted(f"{p.stem}.jpg" for p in args.held10.glob("*.txt"))
        all_train = _pick_images(layout.root / "train", None)
        missing = set(held) - set(all_train)
        if missing:
            raise RuntimeError(f"{len(missing)} held-out images not in LaRS train, e.g. {sorted(missing)[:3]}")
        train = [n for n in all_train if n not in set(held)] if args.heldout_stage == "A" else all_train
        out_root = args.out or (ROOT.parent / "data/water_debris/lars_eomt_zips" / f"stage{args.heldout_stage}")
        pack(layout.root, out_root, None, train=("train", train), val=("train", held))
        return 0
    n = None if args.full else args.smoke
    default_out = layout.root / ("eomt_full_zips" if args.full else "eomt_smoke_zips")
    out_root = args.out or default_out
    pack(layout.root, out_root, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
