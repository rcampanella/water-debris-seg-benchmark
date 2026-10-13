#!/usr/bin/env python3
"""Pack LaRS train/val images + semantic_masks into lars_semantic.zip for EoMT."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from water_debris_seg.datasets.lars import verify_lars_layout


def pack(layout_root: Path, out_zip: Path, smoke_n: int | None) -> None:
    layout = verify_lars_layout(layout_root)
    staging = out_zip.parent / f"_lars_semantic_staging_{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)

    for split, ade_split in (("train", "training"), ("val", "validation")):
        img_dir = layout_root / split / "images"
        mask_dir = layout_root / split / "semantic_masks"
        names = sorted(p.name for p in img_dir.glob("*.jpg"))
        if smoke_n is not None:
            names = names[:smoke_n]
        out_img = staging / "images" / ade_split
        out_ann = staging / "annotations" / ade_split
        out_img.mkdir(parents=True, exist_ok=True)
        out_ann.mkdir(parents=True, exist_ok=True)
        for n in names:
            shutil.copy2(img_dir / n, out_img / n)
            png = n.replace(".jpg", ".png")
            shutil.copy2(mask_dir / png, out_ann / png)

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in staging.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(staging).as_posix())
    shutil.rmtree(staging, ignore_errors=True)
    n_train = len(list((layout_root / "train" / "images").glob("*.jpg")))
    n_val = len(list((layout_root / "val" / "images").glob("*.jpg")))
    print(f"wrote {out_zip} (train up to {n_train}, val up to {n_val}, smoke_n={smoke_n})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=0, help="If >0, limit images per split")
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Do not repack if output zip already exists",
    )
    args = parser.parse_args()
    from water_debris_seg.paths import scratch_root

    layout = verify_lars_layout().root
    out = scratch_root() / "data/lars/eomt_semantic_zips/lars_semantic.zip"
    if args.skip_if_exists and out.is_file() and out.stat().st_size > 1_000_000:
        print(f"skip existing {out} ({out.stat().st_size} bytes)")
        return
    smoke_n = None if args.full else (args.smoke if args.smoke > 0 else None)
    if not args.full and smoke_n is None:
        smoke_n = 16
    pack(layout, out, smoke_n)


if __name__ == "__main__":
    main()
