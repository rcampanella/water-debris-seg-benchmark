"""FloW-Img layout checks (post ORCA download)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from water_debris_seg.paths import dataset_dir


@dataclass(frozen=True)
class FlowLayout:
    root: Path
    split: str
    images_dir: Path
    annotations_dir: Path  # VOC XML
    labels_dir: Path | None  # YOLO txt if present


def default_flow_root() -> Path:
    return dataset_dir("flow")


def _find_split_root(root: Path, split: str) -> Path | None:
    candidates = [
        root / split,
        root / "FloW-Img" / split,
        root / f"FloW-Img_{split}",
        root / "images" / split,
    ]
    for c in candidates:
        if (c / "images").is_dir() or (c / "JPEGImages").is_dir():
            return c
    return None


def verify_flow_layout(
    root: Path | None = None,
    *,
    split: str = "test",
) -> FlowLayout:
    root = root or default_flow_root()
    if root.is_symlink():
        root = root.resolve()

    split_root = _find_split_root(root, split)
    if split_root is None:
        raise FileNotFoundError(
            f"No FloW-Img {split} split under {root}. "
            "Expected e.g. test/images/ + test/Annotations/ (VOC). "
            "See water-debris-seg/docs/FLOW_DATA.md"
        )

    images = split_root / "images"
    if not images.is_dir():
        images = split_root / "JPEGImages"
    ann = split_root / "Annotations"
    if not ann.is_dir():
        ann = split_root / "annotations"
    labels = split_root / "labels"
    labels_dir = labels if labels.is_dir() else None

    if not images.is_dir():
        raise FileNotFoundError(f"Missing images dir under {split_root}")
    if not ann.is_dir() and labels_dir is None:
        raise FileNotFoundError(
            f"Need VOC Annotations/ or YOLO labels/ under {split_root}"
        )

    n_img = sum(1 for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if n_img < 10:
        raise FileNotFoundError(f"Too few images in {images} ({n_img})")

    return FlowLayout(
        root=root,
        split=split,
        images_dir=images,
        annotations_dir=ann if ann.is_dir() else images,  # placeholder if YOLO-only
        labels_dir=labels_dir,
    )
