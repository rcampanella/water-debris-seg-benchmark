"""CANSURF raw YOLO layout (Zenodo v2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from water_debris_seg.paths import dataset_dir


@dataclass(frozen=True)
class CansurfLayout:
    root: Path
    train_images: Path
    val_images: Path
    train_labels: Path
    val_labels: Path
    data_yaml: Path


def default_cansurf_root() -> Path:
    return dataset_dir("cansurf")


def verify_cansurf_layout(root: Path | None = None) -> CansurfLayout:
    root = root or default_cansurf_root()
    if root.is_symlink():
        root = root.resolve()
    layout = CansurfLayout(
        root=root,
        train_images=root / "train" / "images",
        val_images=root / "val" / "images",
        train_labels=root / "train" / "labels",
        val_labels=root / "val" / "labels",
        data_yaml=root / "data.yaml",
    )
    missing = [p for p in (
        layout.train_images, layout.val_images,
        layout.train_labels, layout.val_labels,
        layout.data_yaml,
    ) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"CANSURF incomplete under {root}: {missing}")
    n_train = sum(1 for _ in layout.train_images.iterdir() if _.is_file())
    n_val = sum(1 for _ in layout.val_images.iterdir() if _.is_file())
    if n_train < 100 or n_val < 100:
        raise ValueError(f"CANSURF image counts suspicious: train={n_train} val={n_val}")
    return layout
