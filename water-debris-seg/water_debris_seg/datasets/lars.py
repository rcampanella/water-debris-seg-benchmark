"""LaRS v1.0.0 layout checks and light indexing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from water_debris_seg.paths import dataset_dir


@dataclass(frozen=True)
class LarsLayout:
    root: Path
    splits: tuple[str, ...]

    def panoptic_json(self, split: str) -> Path:
        return self.root / split / "panoptic_annotations.json"

    def images_dir(self, split: str) -> Path:
        return self.root / split / "images"

    def semantic_masks_dir(self, split: str) -> Path:
        return self.root / split / "semantic_masks"

    def panoptic_masks_dir(self, split: str) -> Path:
        return self.root / split / "panoptic_masks"


def default_lars_root() -> Path:
    return dataset_dir("lars")


def verify_lars_layout(root: Path | None = None) -> LarsLayout:
    root = root or default_lars_root()
    if root.is_symlink():
        root = root.resolve()
    splits = ("train", "val", "test")
    missing: list[str] = []
    for split in ("train", "val"):
        for rel in (
            f"{split}/panoptic_annotations.json",
            f"{split}/images",
            f"{split}/semantic_masks",
            f"{split}/panoptic_masks",
        ):
            if not (root / rel).exists():
                missing.append(rel)
    if not (root / "test" / "images").exists():
        missing.append("test/images")
    if missing:
        raise FileNotFoundError(f"LaRS layout incomplete under {root}: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    for split in ("train", "val"):
        with (root / split / "panoptic_annotations.json").open(encoding="utf-8") as f:
            coco = json.load(f)
        if "annotations" not in coco or "images" not in coco:
            raise ValueError(f"{split}/panoptic_annotations.json is not COCO panoptic")

    return LarsLayout(root=root, splits=splits)


def val_image_paths(root: Path | None = None, limit: int | None = None) -> list[Path]:
    layout = verify_lars_layout(root)
    img_dir = layout.images_dir("val")
    paths = sorted(img_dir.glob("*.*"))
    if limit is not None:
        paths = paths[:limit]
    return paths
