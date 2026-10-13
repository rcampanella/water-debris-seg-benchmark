"""Register LaRS clips exported as YouTube-VIS JSON for VidEoMT."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from water_debris_seg.paths import scratch_root
from water_debris_seg.taxonomy import LARS_PANOPTIC_THINGS

DEFAULT_YTVIS_DIR = scratch_root() / "data/lars/ytvis"


def lars_ytvis_json(split: str, root: Path | None = None) -> Path:
    root = root or DEFAULT_YTVIS_DIR
    return root / f"lars_ytvis_{split}.json"


def lars_ytvis_image_root(split: str, lars_root: Path | None = None) -> Path:
    lars_root = lars_root or scratch_root() / "data/lars"
    return lars_root / split / "images_seq"


def load_thing_categories(json_path: Path) -> list[dict[str, Any]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    cats = [c for c in data["categories"] if c["id"] in LARS_PANOPTIC_THINGS]
    return sorted(cats, key=lambda c: c["id"])


def lars_ytvis_metadata(json_path: Path) -> dict[str, Any]:
    cats = load_thing_categories(json_path)
    thing_ids = [c["id"] for c in cats]
    return {
        "thing_dataset_id_to_contiguous_id": {k: i for i, k in enumerate(thing_ids)},
        "thing_classes": [c["name"] for c in cats],
        "thing_colors": [c.get("color", [220, 20, 60]) for c in cats],
    }


def register_lars_ytvis_splits(
    splits: tuple[str, ...] = ("train", "val"),
    ytvis_dir: Path | None = None,
    lars_root: Path | None = None,
) -> None:
    """Register ``lars_ytvis_{split}`` with Detectron2 (VidEoMT YTVIS loader)."""
    from videomt.data_video.datasets.ytvis import register_ytvis_instances

    ytvis_dir = ytvis_dir or DEFAULT_YTVIS_DIR
    for split in splits:
        json_path = lars_ytvis_json(split, ytvis_dir)
        if not json_path.is_file():
            raise FileNotFoundError(json_path)
        name = f"lars_ytvis_{split}"
        register_ytvis_instances(
            name,
            lars_ytvis_metadata(json_path),
            os.fspath(json_path),
            os.fspath(lars_ytvis_image_root(split, lars_root)),
        )
