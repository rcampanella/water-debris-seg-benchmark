"""Unified class hooks (expand during Agent 7)."""

from __future__ import annotations

# LaRS semantic supercategories (PNG IDs)
LARS_SEMANTIC = {
    0: "obstacle",
    1: "water",
    2: "sky",
    255: "ignore",
}

# LaRS panoptic category IDs (subset for USV headline)
LARS_PANOPTIC_THINGS = {11, 12, 13, 14, 15, 16, 17, 19}
LARS_PANOPTIC_STUFF = {1, 3, 5}

UNIFIED_CLASSES = ("debris", "dynamic_obstacle", "static_context", "water", "sky")

# Bridge LaRS panoptic category_id -> COCO panoptic category_id (ablation 01 zero-shot).
# Chosen from IDs present in EoMT `datasets.coco_panoptic.CLASS_MAPPING`.
LARS_TO_COCO_PANOPTIC_CATEGORY: dict[int, int] = {
    1: 199,   # Static Obstacle -> wall-stone (COCO cat id)
    3: 196,   # Water -> river (stuff)
    5: 195,   # Sky -> sky-other-merged (stuff)
    11: 50,   # Boat/ship -> boat
    12: 50,
    13: 50,
    14: 49,   # Buoy -> surfboard (proxy)
    15: 1,    # Swimmer -> person
    16: 16,   # Animal -> bird (proxy)
    17: 49,
    19: 49,
}
