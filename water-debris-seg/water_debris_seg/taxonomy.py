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

# Bridge LaRS panoptic category_id -> COCO panoptic category_id (GT-side, legacy ablation 01 path).
# Ids corrected 2026-10-13 against configs/panoptic_coco_categories.json (cocodataset/panopticapi);
# the previous ids pointed at spoon/knife/food-other/paper (see decision log). Superseded for
# ablation 01 by prediction-side remapping (scripts/eval_eomt_lars_panoptic_pq.py,
# configs/coco133_to_lars_pred_map.json), because a one-to-one GT bridge penalises sibling
# COCO classes (e.g. sea vs river). Kept only so the legacy dataset adapter is not wrong.
LARS_TO_COCO_PANOPTIC_CATEGORY: dict[int, int] = {
    1: 175,   # Static Obstacle -> wall-stone (stuff)
    3: 148,   # Water -> river (stuff)
    5: 187,   # Sky -> sky-other-merged (stuff)
    11: 9,    # Boat/ship -> boat
    12: 9,    # Row boats -> boat
    13: 42,   # Paddle board -> surfboard
    14: 42,   # Buoy -> surfboard (proxy)
    15: 1,    # Swimmer -> person
    16: 16,   # Animal -> bird (proxy)
    17: 42,   # Float -> surfboard (proxy)
    19: 42,   # Other -> surfboard (proxy)
}
