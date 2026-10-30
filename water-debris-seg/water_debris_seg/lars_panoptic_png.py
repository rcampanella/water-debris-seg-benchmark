"""Write LaRS panoptic predictions in the official evaluator's PNG format (lojzezust/lars_evaluator).

Format (same as the LaRS GT panoptic masks): R = LaRS category id (1 Static Obstacle, 3 Water, 5 Sky, 11-19 objects),
G * 256 + B = instance id, (0, 0, 0) = void. Input is the (H, W, 2) tensor our scorers build: channel 0 = LaRS 11-class
index in LARS_NAMES order (11 = void), channel 1 = segment id.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# LaRS 11-class index (Static Obstacle, Water, Sky, Boat/ship, Row boats, Paddle board, Buoy, Swimmer, Animal, Float,
# Other) -> LaRS panoptic category id (panoptic_annotations.json)
LARS_CATEGORY_IDS = [1, 3, 5, 11, 12, 13, 14, 15, 16, 17, 19]
NUM_LARS = 11


def to_lars_rgb(preds) -> np.ndarray:
    cls = np.asarray(preds[..., 0].cpu() if hasattr(preds, "cpu") else preds[..., 0]).astype(np.int64)
    seg = np.asarray(preds[..., 1].cpu() if hasattr(preds, "cpu") else preds[..., 1]).astype(np.int64)
    rgb = np.zeros(cls.shape + (3,), np.uint8)
    valid = (cls >= 0) & (cls < NUM_LARS)
    lut = np.array(LARS_CATEGORY_IDS + [0], np.uint8)
    rgb[..., 0] = lut[np.where(valid, cls, NUM_LARS)]
    # one instance id per (class, segment) pair, numbered 1..N per image
    keys = cls * (1 << 32) + (seg + 1)
    uniq, inv = np.unique(np.where(valid, keys, -1), return_inverse=True)
    inst = inv.reshape(cls.shape) + (1 if uniq[0] != -1 else 0)
    inst[~valid] = 0
    assert inst.max() < 65536
    rgb[..., 1] = inst // 256
    rgb[..., 2] = inst % 256
    return rgb


def write_lars_png(preds, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_lars_rgb(preds)).save(path)
