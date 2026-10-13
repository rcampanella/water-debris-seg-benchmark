#!/usr/bin/env python3
"""Agent 3 checklist: panoptic JSON + mask dirs present."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from water_debris_seg.datasets.lars import val_image_paths, verify_lars_layout


def main() -> int:
    layout = verify_lars_layout()
    n_val = len(val_image_paths(limit=None))
    print("lars_root", layout.root)
    print("splits", layout.splits)
    print("val_images", n_val)
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
