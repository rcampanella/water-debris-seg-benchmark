#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from water_debris_seg.datasets.cansurf import verify_cansurf_layout


def main() -> int:
    layout = verify_cansurf_layout()
    n_tr = sum(1 for _ in layout.train_images.iterdir() if _.is_file())
    n_va = sum(1 for _ in layout.val_images.iterdir() if _.is_file())
    print("cansurf_root", layout.root)
    print("train_images", n_tr)
    print("val_images", n_va)
    print("data_yaml", layout.data_yaml)
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
