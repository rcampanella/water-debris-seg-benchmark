#!/usr/bin/env python3
"""Ensure LaRS->COCO bridge IDs exist in EoMT CLASS_MAPPING."""
from __future__ import annotations

import sys
from pathlib import Path

EOMT = Path("/scratch-shared/rcampanella/water-debris-seg/third_party/eomt")
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(EOMT))

from datasets.coco_panoptic import CLASS_MAPPING  # type: ignore[import-untyped]
from water_debris_seg.taxonomy import LARS_TO_COCO_PANOPTIC_CATEGORY


def main() -> int:
    bad = []
    for lars, coco in LARS_TO_COCO_PANOPTIC_CATEGORY.items():
        ok = coco in CLASS_MAPPING
        print(lars, "->", coco, "ok" if ok else "MISSING")
        if not ok:
            bad.append((lars, coco))
    if bad:
        raise SystemExit(f"invalid bridge ids: {bad}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
