#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from water_debris_seg.datasets.flow import verify_flow_layout
from water_debris_seg.paths import scratch_root


def main() -> int:
    root = scratch_root() / "data/flow"
    marker = root / ".flow_img_extracted"
    layout = verify_flow_layout(root, split="test")
    n_img = sum(
        1
        for p in layout.images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print("flow_root", layout.root)
    print("split", layout.split)
    print("images_dir", layout.images_dir)
    if layout.labels_dir:
        print("labels_dir", layout.labels_dir, "(YOLO)")
    else:
        print("annotations_dir", layout.annotations_dir, "(VOC)")
    print("n_test_images", n_img)
    print("marker", marker, "present" if marker.is_file() else "missing (optional)")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
