#!/usr/bin/env python3
"""EoMT zips (COCO instance layout) for FloW / CANSURF SAM pseudo-masks, held-out protocol (fairness fix 3).

  stage A: train2017 = train90, val2017 = held10        stage B: train2017 = train_all, val2017 = held10 (monitored)
Images come from data/water_debris/yolo_splits/<dataset>/<split>/images (the same splits as YOLOv11s), masks from
data/water_debris/sam_masks/<dataset>/instances_<split>.json. Output: data/water_debris/eomt_box_zips/<dataset>/stage<X>/.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPLITS = REPO / "data/water_debris/yolo_splits"
MASKS = REPO / "data/water_debris/sam_masks"
OUT = REPO / "data/water_debris/eomt_box_zips"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["flow", "cansurf"], required=True)
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    args = ap.parse_args()
    train_split = "train90" if args.stage == "A" else "train_all"
    out = OUT / args.dataset / f"stage{args.stage}"
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    with zipfile.ZipFile(out / "annotations_trainval2017.zip", "w", zipfile.ZIP_STORED) as az:
        for split, folder in ((train_split, "train2017"), ("held10", "val2017")):
            doc = json.loads((MASKS / args.dataset / f"instances_{split}.json").read_text())
            names = {im["file_name"] for im in doc["images"]}
            with zipfile.ZipFile(out / f"{folder}.zip", "w", zipfile.ZIP_STORED) as iz:
                for p in sorted((SPLITS / args.dataset / split / "images").iterdir()):
                    if p.name in names:
                        iz.write(p.resolve(), f"{folder}/{p.name}")
            az.writestr(f"annotations/instances_{folder}.json", json.dumps(doc))
            stats[folder] = {"split": split, "images": len(names), "masks": len(doc["annotations"])}
    (out / "pack_stats.json").write_text(json.dumps(stats, indent=2))
    print(out, json.dumps(stats))


if __name__ == "__main__":
    main()
