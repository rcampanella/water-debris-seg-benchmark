#!/usr/bin/env python3
"""Held-out YOLO splits for FloW and CANSURF (fairness fix 3), on home.

Layout per dataset under data/water_debris/yolo_splits/<dataset>/:
  train90/  held10/  train_all/  test/      each with images/ (symlinks to the home images) and labels/ (copies)
  stageA.yaml (train90 -> held10)  stageB.yaml (train_all, held10 monitored)  eval_test.yaml (test only)

FloW:    pool = FloW training/ (1,200); held10 = ablation 08's seeded 120-image slice (flow_val_split.txt), for
         comparability; test = official test (800).
CANSURF: pool = official train (4,950); held10 = 10% of *sequences* (seed 0). Roboflow made several augmented copies
         of each frame and frames come in sequences, so the split groups by the file stem with Roboflow's suffix and the
         trailing frame number removed. test = official val (2,221).

Writes the split lists and statistics to eval_outputs/water_debris_seg/fairness_fixes/sources/.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOME_DATA = REPO / "data/water_debris"
OUT_SRC = REPO / "eval_outputs/water_debris_seg/fairness_fixes/sources"
ROBOFLOW = re.compile(r"_(jpg|jpeg|png)\.rf\..*$", re.I)


def cansurf_group(name: str) -> str:
    return re.sub(r"[_-]\d+$", "", ROBOFLOW.sub("", Path(name).stem))


def link_split(dst: Path, pairs: list[tuple[Path, Path]]) -> int:
    """pairs: (image, label). Images are symlinked, labels copied (Ultralytics maps /images/ -> /labels/)."""
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir(parents=True)
    n_boxes = 0
    for img, lab in pairs:
        os.symlink(img, dst / "images" / img.name)
        if lab.exists():
            shutil.copy2(lab, dst / "labels" / lab.name)
            n_boxes += sum(1 for line in lab.read_text().splitlines() if line.strip())
        else:
            (dst / "labels" / f"{img.stem}.txt").write_text("")
    return n_boxes


def pairs_from(img_dir: Path, lab_dir: Path, names=None) -> list[tuple[Path, Path]]:
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if names is not None:
        imgs = [p for p in imgs if p.stem in names or p.name in names]
    return [(p.resolve(), lab_dir / f"{p.stem}.txt") for p in imgs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["flow", "cansurf"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = HOME_DATA / "yolo_splits" / args.dataset
    OUT_SRC.mkdir(parents=True, exist_ok=True)

    if args.dataset == "flow":
        y = HOME_DATA / "flow/yolo"           # ablation 08's layout (train 1,080 / val 120 / test 800), home copy
        held_names = set((y / "val_split.txt").read_text().split())
        pool = pairs_from(y / "train/images", y / "train/labels") + pairs_from(y / "val/images", y / "val/labels")
        held = [p for p in pool if p[0].stem in held_names]
        train90 = [p for p in pool if p[0].stem not in held_names]
        test = pairs_from(y / "test/images", y / "test/labels")
        names = ["waste"]
        rule = "ablation 08 seeded held-out list (flow_val_split.txt)"
    else:
        c = HOME_DATA / "cansurf_home"
        pool = pairs_from(c / "train/images", c / "train/labels")
        groups = sorted({cansurf_group(p[0].name) for p in pool})
        rng = random.Random(args.seed)
        rng.shuffle(groups)
        held_groups = set(groups[: round(0.10 * len(groups))])
        held = [p for p in pool if cansurf_group(p[0].name) in held_groups]
        train90 = [p for p in pool if cansurf_group(p[0].name) not in held_groups]
        test = pairs_from(c / "val/images", c / "val/labels")
        names = ["debris"]
        rule = f"10% of {len(groups)} sequence groups (Roboflow suffix and trailing frame number removed), seed {args.seed}"
        (OUT_SRC / "cansurf_held10_groups.txt").write_text("\n".join(sorted(held_groups)) + "\n")

    stats = {"dataset": args.dataset, "held_out_rule": rule, "root": str(root), "splits": {}}
    for split, pairs in [("train90", train90), ("held10", held), ("train_all", train90 + held), ("test", test)]:
        stats["splits"][split] = {"images": len(pairs), "boxes": link_split(root / split, pairs)}
    (OUT_SRC / f"{args.dataset}_held10_images.txt").write_text("\n".join(sorted(p[0].name for p in held)) + "\n")
    for fname, train, val in [("stageA.yaml", "train90", "held10"), ("stageB.yaml", "train_all", "held10"),
                              ("eval_test.yaml", "test", "test")]:
        (root / fname).write_text(f"path: {root}\ntrain: {train}/images\nval: {val}/images\nnc: 1\nnames: {names}\n")
    (OUT_SRC / f"{args.dataset}_yolo_split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
