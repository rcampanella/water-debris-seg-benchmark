#!/usr/bin/env python3
"""Load ``lars_ytvis_{split}`` via VidEoMT's YTVIS Detectron2 catalog (no training)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(f"/scratch-shared/{os.environ.get('USER', '')}/water-debris-seg")
VID = SCRATCH / "third_party/videomt"
sys.path.insert(0, str(VID))
sys.path.insert(0, str(ROOT))

import copy
import random

from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from videomt.config import add_videomt_config  # noqa: E402
from videomt.data_video.dataset_mapper import YTVISDatasetMapper  # noqa: E402

from water_debris_seg.videomt.lars_ytvis import register_lars_ytvis_splits  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/ablation07_videomt_lars_vitb.yaml",
    )
    args = parser.parse_args()

    cfg = get_cfg()
    add_videomt_config(cfg)
    cfg.merge_from_file(str(args.config))
    n_cls = cfg.MODEL.BACKBONE.NUM_CLASSES
    sem_cls = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    if n_cls != sem_cls:
        print(f"FAIL: BACKBONE.NUM_CLASSES={n_cls} != SEM_SEG_HEAD.NUM_CLASSES={sem_cls}")
        return 1

    register_lars_ytvis_splits(splits=(args.split,))
    name = f"lars_ytvis_{args.split}"
    records = DatasetCatalog.get(name)
    print(f"Loaded {len(records)} videos from {name}")
    if not records:
        print("FAIL: empty dataset")
        return 1
    rec = records[0]
    print(
        f"sample video_id={rec.get('video_id')} length={rec.get('length')} "
        f"frames_with_anns={sum(1 for fr in rec.get('annotations', []) if fr)}"
    )

    mapper = YTVISDatasetMapper(cfg, is_train=True, is_tgt=True, src_dataset_name=name)
    random.seed(0)
    for _ in range(min(50, len(records))):
        mapped = mapper(copy.deepcopy(random.choice(records)))
        for inst in mapped["instances"]:
            active = inst.gt_ids != -1
            if not active.any():
                continue
            labels = inst.gt_classes[active]
            bad = labels[(labels < 0) | (labels >= n_cls)]
            if bad.numel():
                print(f"FAIL: invalid gt_classes {bad.tolist()} (num_classes={n_cls})")
                return 1
    print(f"Mapper smoke OK ({n_cls} classes, 50 random train samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
