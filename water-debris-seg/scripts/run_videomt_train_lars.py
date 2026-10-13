#!/usr/bin/env python3
"""Launch VidEoMT ``train_net_video.py`` with LaRS YTVIS registered + benchmark W&B."""

from __future__ import annotations

import os
import sys
from functools import wraps
from pathlib import Path


def _patch_wandb() -> None:
    import wandb

    _orig = wandb.init

    @wraps(_orig)
    def _init(*args, **kwargs):
        # Upstream train_net_video passes project="videomt"; benchmark uses water-debris-seg.
        kwargs["project"] = os.environ.get("WANDB_PROJECT", "water-debris-seg")
        if os.environ.get("WANDB_RUN_GROUP"):
            kwargs["group"] = os.environ["WANDB_RUN_GROUP"]
        if os.environ.get("WANDB_NAME"):
            kwargs["name"] = os.environ["WANDB_NAME"]
        kwargs.setdefault("job_type", "videomt_train")
        return _orig(*args, **kwargs)

    wandb.init = _init  # type: ignore[method-assign]


def main() -> None:
    scratch = Path(f"/scratch-shared/{os.environ.get('USER', '')}/water-debris-seg")
    vid = scratch / "third_party/videomt"
    wds = Path(__file__).resolve().parents[1]

    sys.path.insert(0, str(vid))
    sys.path.insert(0, str(wds))

    from water_debris_seg.videomt.lars_ytvis import register_lars_ytvis_splits
    from water_debris_seg.videomt.keyframe_patch import (
        apply_keyframe_eval_patch,
        apply_keyframe_sampling_patch,
        apply_labeled_frames_loss_patch,
    )
    from water_debris_seg.videomt.prepare_targets_patch import apply_prepare_targets_patch

    register_lars_ytvis_splits()
    apply_prepare_targets_patch()
    apply_keyframe_sampling_patch()
    apply_labeled_frames_loss_patch()
    apply_keyframe_eval_patch()

    _patch_wandb()

    from train_net_video import default_argument_parser, launch, main as train_main

    parser = default_argument_parser()
    args = parser.parse_args()
    if not args.dist_url:
        args.dist_url = "tcp://127.0.0.1:50263"
    launch(
        train_main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    main()
