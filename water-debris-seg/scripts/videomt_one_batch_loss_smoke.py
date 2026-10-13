#!/usr/bin/env python3
"""One-batch VidEoMT online loss check on LaRS YTVIS (no Trainer, no W&B).

Reports (a) fraction of sampled training windows that contain a labeled keyframe and
(b) per-batch losses from a single forward pass. Run before any Slurm retrain: with the
keyframe patches, ``loss_mask`` / ``loss_dice`` must be > 0.

    python scripts/videomt_one_batch_loss_smoke.py --device cpu --batches 3
    python scripts/videomt_one_batch_loss_smoke.py --no-patch   # upstream behaviour
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(f"/scratch-shared/{os.environ.get('USER', '')}/water-debris-seg")
VID = SCRATCH / "third_party/videomt"
sys.path.insert(0, str(VID))
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from videomt.config import add_videomt_config  # noqa: E402
from videomt.data_video.dataset_mapper import YTVISDatasetMapper  # noqa: E402

from water_debris_seg.videomt.keyframe_patch import (  # noqa: E402
    apply_keyframe_sampling_patch,
    apply_labeled_frames_loss_patch,
)
from water_debris_seg.videomt.lars_ytvis import register_lars_ytvis_splits  # noqa: E402
from water_debris_seg.videomt.prepare_targets_patch import apply_prepare_targets_patch  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ablation07_videomt_lars_vitb.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--window-trials", type=int, default=500)
    parser.add_argument("--no-patch", action="store_true", help="keyframe patches off (upstream sampling/loss)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    apply_prepare_targets_patch()
    if not args.no_patch:
        apply_keyframe_sampling_patch()
        apply_labeled_frames_loss_patch()

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_videomt_config(cfg)
    cfg.merge_from_file(str(args.config))
    cfg.MODEL.DEVICE = args.device
    cfg.freeze()

    register_lars_ytvis_splits(splits=("train",))
    # Mirror D2's train-time filter: drop videos without any labeled frame.
    records = [r for r in DatasetCatalog.get("lars_ytvis_train") if any(r["annotations"])]
    mapper = YTVISDatasetMapper(cfg, is_train=True, is_tgt=True, src_dataset_name="lars_ytvis_train")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # (a) window coverage: any active (gt_ids != -1) instance in the sampled clip
    hits = 0
    for _ in range(args.window_trials):
        rec = copy.deepcopy(random.choice(records))
        L, F = rec["length"], mapper.sampling_frame_num
        idx = mapper.select_frames(L) if args.no_patch else None
        if idx is None:
            labeled = [i for i, fr in enumerate(rec["annotations"]) if fr]
            from water_debris_seg.videomt.keyframe_patch import _labeled_window

            idx = _labeled_window(L, labeled, F) if labeled and L >= F else mapper.select_frames(L)
        hits += any(rec["annotations"][i] for i in idx)
    print(f"windows with a labeled frame: {hits}/{args.window_trials} = {hits / args.window_trials:.3f}")

    # (b) one-batch losses
    model = build_model(cfg)
    model.train()
    n_with_gt, ok = 0, True
    bs = cfg.SOLVER.IMS_PER_BATCH
    for b in range(args.batches):
        batch = [mapper(copy.deepcopy(random.choice(records))) for _ in range(bs)]
        n_active = sum(int((inst.gt_ids != -1).sum()) for v in batch for inst in v["instances"])
        with torch.no_grad():
            losses = model(batch)
        main_l = {k: round(float(v), 4) for k, v in losses.items() if not k[-1].isdigit()}
        print(f"batch {b}: active (frame,track) GT = {n_active}; {main_l}")
        if n_active:
            n_with_gt += 1
            ok &= float(losses["loss_mask"]) > 0 and float(losses["loss_dice"]) > 0
    ok &= n_with_gt * 2 >= args.batches
    print(f"batches with GT: {n_with_gt}/{args.batches}")
    print("PASS: mask/dice loss > 0 whenever GT present" if ok else "FAIL: mask supervision missing")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
