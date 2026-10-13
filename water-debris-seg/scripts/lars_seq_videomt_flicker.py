#!/usr/bin/env python3
"""Consecutive-frame mask stability for VidEoMT on LaRS val clips (flicker proxy).

Rasterizes per-track VIS masks from a VidEoMT ``results.json`` (full-clip inference) into
per-frame label maps, then reports the same proxy as ``lars_seq_eomt_flicker.py``:
``flicker = 1 - mean consecutive per-class mIoU`` (ignore label 255 = no prediction).

This is **not** GT ID tracking — LaRS has no cross-keyframe instance IDs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from inventory_lars_sequences import clip_id_from_name  # noqa: E402

try:
    from water_debris_seg.paths import scratch_root
except ModuleNotFoundError:

    def scratch_root() -> Path:
        import os

        return Path(f"/scratch-shared/{os.environ.get('USER', '')}/water-debris-seg")


IGNORE = 255


def mean_iou(a: np.ndarray, b: np.ndarray, ignore: int = IGNORE) -> float:
    valid = (a != ignore) & (b != ignore)
    if not valid.any():
        return float("nan")
    av, bv = a[valid], b[valid]
    ious = []
    for c in np.unique(av):
        if c == ignore:
            continue
        m1, m2 = av == c, bv == c
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        if union > 0:
            ious.append(inter / union)
    return float(sum(ious) / len(ious)) if ious else float("nan")


def category_map(categories: list[dict]) -> dict[int, int]:
    ids = sorted(c["id"] for c in categories)
    return {k: i for i, k in enumerate(ids)}


def frame_label_map(
    dets: list[dict],
    frame_idx: int,
    height: int,
    width: int,
    cat_map: dict[int, int],
) -> np.ndarray:
    label = np.full((height, width), IGNORE, dtype=np.int32)
    for det in sorted(dets, key=lambda x: -float(x.get("score", 0))):
        segs = det.get("segmentations") or []
        if frame_idx >= len(segs):
            continue
        seg = segs[frame_idx]
        if not seg:
            continue
        if isinstance(seg, dict):
            if isinstance(seg.get("counts"), list):
                seg = mask_util.frPyObjects(seg, height, width)
            mask = mask_util.decode(seg)
        else:
            continue
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = mask_util.decode(mask_util.encode(np.asfortranarray(mask.astype(np.uint8))))
        cid = cat_map.get(int(det["category_id"]), IGNORE)
        if cid == IGNORE:
            continue
        label[mask.astype(bool)] = cid
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ytvis-json",
        type=Path,
        default=scratch_root() / "data/lars/ytvis/lars_ytvis_val.json",
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reference-clips",
        type=Path,
        default=None,
        help="EoMT flicker JSON; restrict to the same clip_id set when provided",
    )
    args = parser.parse_args()

    gt = json.loads(args.ytvis_json.read_text(encoding="utf-8"))
    cat_map = category_map(gt["categories"])
    results = json.loads(args.results.read_text(encoding="utf-8"))
    by_video: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_video[int(r["video_id"])].append(r)

    ref_clips: set[str] | None = None
    if args.reference_clips and args.reference_clips.is_file():
        ref = json.loads(args.reference_clips.read_text(encoding="utf-8"))
        ref_clips = {c["clip_id"] for c in ref.get("clips", [])}

    clip_results = []
    all_flicker = []

    for video in gt["videos"]:
        vid = int(video["id"])
        fn0 = video["file_names"][0]
        clip_id = clip_id_from_name(fn0)
        if clip_id is None or clip_id == "_unparsed":
            continue
        if ref_clips is not None and clip_id not in ref_clips:
            continue
        h, w = int(video["height"]), int(video["width"])
        length = int(video["length"])
        dets = by_video.get(vid, [])
        if length < 2:
            continue
        ious = []
        prev: np.ndarray | None = None
        for t in range(length):
            cur = frame_label_map(dets, t, h, w, cat_map)
            if prev is not None:
                ious.append(mean_iou(prev, cur))
            prev = cur
        valid = [x for x in ious if x == x]
        flicker = 1.0 - (sum(valid) / len(valid)) if valid else float("nan")
        row = {
            "clip_id": clip_id,
            "video_id": vid,
            "n_frames": length,
            "n_tracks_pred": len(dets),
            "mean_pairwise_iou": sum(valid) / len(valid) if valid else None,
            "flicker_proxy": flicker,
        }
        clip_results.append(row)
        if valid:
            all_flicker.append(flicker)

    summary = {
        "n_clips": len(clip_results),
        "mean_flicker_proxy": sum(all_flicker) / len(all_flicker) if all_flicker else None,
        "clips": clip_results,
        "results_json": str(args.results),
        "ytvis_json": str(args.ytvis_json),
        "note": "VidEoMT thing-class flicker = 1 - mean consecutive per-class mIoU (ignore=255)",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n_clips", "mean_flicker_proxy")}, indent=2))


if __name__ == "__main__":
    main()
