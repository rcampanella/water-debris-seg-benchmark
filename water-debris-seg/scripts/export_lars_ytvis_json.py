#!/usr/bin/env python3
"""Export LaRS ``images_seq`` clips + panoptic keyframes to YouTube-VIS JSON for VidEoMT."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from inventory_lars_sequences import clip_id_from_name  # noqa: E402
from water_debris_seg.taxonomy import LARS_PANOPTIC_THINGS  # noqa: E402
from water_debris_seg.videomt.lars_ytvis import DEFAULT_YTVIS_DIR  # noqa: E402


def rgb2id(color: np.ndarray) -> np.ndarray:
    if color.dtype != np.int32:
        color = color.astype(np.int32)
    return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]


def segment_rle(panoptic_rgb: np.ndarray, segment_id: int) -> dict | None:
    import pycocotools.mask as mask_util

    seg_map = rgb2id(panoptic_rgb)
    mask = (seg_map == segment_id).astype(np.uint8)
    if mask.sum() == 0:
        return None
    rle = mask_util.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def clips_for_split(lars_root: Path, split: str) -> dict[str, list[str]]:
    seq_dir = lars_root / split / "images_seq"
    by_clip: dict[str, list[str]] = defaultdict(list)
    for p in sorted(seq_dir.glob("*.jpg")):
        cid = clip_id_from_name(p.name)
        if cid:
            by_clip[cid].append(p.name)
    for cid in by_clip:
        by_clip[cid].sort()
    return dict(by_clip)


def export_split(
    lars_root: Path,
    split: str,
    out_json: Path,
    max_clips: int = 0,
) -> dict:
    pano_path = lars_root / split / "panoptic_annotations.json"
    pano = json.loads(pano_path.read_text(encoding="utf-8"))
    images_by_fn = {im["file_name"]: im for im in pano["images"]}
    ann_by_image_id = {a["image_id"]: a for a in pano["annotations"]}
    thing_cats = sorted(
        (c for c in pano["categories"] if c["id"] in LARS_PANOPTIC_THINGS),
        key=lambda c: c["id"],
    )

    by_clip = clips_for_split(lars_root, split)
    clip_ids = sorted(by_clip.keys())
    if max_clips > 0:
        clip_ids = clip_ids[:max_clips]

    videos = []
    annotations = []
    video_id = 1
    ann_id = 1
    mask_dir = lars_root / split / "panoptic_masks"
    png_cache: dict[str, np.ndarray] = {}

    def load_panoptic(file_name: str) -> np.ndarray:
        if file_name not in png_cache:
            png_path = mask_dir / file_name.replace(".jpg", ".png")
            png_cache[file_name] = np.array(Image.open(png_path).convert("RGB"))
        return png_cache[file_name]

    n_clips_with_labels = 0
    n_thing_instances = 0

    for clip_id in clip_ids:
        file_names = by_clip[clip_id]
        labeled_frames = [fn for fn in file_names if fn in images_by_fn]
        if not labeled_frames:
            continue
        n_clips_with_labels += 1
        ref = images_by_fn[labeled_frames[0]]
        videos.append(
            {
                "id": video_id,
                "file_names": file_names,
                "width": ref["width"],
                "height": ref["height"],
                "length": len(file_names),
            }
        )
        fn_to_idx = {fn: i for i, fn in enumerate(file_names)}

        for fn in labeled_frames:
            frame_idx = fn_to_idx[fn]
            im = images_by_fn[fn]
            pano_ann = ann_by_image_id[im["id"]]
            pano_rgb = load_panoptic(fn)
            length = len(file_names)
            for seg in pano_ann["segments_info"]:
                if seg["category_id"] not in LARS_PANOPTIC_THINGS:
                    continue
                rle = segment_rle(pano_rgb, seg["id"])
                if rle is None:
                    continue
                bbox = seg["bbox"]
                segmentations: list = [None] * length
                bboxes: list = [None] * length
                areas: list = [None] * length
                segmentations[frame_idx] = rle
                bboxes[frame_idx] = bbox
                areas[frame_idx] = int(seg["area"])
                annotations.append(
                    {
                        "id": ann_id,
                        "video_id": video_id,
                        "category_id": seg["category_id"],
                        "segmentations": segmentations,
                        "bboxes": bboxes,
                        "areas": areas,
                        "iscrowd": seg.get("iscrowd", 0),
                    }
                )
                ann_id += 1
                n_thing_instances += 1

        video_id += 1

    out = {
        "info": {
            "description": f"LaRS {split} sequences (panoptic things on keyframes)",
            "version": "1.0",
        },
        "categories": thing_cats,
        "videos": videos,
        "annotations": annotations,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out), encoding="utf-8")
    stats = {
        "split": split,
        "out_json": str(out_json),
        "n_videos": len(videos),
        "n_clips_with_labeled_keyframe": n_clips_with_labels,
        "n_thing_track_annotations": n_thing_instances,
        "n_thing_categories": len(thing_cats),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lars-root",
        type=Path,
        default=Path(f"/scratch-shared/{__import__('os').environ.get('USER', '')}/water-debris-seg/data/lars"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_YTVIS_DIR,
    )
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--max-clips", type=int, default=0, help="0 = all clips in split")
    args = parser.parse_args()

    splits = ("train", "val") if args.split == "both" else (args.split,)
    for split in splits:
        out_json = args.out_dir / f"lars_ytvis_{split}.json"
        stats = export_split(args.lars_root, split, out_json, max_clips=args.max_clips)
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
