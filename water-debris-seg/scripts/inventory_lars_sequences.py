#!/usr/bin/env python3
"""Summarize LaRS images_seq clips (flat JPG names) for VidEoMT / ablation 07 planning."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from water_debris_seg.paths import scratch_root

CLIP_PATTERNS = (
    re.compile(r"^(?P<prefix>.+)_seq_(?P<id>\d+)_(?P<frame>\d+)\.jpg$", re.I),
    re.compile(r"^mastr_(?P<id>\d+)_(?P<frame>\d+)\.jpg$", re.I),
)


def clip_id_from_name(name: str) -> str | None:
    for pat in CLIP_PATTERNS:
        m = pat.match(name)
        if not m:
            continue
        if "prefix" in m.groupdict() and m.group("prefix") is not None:
            return f"{m.group('prefix')}_seq_{m.group('id')}"
        return f"mastr_{m.group('id')}"
    return None


def clips_in_dir(seq_dir: Path) -> dict[str, list[str]]:
    by_clip: dict[str, list[str]] = defaultdict(list)
    for p in sorted(seq_dir.glob("*.jpg")):
        clip_id = clip_id_from_name(p.name)
        if clip_id is None:
            by_clip["_unparsed"].append(p.name)
            continue
        by_clip[clip_id].append(p.name)
    return dict(by_clip)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=scratch_root() / "data/lars",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary: dict = {"splits": {}, "total_clips": 0, "total_frames": 0}
    for split in ("train", "val", "test"):
        seq_dir = args.root / split / "images_seq"
        if not seq_dir.is_dir():
            continue
        clips = clips_in_dir(seq_dir)
        n_frames = sum(len(v) for k, v in clips.items() if k != "_unparsed")
        summary["splits"][split] = {
            "seq_dir": str(seq_dir),
            "n_clips": len([k for k in clips if k != "_unparsed"]),
            "n_frames": n_frames,
            "n_unparsed": len(clips.get("_unparsed", [])),
            "clip_frame_counts_sample": sorted(
                (len(v), k) for k, v in clips.items() if k != "_unparsed"
            )[:5],
        }
        summary["total_clips"] += summary["splits"][split]["n_clips"]
        summary["total_frames"] += n_frames

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
