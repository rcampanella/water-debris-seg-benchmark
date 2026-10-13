#!/usr/bin/env python3
"""Summarize LaRS val image_annotations.json for stratified eval planning."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from water_debris_seg.paths import scratch_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="val",
        choices=("train", "val", "test"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ann_path = scratch_root() / f"data/lars/{args.split}/image_annotations.json"
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    rows = data.get("annotations", data if isinstance(data, list) else [])

    counters: dict[str, Counter] = {}
    special_flags: Counter[str] = Counter()

    for row in rows:
        labels = row.get("labels", {})
        for key in ("scene_type", "lighting", "reflections", "waves"):
            val = labels.get(key)
            if val is None:
                continue
            counters.setdefault(key, Counter())[str(val)] += 1
        special = labels.get("special") or {}
        if isinstance(special, dict):
            for k, v in special.items():
                if v:
                    special_flags[k] += 1

    summary = {
        "split": args.split,
        "n_images": len(rows),
        "attributes": {k: dict(v) for k, v in counters.items()},
        "special_true_counts": dict(special_flags),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
