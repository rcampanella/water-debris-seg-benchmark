#!/usr/bin/env python3
"""Drop COCO images (and their anns) listed one filename per line in a blocklist file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-coco", type=Path, required=True)
    parser.add_argument("--blocklist", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    skip = {
        line.strip()
        for line in args.blocklist.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    gt = json.loads(args.gt_coco.read_text(encoding="utf-8"))
    keep_ids = {im["id"] for im in gt["images"] if im["file_name"] not in skip}
    gt["images"] = [im for im in gt["images"] if im["id"] in keep_ids]
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in keep_ids]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(gt), encoding="utf-8")
    print(
        f"wrote {args.out}: {len(gt['images'])} images, "
        f"{len(gt['annotations'])} anns (dropped {len(skip)} blocklisted names)"
    )


if __name__ == "__main__":
    main()
