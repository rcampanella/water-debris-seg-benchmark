#!/usr/bin/env python3
"""Re-score a VidEoMT ``results.json`` on LaRS YTVIS per labeled keyframe (and with upstream video IoU).

    python scripts/rescore_videomt_ytvis_keyframes.py --results $RUN/inference/results.json
    python scripts/rescore_videomt_ytvis_keyframes.py --oracle exact   # sanity: AP ~1
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(f"/scratch-shared/{os.environ.get('USER', '')}/water-debris-seg")
sys.path.insert(0, str(SCRATCH / "third_party/videomt"))
sys.path.insert(0, str(ROOT))

from videomt.data_video.datasets.ytvis_api.ytvos import YTVOS  # noqa: E402
from videomt.data_video.datasets.ytvis_api.ytvoseval import YTVOSeval  # noqa: E402

from water_debris_seg.videomt.keyframe_patch import split_ytvis_by_keyframe, ytvos_from_dict  # noqa: E402

STAT_NAMES = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10")


def oracle_results(gt: dict, copy_all_frames: bool) -> list[dict]:
    """GT tracks as predictions (score 1); optionally the labeled mask copied to every frame."""
    out = []
    for ann in gt["annotations"]:
        segs = ann["segmentations"]
        if copy_all_frames:
            segs = [next(s for s in segs if s)] * len(segs)
        out.append({"video_id": ann["video_id"], "category_id": ann["category_id"],
                    "score": 1.0, "segmentations": segs})
    return out


def score(coco_gt, results: list[dict]) -> dict[str, float]:
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(copy.deepcopy(results)) if results else None
        if coco_dt is None:
            return {k: 0.0 for k in STAT_NAMES}
        ev = YTVOSeval(coco_gt, coco_dt)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {k: round(float(v), 4) for k, v in zip(STAT_NAMES, ev.stats)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=SCRATCH / "data/lars/ytvis/lars_ytvis_val.json")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--oracle", choices=("exact", "copy-all-frames"),
                        help="sanity check: score GT itself (exact -> AP 1 expected)")
    parser.add_argument("--out", type=Path, help="optional JSON output")
    args = parser.parse_args()

    if args.oracle:
        results = oracle_results(json.loads(args.gt.read_text()), args.oracle == "copy-all-frames")
    elif args.results:
        results = json.loads(args.results.read_text())
    else:
        parser.error("--results or --oracle required")

    gt = json.loads(args.gt.read_text())
    with contextlib.redirect_stdout(io.StringIO()):
        upstream = score(YTVOS(str(args.gt)), results)
    split_gt, split_preds = split_ytvis_by_keyframe(gt, results)
    keyframe = score(ytvos_from_dict(split_gt), split_preds)
    report = {"results": str(args.results or f"oracle:{args.oracle}"), "n_preds": len(results),
              "upstream_all_frames": upstream, "per_keyframe": keyframe}
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
