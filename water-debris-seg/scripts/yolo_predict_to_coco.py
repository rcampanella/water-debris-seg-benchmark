"""Run an Ultralytics detector over a COCO GT image list and write COCO-style predictions.

Image ids come from the GT JSON (matched by file_name), so the output scores with
eval_coco_bbox.py exactly like the EoMT mask→box predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--gt-coco", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--category-id", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    from ultralytics import YOLO

    gt = json.loads(args.gt_coco.read_text())
    entries = [(im["id"], args.images / im["file_name"]) for im in gt["images"]]
    missing = [p for _, p in entries if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} GT images missing under {args.images}, e.g. {missing[:3]}")

    model = YOLO(str(args.weights))
    preds: list[dict] = []
    for i in range(0, len(entries), args.batch):
        chunk = entries[i : i + args.batch]
        results = model.predict(
            [str(p) for _, p in chunk],
            imgsz=args.imgsz,
            conf=args.conf,
            max_det=args.max_det,
            verbose=False,
        )
        for (image_id, _), res in zip(chunk, results):
            for (x1, y1, x2, y2), score in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
                preds.append(
                    {
                        "image_id": image_id,
                        "category_id": args.category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(preds))
    print(f"{len(preds)} detections on {len(entries)} images -> {args.out}")


if __name__ == "__main__":
    main()
