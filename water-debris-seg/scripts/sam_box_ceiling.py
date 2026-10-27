#!/usr/bin/env python3
"""Box AP of the SAM pseudo-mask boxes against the GT boxes they were prompted with (fairness fix 3).

Each SAM mask's min/max box is scored as a detection (score 1) against the GT boxes of the training pool. The result is
the box AP of a model that reproduces the SAM masks exactly: an approximate ceiling for EoMT trained on SAM masks and
scored with mask-derived boxes against the (looser) GT boxes. Not strict: the test split has no SAM masks, and a model
can learn boxes looser than SAM's.

Output: eval_outputs/water_debris_seg/fairness_fixes/sources/sam_box_ceiling_ap.json
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MASKS = REPO / "data/water_debris/sam_masks"
OUT = REPO / "eval_outputs/water_debris_seg/fairness_fixes/sources/sam_box_ceiling_ap.json"
KEYS = ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")


def main() -> None:
    from pycocotools import mask as cm
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    out = {}
    for ds in ("flow", "cansurf"):
        src = MASKS / ds / "instances_train_all.json"
        doc = json.loads(src.read_text())
        gt = copy.deepcopy(doc)
        for a in gt["annotations"]:
            a.pop("segmentation")
            a["area"] = a["bbox"][2] * a["bbox"][3]
        dets = []
        for a in doc["annotations"]:
            rle = dict(a["segmentation"], counts=a["segmentation"]["counts"].encode())
            dets.append({"image_id": a["image_id"], "category_id": 1, "bbox": [float(v) for v in cm.toBbox(rle)],
                         "score": 1.0})
        with contextlib.redirect_stdout(io.StringIO()):
            g = COCO()
            g.dataset = gt
            g.createIndex()
            ev = COCOeval(g, g.loadRes(dets), "bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        out[ds] = {"source": str(src.relative_to(REPO)), "n_images": len(doc["images"]), "n_boxes": len(dets),
                   **{k: float(v) for k, v in zip(KEYS, ev.stats)}}
        print(ds, {k: round(out[ds][k], 3) for k in KEYS})
    out["note"] = __doc__.split("\n\n")[1].replace("\n", " ")
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
