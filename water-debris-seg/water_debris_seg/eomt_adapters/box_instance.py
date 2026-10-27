"""Single-class instance datamodule for EoMT on FloW / CANSURF SAM pseudo-masks (fairness fix 3).

Same zip layout as EoMT's COCOInstance (train2017.zip, val2017.zip, annotations_trainval2017.zip with
annotations/instances_{train,val}2017.json); built by scripts/pack_box_eomt_zips.py. Differences from COCOInstance:
one class (category id 1 -> class 0) and masks stored as compressed RLE (decoded directly).
"""

from __future__ import annotations

import torch
from pycocotools import mask as coco_mask
from torchvision import tv_tensors

from datasets.coco_instance import COCOInstance  # type: ignore[import-untyped]


class BoxSetInstance(COCOInstance):
    def __init__(self, path, num_workers: int = 4, batch_size: int = 4, img_size: tuple[int, int] = (640, 640),
                 num_classes: int = 1, color_jitter_enabled=False, scale_range=(0.1, 2.0), check_empty_targets=True) -> None:
        super().__init__(path=path, num_workers=num_workers, batch_size=batch_size, img_size=img_size,
                         num_classes=num_classes, color_jitter_enabled=color_jitter_enabled, scale_range=scale_range,
                         check_empty_targets=check_empty_targets)

    @staticmethod
    def target_parser(polygons_by_id, labels_by_id, is_crowd_by_id, width, height, **kwargs):
        masks, labels, is_crowd = [], [], []
        for label_id, cls_id in labels_by_id.items():
            if cls_id != 1:
                continue
            seg = polygons_by_id[label_id]
            if isinstance(seg, dict):
                rle = {"size": seg["size"], "counts": seg["counts"].encode() if isinstance(seg["counts"], str) else seg["counts"]}
                if not isinstance(seg["counts"], (str, bytes)):
                    rle = coco_mask.frPyObjects(seg, height, width)
            else:
                rles = coco_mask.frPyObjects(seg, height, width)
                rle = coco_mask.merge(rles) if isinstance(rles, list) else rles
            masks.append(tv_tensors.Mask(coco_mask.decode(rle), dtype=torch.bool))
            labels.append(0)
            is_crowd.append(is_crowd_by_id[label_id])
        return masks, labels, is_crowd
