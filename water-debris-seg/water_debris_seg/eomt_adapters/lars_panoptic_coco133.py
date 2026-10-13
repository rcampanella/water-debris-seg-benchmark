"""LaRS panoptic with GT mapped to COCO 133-class space (zero-shot eval with COCO head)."""

from __future__ import annotations

from datasets.coco_panoptic import CLASS_MAPPING  # type: ignore[import-untyped]

from water_debris_seg.eomt_adapters.lars_panoptic import LarsPanoptic
from water_debris_seg.taxonomy import LARS_TO_COCO_PANOPTIC_CATEGORY

# COCO panoptic stuff indices used by EoMT config (same as coco_panoptic yaml).
COCO133_STUFF_CLASSES = list(range(80, 133))


class LarsPanopticCoco133(LarsPanoptic):
    """LaRS zip layout; segment labels remapped for COCO-pretrained class head."""

    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 133,
        stuff_classes: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            path=path,
            num_workers=num_workers,
            batch_size=batch_size,
            img_size=img_size,
            num_classes=num_classes,
            stuff_classes=stuff_classes or COCO133_STUFF_CLASSES,
            **kwargs,
        )

    @staticmethod
    def target_parser(target, labels_by_id, is_crowd_by_id, **kwargs):
        target = target[0, :, :] + target[1, :, :] * 256 + target[2, :, :] * 256**2
        masks, labels, is_crowd = [], [], []
        for label_id in target.unique():
            if label_id.item() not in labels_by_id:
                continue
            lars_cat = labels_by_id[label_id.item()]
            coco_cat = LARS_TO_COCO_PANOPTIC_CATEGORY.get(lars_cat)
            if coco_cat is None or coco_cat not in CLASS_MAPPING:
                continue
            masks.append(target == label_id)
            labels.append(CLASS_MAPPING[coco_cat])
            is_crowd.append(is_crowd_by_id[label_id.item()])
        return masks, labels, is_crowd
