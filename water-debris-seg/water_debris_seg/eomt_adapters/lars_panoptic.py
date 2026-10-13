"""LaRS panoptic datamodule for EoMT (COCO zip layout, native 11-class PQ)."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from torch.utils.data import DataLoader

from datasets.dataset import Dataset  # type: ignore[import-untyped]
from datasets.lightning_data_module import LightningDataModule  # type: ignore[import-untyped]
from datasets.transforms import Transforms  # type: ignore[import-untyped]

# LaRS panoptic category_id -> contiguous class index (11 classes).
LARS_CLASS_MAPPING: dict[int, int] = {
    1: 0,   # Static Obstacle
    3: 1,   # Water
    5: 2,   # Sky
    11: 3,  # Boat/ship
    12: 4,  # Row boats
    13: 5,  # Paddle board
    14: 6,  # Buoy
    15: 7,  # Swimmer
    16: 8,  # Animal
    17: 9,  # Float
    19: 10,  # Other
}

LARS_STUFF_CLASSES = [0, 1, 2]


class LarsPanoptic(LightningDataModule):
    """Same zip layout as COCOPanoptic; LaRS category IDs in panoptic JSON."""

    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 11,
        stuff_classes: list[int] | None = None,
        color_jitter_enabled: bool = False,
        scale_range: tuple[float, float] = (0.1, 2.0),
        check_empty_targets: bool = True,
    ) -> None:
        stuff_classes = stuff_classes or list(LARS_STUFF_CLASSES)
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.stuff_classes = stuff_classes
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(target, labels_by_id, is_crowd_by_id, **kwargs):
        target = target[0, :, :] + target[1, :, :] * 256 + target[2, :, :] * 256**2
        masks, labels, is_crowd = [], [], []
        for label_id in target.unique():
            if label_id.item() not in labels_by_id:
                continue
            cls_id = labels_by_id[label_id.item()]
            if cls_id not in LARS_CLASS_MAPPING:
                continue
            masks.append(target == label_id)
            labels.append(LARS_CLASS_MAPPING[cls_id])
            is_crowd.append(is_crowd_by_id[label_id.item()])
        return masks, labels, is_crowd

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }
        self.train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path("./train2017"),
            target_folder_path_in_zip=Path("./panoptic_train2017"),
            annotations_json_path_in_zip=Path("./annotations/panoptic_train2017.json"),
            target_zip_path_in_zip=Path("./annotations/panoptic_train2017.zip"),
            target_zip_path=Path(self.path, "panoptic_annotations_trainval2017.zip"),
            zip_path=Path(self.path, "train2017.zip"),
            **dataset_kwargs,
        )
        self.val_dataset = Dataset(
            img_folder_path_in_zip=Path("./val2017"),
            target_folder_path_in_zip=Path("./panoptic_val2017"),
            annotations_json_path_in_zip=Path("./annotations/panoptic_val2017.json"),
            target_zip_path_in_zip=Path("./annotations/panoptic_val2017.zip"),
            target_zip_path=Path(self.path, "panoptic_annotations_trainval2017.zip"),
            zip_path=Path(self.path, "val2017.zip"),
            **dataset_kwargs,
        )
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
