"""LaRS semantic (3-class) datamodule for EoMT — ADE20K-style zip layout."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from torch.utils.data import DataLoader

from datasets.dataset import Dataset  # type: ignore[import-untyped]
from datasets.lightning_data_module import LightningDataModule  # type: ignore[import-untyped]
from datasets.transforms import Transforms  # type: ignore[import-untyped]

# LaRS semantic PNG IDs → contiguous class index
LARS_SEMANTIC_CLASS_MAPPING: dict[int, int] = {0: 0, 1: 1, 2: 2}


class LarsSemantic(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 3,
        color_jitter_enabled: bool = True,
        scale_range: tuple[float, float] = (0.5, 2.0),
        check_empty_targets: bool = True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
            ignore_idx=255,
        )
        self.save_hyperparameters(ignore=["_class_path"])
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(target, **kwargs):
        masks, labels = [], []
        for label_id in target[0].unique():
            cls_id = label_id.item()
            if cls_id not in LARS_SEMANTIC_CLASS_MAPPING:
                continue
            masks.append(target[0] == label_id)
            labels.append(LARS_SEMANTIC_CLASS_MAPPING[cls_id])
        return masks, labels, [False for _ in range(len(masks))]

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        zip_path = Path(self.path, "lars_semantic.zip")
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "zip_path": zip_path,
            "target_zip_path": zip_path,
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }
        self.train_dataset = Dataset(
            img_folder_path_in_zip=Path("./images/training"),
            target_folder_path_in_zip=Path("./annotations/training"),
            transforms=self.transforms,
            **dataset_kwargs,
        )
        self.val_dataset = Dataset(
            img_folder_path_in_zip=Path("./images/validation"),
            target_folder_path_in_zip=Path("./annotations/validation"),
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
