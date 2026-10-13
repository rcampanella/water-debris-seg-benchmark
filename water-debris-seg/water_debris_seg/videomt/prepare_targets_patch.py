"""Fix VidEoMT ``prepare_targets`` using last-frame class labels on keyframe-sparse LaRS clips."""

from __future__ import annotations

import torch
from detectron2.structures import BitMasks


def prepare_targets_fixed(self, targets, images):
    """Per-video labels from any labeled frame (not only the last sampled frame)."""
    h_pad, w_pad = images.tensor.shape[-2:]
    num_thing = self.backbone.num_classes
    gt_instances = []
    for targets_per_video in targets:
        _num_instance = len(targets_per_video["instances"][0])
        mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
        gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

        gt_ids_per_video = []
        gt_classes_per_frame = []
        for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
            targets_per_frame = targets_per_frame.to(self.device)
            h, w = targets_per_frame.image_size

            gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
            gt_classes_per_frame.append(targets_per_frame.gt_classes[:, None])
            if isinstance(targets_per_frame.gt_masks, BitMasks):
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
            else:
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

        gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
        gt_classes_per_frame = torch.cat(gt_classes_per_frame, dim=1)
        valid_idx = (gt_ids_per_video != -1).any(dim=-1)

        # Padding instances use category_id == num_thing; real LaRS labels are 0..num_thing-1.
        real_classes = gt_classes_per_frame.clone()
        real_classes[real_classes >= num_thing] = num_thing + 10_000
        gt_classes_per_video = real_classes.min(dim=1).values[valid_idx]

        gt_ids_per_video = gt_ids_per_video[valid_idx]
        gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
        gt_masks_per_video = gt_masks_per_video[valid_idx].float()
        gt_instances[-1].update({"masks": gt_masks_per_video})

    return gt_instances


def apply_prepare_targets_patch() -> None:
    from videomt.videomt import videomt

    videomt.prepare_targets = prepare_targets_fixed
