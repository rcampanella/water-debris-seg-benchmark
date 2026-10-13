"""Make VidEoMT online training work with LaRS keyframe-only labels.

Two upstream assumptions break on LaRS (``images_seq`` clips, panoptic masks on keyframes only):

1. **Sampling.** ``YTVISDatasetMapper.select_frames`` picks a contiguous window with
   ``start = randrange(L - F)``. Most LaRS clips are 10 frames with the keyframe at index 9,
   which that window can never reach; only ~8.5% of sampled windows held a labeled frame, so
   most iterations had zero GT (logged median ``loss_mask`` = 0, VIS AP 0 in job 27769861).
   Patch: when the clip has labeled frames, pick one and sample a window that contains it.

2. **Loss.** ``VideoSetCriterion`` (via ``VideoHungarianMatcher_Consistent``) repeats each
   window's matching on every frame. On YTVIS every sampled frame is annotated, so ``id == -1``
   means "absent". On LaRS it means "not annotated": upstream would push matched queries to
   empty masks and all queries to no-object on unlabeled frames. Patch: match on the full
   window (keeps consistent query assignment), then compute losses only on labeled frames and
   only for tracks present in that frame.
"""

from __future__ import annotations

import random

import torch


def _labeled_window(video_length: int, labeled: list[int], num_frames: int) -> list[int]:
    key = random.choice(labeled)
    lo = max(0, key - num_frames + 1)
    hi = min(key, video_length - num_frames)
    start = random.randint(lo, hi)
    return list(range(start, start + num_frames))


def apply_keyframe_sampling_patch() -> None:
    from videomt.data_video.dataset_mapper import YTVISDatasetMapper

    if getattr(YTVISDatasetMapper, "_wds_keyframe_patched", False):
        return
    orig_call = YTVISDatasetMapper.__call__
    orig_select = YTVISDatasetMapper.select_frames

    def select_frames(self, video_length):
        forced = getattr(self, "_wds_forced_idx", None)
        if forced is not None:
            return forced
        return orig_select(self, video_length)

    def __call__(self, dataset_dict):
        self._wds_forced_idx = None
        if self.is_train and self.sampling_frame_ratio >= 1.0:
            L = dataset_dict["length"]
            F = self.sampling_frame_num
            labeled = [i for i, fr in enumerate(dataset_dict.get("annotations") or []) if fr]
            if labeled and L >= F:
                idx = _labeled_window(L, labeled, F)
                if self.reverse_agu and random.random() < 0.5:
                    idx = idx[::-1]
                self._wds_forced_idx = idx
        try:
            return orig_call(self, dataset_dict)
        finally:
            self._wds_forced_idx = None

    YTVISDatasetMapper.select_frames = select_frames
    YTVISDatasetMapper.__call__ = __call__
    YTVISDatasetMapper._wds_keyframe_patched = True


def _labeled_frame_inputs(outputs, targets, indices):
    """Keep only labeled frames; drop matches to tracks not annotated in that frame."""
    keep = [i for i, t in enumerate(targets) if bool((t["ids"] != -1).any())]
    sub_out = {k: outputs[k][keep] for k in ("pred_logits", "pred_masks")}
    sub_tgt, sub_idx = [], []
    for i in keep:
        t = targets[i]
        src, tgt = indices[i]
        present = (t["ids"][:, 0] != -1).cpu()
        sel = present[tgt] if len(tgt) else torch.zeros(0, dtype=torch.bool)
        sub_idx.append((src[sel], tgt[sel]))
        sub_tgt.append(t)
    return keep, sub_out, sub_tgt, sub_idx


def labeled_frames_criterion_forward(self, outputs, targets, matcher_outputs=None, ret_match_result=False):
    from videomt.criterion_videomt import get_world_size, is_dist_avail_and_initialized

    match_src = outputs if matcher_outputs is None else matcher_outputs
    indices = self.matcher({k: v for k, v in match_src.items() if k != "aux_outputs"}, targets)

    keep, sub_out, sub_tgt, sub_idx = _labeled_frame_inputs(outputs, targets, indices)
    device = outputs["pred_logits"].device
    num_masks = torch.as_tensor([float(sum(len(j) for _, j in sub_idx))], device=device)
    if is_dist_avail_and_initialized():
        torch.distributed.all_reduce(num_masks)
    num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

    def layer_losses(out, tgt, idx):
        if not keep:
            # No labeled frame in this batch: zero loss that still touches the graph.
            z = out["pred_logits"].sum() * 0 + out["pred_masks"].sum() * 0
            return {"loss_ce": z, "loss_mask": z, "loss_dice": z}
        res = {}
        for loss in self.losses:
            res.update(self.get_loss(loss, out, tgt, idx, num_masks))
        return res

    losses = layer_losses(outputs, sub_tgt, sub_idx)

    if "aux_outputs" in outputs:
        for i, aux in enumerate(outputs["aux_outputs"]):
            aux_indices = self.matcher(aux, targets) if matcher_outputs is None else indices
            _, aux_out, aux_tgt, aux_idx = _labeled_frame_inputs(aux, targets, aux_indices)
            l_dict = layer_losses(aux_out if keep else aux, aux_tgt, aux_idx)
            losses.update({k + f"_{i}": v for k, v in l_dict.items()})

    if ret_match_result:
        return losses, indices
    return losses


def apply_labeled_frames_loss_patch() -> None:
    from videomt.criterion_videomt import VideoSetCriterion

    VideoSetCriterion.forward = labeled_frames_criterion_forward



def split_ytvis_by_keyframe(gt: dict, results: list[dict]) -> tuple[dict, list[dict]]:
    """Turn each (video, labeled keyframe) into a 1-frame pseudo-video for GT and predictions.

    The LaRS YTVIS export has one labeled frame per GT track (no instance IDs link keyframes),
    while val clips hold 2-5 keyframes and are otherwise unlabeled. Upstream video IoU adds
    predictions on GT-less frames to the union, so even a perfect tracker scores AP 0. Scored
    per keyframe, "VIS AP" becomes instance-mask AP on keyframes from the video model.
    """
    from pycocotools import mask as mask_utils

    mult = 1 + max(v["length"] for v in gt["videos"])
    labeled: dict[int, set[int]] = {}
    for ann in gt["annotations"]:
        labeled.setdefault(ann["video_id"], set()).update(
            i for i, seg in enumerate(ann["segmentations"]) if seg
        )

    videos = []
    for v in gt["videos"]:
        for k in sorted(labeled.get(v["id"], ())):
            videos.append({**v, "id": v["id"] * mult + k, "length": 1, "file_names": [v["file_names"][k]]})

    anns = []
    for ann in gt["annotations"]:
        for k, seg in enumerate(ann["segmentations"]):
            if not seg:
                continue
            new = {**ann, "id": len(anns) + 1, "video_id": ann["video_id"] * mult + k, "segmentations": [seg]}
            for key in ("areas", "bboxes"):
                if key in ann:
                    new[key] = [ann[key][k]]
            anns.append(new)

    preds = []
    for r in results:
        for k in sorted(labeled.get(r["video_id"], ())):
            seg = r["segmentations"][k] if k < len(r["segmentations"]) else None
            if not seg or mask_utils.area(seg) == 0:
                continue
            preds.append({"video_id": r["video_id"] * mult + k, "category_id": r["category_id"],
                          "score": r["score"], "segmentations": [seg]})

    return {**gt, "videos": videos, "annotations": anns}, preds


def ytvos_from_dict(dataset: dict):
    from videomt.data_video.datasets.ytvis_api.ytvos import YTVOS

    api = YTVOS()
    api.dataset = dataset
    api.createIndex()
    return api


def apply_keyframe_eval_patch() -> None:
    """In-training YTVIS eval scores per labeled keyframe (see ``split_ytvis_by_keyframe``)."""
    from videomt.data_video import ytvis_eval

    if getattr(ytvis_eval, "_wds_keyframe_patched", False):
        return
    orig = ytvis_eval._evaluate_predictions_on_coco

    def _evaluate_predictions_on_coco(coco_gt, coco_results, img_ids=None):
        gt, preds = split_ytvis_by_keyframe(coco_gt.dataset, coco_results)
        if not preds:
            preds = coco_results[:0]
        return orig(ytvos_from_dict(gt), preds, img_ids=img_ids)

    ytvis_eval._evaluate_predictions_on_coco = _evaluate_predictions_on_coco
    ytvis_eval._wds_keyframe_patched = True
