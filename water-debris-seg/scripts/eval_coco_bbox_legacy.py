#!/usr/bin/env python3
"""LEGACY scorer, kept only to reproduce pre-2026-10-13 numbers for auditing. DO NOT USE.

Known defects (see docs/water_debris_seg_decision_log.md, 2026-10-13):
  - "mAP" = mean(AP@0.5, AP@0.75), not COCO AP@[.50:.95]
  - AP_small never counts false positives (precision is always 1)
  - images with GT but no predictions are skipped (their misses are not counted)
Replaced by eval_coco_bbox.py (pycocotools).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision.ops import box_iou


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def load_gt_targets(gt_path: Path) -> dict[int, dict]:
    with gt_path.open(encoding="utf-8") as f:
        gt = json.load(f)
    by_image: dict[int, dict] = {}
    for im in gt["images"]:
        by_image[im["id"]] = {"boxes": [], "labels": []}
    for ann in gt["annotations"]:
        img_id = ann["image_id"]
        if img_id not in by_image:
            continue
        by_image[img_id]["boxes"].append(xywh_to_xyxy(ann["bbox"]))
        by_image[img_id]["labels"].append(int(ann["category_id"]))
    for img_id in by_image:
        by_image[img_id]["boxes"] = torch.tensor(by_image[img_id]["boxes"], dtype=torch.float32)
        by_image[img_id]["labels"] = torch.tensor(by_image[img_id]["labels"], dtype=torch.int64)
    return by_image


def load_preds_by_image(pred_path: Path) -> dict[int, dict]:
    with pred_path.open(encoding="utf-8") as f:
        preds = json.load(f)
    by_image: dict[int, dict] = {}
    for ann in preds:
        img_id = ann["image_id"]
        if img_id not in by_image:
            by_image[img_id] = {"boxes": [], "scores": [], "labels": []}
        by_image[img_id]["boxes"].append(xywh_to_xyxy(ann["bbox"]))
        by_image[img_id]["scores"].append(float(ann.get("score", 1.0)))
        by_image[img_id]["labels"].append(int(ann["category_id"]))
    for img_id in by_image:
        by_image[img_id]["boxes"] = torch.tensor(by_image[img_id]["boxes"], dtype=torch.float32)
        by_image[img_id]["scores"] = torch.tensor(by_image[img_id]["scores"], dtype=torch.float32)
        by_image[img_id]["labels"] = torch.tensor(by_image[img_id]["labels"], dtype=torch.int64)
    return by_image


def _match_detections(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresh: float,
) -> tuple[int, int, int]:
    """Returns tp, fp, fn counts for one image at one IoU threshold."""
    if gt_boxes.numel() == 0:
        return 0, int(pred_boxes.shape[0]), 0
    if pred_boxes.numel() == 0:
        return 0, 0, int(gt_boxes.shape[0])

    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    tp = fp = 0
    for pb in pred_boxes:
        ious = box_iou(pb.unsqueeze(0), gt_boxes).squeeze(0)
        best_iou, best_j = (ious.max(0) if ious.numel() else (torch.tensor(0.0), torch.tensor(0, dtype=torch.long)))
        if float(best_iou) >= iou_thresh and not matched_gt[best_j]:
            matched_gt[best_j] = True
            tp += 1
        else:
            fp += 1
    fn = int((~matched_gt).sum().item())
    return tp, fp, fn


def _average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def evaluate_bbox_map(gt_path: Path, pred_path: Path) -> dict[str, float]:
    gt_by_img = load_gt_targets(gt_path)
    pred_by_img = load_preds_by_image(pred_path)
    image_ids = sorted(set(gt_by_img) & set(pred_by_img))
    if not image_ids:
        return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP_small": 0.0}

    ap_by_iou: dict[float, float] = {}
    ap_small_by_iou: dict[float, float] = {}
    area_small = 32 * 32

    for iou_thresh in (0.5, 0.75):
        all_scores: list[float] = []
        all_tp: list[int] = []
        all_scores_small: list[float] = []
        all_tp_small: list[int] = []
        total_gt = total_gt_small = 0

        for img_id in image_ids:
            gt_boxes = gt_by_img[img_id]["boxes"]
            total_gt += gt_boxes.shape[0]
            small_mask = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]) < area_small
            total_gt_small += int(small_mask.sum().item())

            pred_boxes = pred_by_img[img_id]["boxes"]
            pred_scores = pred_by_img[img_id]["scores"]
            if pred_boxes.numel() == 0:
                continue

            order = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[order]
            pred_scores = pred_scores[order]
            matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)

            for pb, sc in zip(pred_boxes, pred_scores):
                ious = box_iou(pb.unsqueeze(0), gt_boxes).squeeze(0) if gt_boxes.numel() else torch.tensor([])
                is_tp = 0
                if ious.numel():
                    best_iou, best_j = ious.max(0)
                    if float(best_iou) >= iou_thresh and not matched_gt[best_j]:
                        matched_gt[best_j] = True
                        is_tp = 1
                all_scores.append(float(sc))
                all_tp.append(is_tp)
                if is_tp and float(best_iou) >= iou_thresh:
                    is_small = (
                        (gt_boxes[best_j, 2] - gt_boxes[best_j, 0])
                        * (gt_boxes[best_j, 3] - gt_boxes[best_j, 1])
                    ) < area_small
                else:
                    is_small = torch.tensor(False)
                if is_tp and is_small:
                    all_scores_small.append(float(sc))
                    all_tp_small.append(1)
                elif not is_tp:
                    pass

        def ap_from_lists(scores, tps, n_gt):
            if n_gt == 0:
                return 0.0
            if not scores:
                return 0.0
            idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            tp_cum = fp_cum = 0
            precisions, recalls = [], []
            for i in idx:
                if tps[i]:
                    tp_cum += 1
                else:
                    fp_cum += 1
                precisions.append(tp_cum / (tp_cum + fp_cum))
                recalls.append(tp_cum / n_gt)
            return _average_precision(recalls, precisions)

        ap_by_iou[iou_thresh] = ap_from_lists(all_scores, all_tp, total_gt)
        ap_small_by_iou[iou_thresh] = ap_from_lists(all_scores_small, all_tp_small, total_gt_small)

    return {
        "mAP": float((ap_by_iou[0.5] + ap_by_iou[0.75]) / 2),
        "AP50": float(ap_by_iou[0.5]),
        "AP75": float(ap_by_iou[0.75]),
        "AP_small": float(ap_small_by_iou[0.5]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-coco", type=Path, required=True)
    parser.add_argument("--pred-coco", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-group", default="eval_coco_bbox")
    parser.add_argument("--wandb-name", default=None)
    args = parser.parse_args()

    metrics = evaluate_bbox_map(args.gt_coco, args.pred_coco)
    print(json.dumps(metrics, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if args.wandb:
        import wandb

        wandb.init(
            project="water-debris-seg",
            group=args.wandb_group,
            name=args.wandb_name,
            config={"gt": str(args.gt_coco), "pred": str(args.pred_coco)},
        )
        wandb.log(metrics)
        wandb.finish()


if __name__ == "__main__":
    main()
