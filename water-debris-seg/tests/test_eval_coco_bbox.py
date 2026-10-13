"""Guards for scripts/eval_coco_bbox.py (see decision log 2026-10-13).

Each case targets one defect of the legacy scorer, so a regression to that behaviour fails.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_coco_bbox import evaluate_bbox_map  # noqa: E402


def _write(tmp_path, gt_boxes_by_img, preds):
    images = [{"id": i, "file_name": f"{i}.jpg", "width": 640, "height": 480} for i in gt_boxes_by_img]
    anns = []
    for img_id, boxes in gt_boxes_by_img.items():
        for b in boxes:
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": 0, "bbox": b,
                         "area": b[2] * b[3], "iscrowd": 0})
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"images": images, "annotations": anns, "categories": [{"id": 0, "name": "waste"}]}))
    pr = tmp_path / "pred.json"
    pr.write_text(json.dumps([{"image_id": i, "category_id": 0, "bbox": b, "score": s} for i, b, s in preds]))
    return gt, pr


def test_perfect_predictions_score_one(tmp_path):
    gt, pr = _write(tmp_path, {1: [[10, 10, 100, 100]], 2: [[5, 5, 20, 20]]},
                    [(1, [10, 10, 100, 100], 0.9), (2, [5, 5, 20, 20], 0.8)])
    m = evaluate_bbox_map(gt, pr)
    assert m["mAP"] == pytest.approx(1.0) and m["AP_small"] == pytest.approx(1.0)


def test_map_averages_iou_thresholds_up_to_095(tmp_path):
    # IoU = 0.8: hit at thresholds .50-.80 (7 of 10) → mAP 0.7. Legacy (.50/.75 mean) gave 1.0.
    gt, pr = _write(tmp_path, {1: [[0, 0, 100, 100]]}, [(1, [0, 0, 100, 80], 0.9)])
    m = evaluate_bbox_map(gt, pr)
    assert m["AP50"] == pytest.approx(1.0)
    assert m["mAP"] == pytest.approx(0.7, abs=0.01)


def test_ap_small_counts_false_positives(tmp_path):
    # Higher-scored small FP ranks above the small TP → AP_small ≈ 0.5. Legacy gave 1.0.
    gt, pr = _write(tmp_path, {1: [[10, 10, 20, 20]]},
                    [(1, [300, 300, 20, 20], 0.9), (1, [10, 10, 20, 20], 0.5)])
    m = evaluate_bbox_map(gt, pr)
    assert m["AP_small"] == pytest.approx(0.5, abs=0.01)


def test_images_without_predictions_count_as_misses(tmp_path):
    # Image 2 has GT but no predictions → recall 0.5. Legacy skipped image 2 and gave 1.0.
    gt, pr = _write(tmp_path, {1: [[10, 10, 100, 100]], 2: [[10, 10, 100, 100]]},
                    [(1, [10, 10, 100, 100], 0.9)])
    m = evaluate_bbox_map(gt, pr)
    assert m["AR100"] == pytest.approx(0.5)
    assert m["mAP"] == pytest.approx(0.5, abs=0.01)


def test_predictions_outside_gt_are_dropped_and_counted(tmp_path):
    gt, pr = _write(tmp_path, {1: [[10, 10, 100, 100]]},
                    [(1, [10, 10, 100, 100], 0.9), (99, [0, 0, 5, 5], 0.9)])
    m = evaluate_bbox_map(gt, pr)
    assert m["n_pred_dropped"] == 1 and m["mAP"] == pytest.approx(1.0)
