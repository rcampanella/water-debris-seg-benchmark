"""Per-detection / per-GT error classification on top of pycocotools matching.

Matching is taken from COCOeval.evalImgs (category-agnostic single class, area range
"all", maxDets 100, IoU threshold 0.5), so TP counts reconcile with AP50 in the metrics
files. Error types:

Detections                                  GT boxes
  tp        matched at IoU >= 0.5             tp         matched
  fp_merge  box contains centres of >= 2 GT   fn_merged  centre inside an fp_merge box
  fp_dup    IoU >= 0.5 with an already-       fn_frag    >= 2 dets lie mostly (IoA >= 0.5)
            matched GT                                   inside it, none matched
  fp_loc    best IoU in [0.1, 0.5)            fn_loc     best det IoU in [0.1, 0.5)
  fp_bkg    best IoU < 0.1                    fn_missed  best det IoU < 0.1
Precedence is top-down within each column.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IOU_MATCH = 0.5
IOU_NEAR = 0.1
IOA_FRAGMENT = 0.5
SMALL, MEDIUM = 32**2, 96**2

DET_TYPES = ("tp", "fp_merge", "fp_dup", "fp_loc", "fp_bkg")
GT_TYPES = ("tp", "fn_merged", "fn_frag", "fn_loc", "fn_missed")


def size_bucket(area: float) -> str:
    return "small" if area < SMALL else "medium" if area < MEDIUM else "large"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _xyxy(b) -> np.ndarray:
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)


def _pairwise(dets: np.ndarray, gts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IoU and intersection-over-det-area, shape (n_det, n_gt)."""
    if len(dets) == 0 or len(gts) == 0:
        z = np.zeros((len(dets), len(gts)))
        return z, z
    x1 = np.maximum(dets[:, None, 0], gts[None, :, 0])
    y1 = np.maximum(dets[:, None, 1], gts[None, :, 1])
    x2 = np.minimum(dets[:, None, 2], gts[None, :, 2])
    y2 = np.minimum(dets[:, None, 3], gts[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a_d = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    a_g = (gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1])
    iou = inter / np.maximum(a_d[:, None] + a_g[None, :] - inter, 1e-9)
    ioa = inter / np.maximum(a_d[:, None], 1e-9)
    return iou, ioa


@dataclass
class ImageErrors:
    image_id: int
    file_name: str
    width: int
    height: int
    dets: list[dict] = field(default_factory=list)  # bbox xywh, score, status, max_iou, area
    gts: list[dict] = field(default_factory=list)  # bbox xywh, status, best_iou, size

    def count(self, kind: str, status: str) -> int:
        rows = self.dets if kind == "det" else self.gts
        return sum(r["status"] == status for r in rows)

    @property
    def n_gt(self) -> int:
        return len(self.gts)

    @property
    def recall(self) -> float:
        return self.count("gt", "tp") / self.n_gt if self.gts else float("nan")


def classify_run(gt_path: Path, pred_path: Path, *, min_score: float = 0.0) -> dict[int, ImageErrors]:
    """Return per-image error records for one prediction file.

    `min_score` sets the operating point: detections below it are dropped before matching
    (0.0 reproduces the matching behind the metrics files).
    """
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from eval_coco_bbox import run_cocoeval  # noqa: PLC0415

    preds = [p for p in json.loads(Path(pred_path).read_text()) if p.get("score", 1.0) >= min_score]
    ev, _ = run_cocoeval(gt_path, preds)
    if ev is None:
        raise ValueError(f"No usable predictions in {pred_path}")
    coco_gt, coco_dt = ev.cocoGt, ev.cocoDt
    t = int(np.argmin(np.abs(ev.params.iouThrs - IOU_MATCH)))
    if len(ev.params.catIds) != 1:
        raise ValueError("classify_run expects a single-category GT")
    # evalImgs is ordered [cat][areaRng][img]; one category → index areaRng * n_img + img
    a_all = ev.params.areaRngLbl.index("all")
    img_ids = list(ev.params.imgIds)

    out: dict[int, ImageErrors] = {}
    for k_img, img_id in enumerate(img_ids):
        info = coco_gt.imgs[img_id]
        rec = ImageErrors(img_id, info["file_name"], info["width"], info["height"])
        out[img_id] = rec
        e = ev.evalImgs[a_all * len(img_ids) + k_img]
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        gt_boxes = _xyxy([a["bbox"] for a in gt_anns]) if gt_anns else np.zeros((0, 4))
        if e is None:  # image without GT and without dets
            dt_ids, dt_matches = [], np.zeros((1, 0))
        else:
            dt_ids, dt_matches = e["dtIds"], np.asarray(e["dtMatches"])
        dt_anns = coco_dt.loadAnns(dt_ids) if dt_ids else []
        dt_boxes = _xyxy([d["bbox"] for d in dt_anns]) if dt_anns else np.zeros((0, 4))
        iou, ioa = _pairwise(dt_boxes, gt_boxes)

        gt_id_to_idx = {a["id"]: j for j, a in enumerate(gt_anns)}
        matched_gt = set()
        det_status = []
        for i in range(len(dt_anns)):
            m = int(dt_matches[t, i]) if dt_matches.size else 0
            if m > 0:  # dtMatches holds the matched GT ann id, 0 = unmatched
                matched_gt.add(gt_id_to_idx[m])
                det_status.append("tp")
            else:
                det_status.append(None)
        gc = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2 if len(gt_boxes) else np.zeros((0, 2))
        merge_dets = set()
        for i in range(len(dt_anns)):
            if det_status[i] is not None:
                continue
            b = dt_boxes[i]
            inside = (gc[:, 0] >= b[0]) & (gc[:, 0] <= b[2]) & (gc[:, 1] >= b[1]) & (gc[:, 1] <= b[3])
            best = iou[i].max() if iou.shape[1] else 0.0
            if inside.sum() >= 2:
                det_status[i] = "fp_merge"
                merge_dets.add(i)
            elif best >= IOU_MATCH:
                det_status[i] = "fp_dup"
            elif best >= IOU_NEAR:
                det_status[i] = "fp_loc"
            else:
                det_status[i] = "fp_bkg"
        for i, d in enumerate(dt_anns):
            rec.dets.append({
                "det_id": d["id"], "bbox": d["bbox"], "score": d.get("score", 1.0),
                "area": d["bbox"][2] * d["bbox"][3], "status": det_status[i],
                "max_iou": float(iou[i].max()) if iou.shape[1] else 0.0,
            })

        for j, a in enumerate(gt_anns):
            best = float(iou[:, j].max()) if iou.shape[0] else 0.0
            if j in matched_gt:
                st = "tp"
            elif any(gc_inside(dt_boxes[i], gc[j]) for i in merge_dets):
                st = "fn_merged"
            elif iou.shape[0] and (ioa[:, j] >= IOA_FRAGMENT).sum() >= 2:
                st = "fn_frag"
            elif best >= IOU_NEAR:
                st = "fn_loc"
            else:
                st = "fn_missed"
            rec.gts.append({
                "gt_id": a["id"], "bbox": a["bbox"], "area": a["bbox"][2] * a["bbox"][3],
                "size": size_bucket(a["bbox"][2] * a["bbox"][3]), "status": st, "best_iou": best,
            })
    return out


def gc_inside(box: np.ndarray, c: np.ndarray) -> bool:
    return bool(box[0] <= c[0] <= box[2] and box[1] <= c[1] <= box[3])


def summarize(records: dict[int, ImageErrors]) -> dict:
    det = {s: 0 for s in DET_TYPES}
    gt = {s: {z: 0 for z in ("small", "medium", "large")} for s in GT_TYPES}
    for r in records.values():
        for d in r.dets:
            det[d["status"]] += 1
        for g in r.gts:
            gt[g["status"]][g["size"]] += 1
    n_gt = sum(sum(v.values()) for v in gt.values())
    return {
        "n_images": len(records),
        "n_gt": n_gt,
        "n_det_top100": sum(det.values()),
        "det": det,
        "gt": gt,
        "recall_at_iou50": (sum(gt["tp"].values()) / n_gt) if n_gt else None,
    }
