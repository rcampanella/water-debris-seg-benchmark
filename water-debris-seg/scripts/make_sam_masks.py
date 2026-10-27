#!/usr/bin/env python3
"""SAM pseudo-masks for FloW / CANSURF training images, prompted by the ground-truth boxes (fairness fix 3), + QC.

For every training image (the split pool in data/water_debris/yolo_splits/<dataset>/train_all), each GT box prompts the
frozen SAM 2.1 large; the mask is stored as COCO instance RLE with the GT box. Test images get no masks.

Outputs (home):
  data/water_debris/sam_masks/<dataset>/instances_{train90,held10,train_all}.json      (COCO instances, compressed RLE)
  eval_outputs/water_debris_seg/fairness_fixes/sources/sam_qc_<dataset>.json             (QC numbers + stop rules)
  eval_outputs/water_debris_seg/fairness_fixes/sam_qc/<dataset>_*.png                     (20 renders, fixed selection)

QC (FIX3_SAM_BOX_DATASETS.md, step 2; rule revised 2026-10-23): stop if > 5% of masks are empty or cover < 10% of their
box, or if the 20 renders show wrong masks. The IoU of each mask's min/max box with its GT box (by GT box size) is
reported as the SAM-box ceiling, not as a stop rule: FloW / CANSURF GT boxes are looser than the objects, so correct masks
give box IoUs of ~0.8 (median on 10 FloW images: 0.80, small boxes 0.68).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SPLITS = REPO / "data/water_debris/yolo_splits"
OUT = REPO / "data/water_debris/sam_masks"
SRC = REPO / "eval_outputs/water_debris_seg/fairness_fixes/sources"
FIG = REPO / "eval_outputs/water_debris_seg/fairness_fixes/sam_qc"
SAM_W = REPO / "data/water_debris/checkpoints/sam/sam2.1_l.pt"
NAMES = {"flow": "waste", "cansurf": "debris"}


def box_iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


def sam_masks(sam, im, boxes_xyxy):
    """One SAM mask per box, in order. SAM sometimes returns fewer masks than prompts for a batch (e.g. duplicate or
    degenerate boxes); then each box is prompted alone, and a box that still gets no mask gets an empty mask."""
    H, W = im.shape[:2]
    if not boxes_xyxy:
        return []
    res = sam(im, bboxes=boxes_xyxy, verbose=False)[0]
    if res.masks is not None and len(res.masks.data) == len(boxes_xyxy):
        return [m.astype(bool) for m in res.masks.data.cpu().numpy()]
    out = []
    for b in boxes_xyxy:
        r = sam(im, bboxes=[b], verbose=False)[0]
        out.append(r.masks.data[0].cpu().numpy().astype(bool) if r.masks is not None and len(r.masks.data) else np.zeros((H, W), bool))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(NAMES), required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from PIL import Image
    from pycocotools import mask as cm
    from ultralytics import SAM

    root = SPLITS / args.dataset
    imgs = sorted((root / "train_all/images").iterdir())[: args.limit or None]
    split_of = {s: {p.name for p in (root / s / "images").iterdir()} for s in ("train90", "held10")}
    sam = SAM(str(SAM_W))
    images, anns, qc_rows = [], [], []
    aid = 0
    for k, p in enumerate(imgs):
        im = np.asarray(Image.open(p).convert("RGB"))
        H, W = im.shape[:2]
        iid = k + 1
        images.append({"id": iid, "file_name": p.name, "width": W, "height": H})
        lab = root / "train_all/labels" / f"{p.stem}.txt"
        boxes = []
        for line in lab.read_text().splitlines() if lab.exists() else []:
            if line.strip():
                _, cx, cy, w, h = map(float, line.split())
                boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, w * W, h * H])
        if boxes:
            masks = sam_masks(sam, im, [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes])
            for b, m in zip(boxes, masks):
                aid += 1
                rle = cm.encode(np.asfortranarray(m.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode()
                ys, xs = np.nonzero(m)
                mbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1)] if xs.size else [0, 0, 0, 0]
                area = float(m.sum())
                anns.append({"id": aid, "image_id": iid, "category_id": 1, "bbox": [round(v, 2) for v in b],
                             "segmentation": rle, "area": area, "iscrowd": 0})
                qc_rows.append({"image": p.name, "gt_area": b[2] * b[3], "iou": box_iou(mbox, b) if xs.size else 0.0,
                                "empty": not xs.size, "tiny": area < 0.10 * b[2] * b[3]})
        if (k + 1) % 200 == 0:
            print(f"{k + 1}/{len(imgs)}", flush=True)

    out = OUT / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    cats = [{"id": 1, "name": NAMES[args.dataset]}]
    by_name = {im["file_name"]: im for im in images}
    for split in ("train_all", "train90", "held10"):
        keep = set(by_name) if split == "train_all" else split_of[split] & set(by_name)
        ids = {by_name[n]["id"] for n in keep}
        doc = {"images": [im for im in images if im["id"] in ids], "annotations": [a for a in anns if a["image_id"] in ids],
               "categories": cats, "info": {"masks": "SAM 2.1 large prompted by GT boxes", "sam_sha256":
                                             hashlib.sha256(SAM_W.read_bytes()).hexdigest()}}
        (out / f"instances_{split}.json").write_text(json.dumps(doc))

    iou = np.array([r["iou"] for r in qc_rows])
    small = np.array([r["gt_area"] < 32 * 32 for r in qc_rows])
    med = np.array([32 * 32 <= r["gt_area"] < 96 * 96 for r in qc_rows])
    empty = float(np.mean([r["empty"] for r in qc_rows])) if qc_rows else 0.0
    tiny = float(np.mean([r["tiny"] for r in qc_rows])) if qc_rows else 0.0
    qc = {"dataset": args.dataset, "images": len(images), "masks": len(anns),
          "box_iou_median": {"all": float(np.median(iou)), "small": float(np.median(iou[small])) if small.any() else None,
                             "medium": float(np.median(iou[med])) if med.any() else None,
                             "large": float(np.median(iou[~small & ~med])) if (~small & ~med).any() else None},
          "box_iou_share_below_0.5": float(np.mean(iou < 0.5)),
          "n_by_size": {"small": int(small.sum()), "medium": int(med.sum()), "large": int((~small & ~med).sum())},
          "empty_share": empty, "tiny_share": tiny}
    qc["stop_rules"] = {"empty_plus_tiny_le_5pct": empty + tiny <= 0.05,
                        "renders_reviewed": "pending manual review of the 20 renders"}
    qc["pass_automatic"] = qc["stop_rules"]["empty_plus_tiny_le_5pct"]
    qc["box_iou_is_ceiling_not_stop_rule"] = True

    # 20 renders: 7 with the smallest boxes, 7 with the most boxes, 6 random (seed 0)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    per_img = {}
    for a in anns:
        per_img.setdefault(a["image_id"], []).append(a)
    ids = list(per_img)
    pick = sorted(ids, key=lambda i: min(a["bbox"][2] * a["bbox"][3] for a in per_img[i]))[:7]
    pick += [i for i in sorted(ids, key=lambda i: -len(per_img[i])) if i not in pick][:7]
    rest = [i for i in ids if i not in pick]
    random.Random(0).shuffle(rest)
    pick += rest[:6]
    id2im = {im["id"]: im for im in images}
    qc["renders"] = []
    for n, i in enumerate(pick):
        im = id2im[i]
        img = np.asarray(Image.open(root / "train_all/images" / im["file_name"]).convert("RGB"))
        overlay = img.copy()
        fig, ax = plt.subplots(figsize=(8, 8 * img.shape[0] / img.shape[1]))
        for a in per_img[i]:
            m = cm.decode({"size": a["segmentation"]["size"], "counts": a["segmentation"]["counts"].encode()}).astype(bool)
            overlay[m] = (0.5 * overlay[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        ax.imshow(overlay)
        for a in per_img[i]:
            x, y, w, h = a["bbox"]
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="lime", lw=1))
        ax.set_title(f"{im['file_name']}  GT boxes (green), SAM masks (magenta)", fontsize=8)
        ax.axis("off")
        f = FIG / f"{args.dataset}_{n:02d}.png"
        fig.savefig(f, dpi=110, bbox_inches="tight")
        plt.close(fig)
        qc["renders"].append(str(f.relative_to(REPO)))
    SRC.mkdir(parents=True, exist_ok=True)
    (SRC / f"sam_qc_{args.dataset}{'_limit' if args.limit else ''}.json").write_text(json.dumps(qc, indent=2))
    print(json.dumps({k: v for k, v in qc.items() if k != "renders"}, indent=1))


if __name__ == "__main__":
    main()
