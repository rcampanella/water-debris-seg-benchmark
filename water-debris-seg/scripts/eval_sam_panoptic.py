#!/usr/bin/env python3
"""Detections -> SAM masks -> LaRS panoptic map -> PQ and mIoU (fairness fix 2: YOLOv11s + SAM; fix 1: EoMT + SAM control).

  --source yolo   YOLO checkpoint with 11 classes (9 + Water, Sky); boxes above a confidence threshold prompt SAM
  --source eomt   EoMT panoptic checkpoint; the segments EoMT keeps for PQ (its own keep rule, no threshold) give the boxes

SAM (frozen, Ultralytics SAM 2.1 large) turns each box into a mask. Masks are merged into one segment per pixel with
EoMT's own rule: objects first (highest score first), then Static Obstacle, Water, Sky; a segment keeping less than
80% (EoMT's overlap_thresh) of its mask is dropped; region classes merge into one segment per class; unclaimed pixels
are void. PQ is computed with EoMT's metric code (as eval_eomt_lars_panoptic_pq.py); mIoU on obstacle / water / sky.

--split held10   (LaRS train sequences held out in ablation 10) PQ only; --thresholds sweeps YOLO confidence (stage-A
                 model) to pick the operating point; also writes the pilot checks (Water / Sky mask IoU)
--split val      LaRS val; PQ and mIoU at --thresholds (one value)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "water-debris-seg"))

LARS = Path("/scratch-shared/rcampanella/water-debris-seg/data/lars")
HELD10 = REPO / "data/water_debris/lars_backup/yolo_boxes/held10/labels"
SAM_W = REPO / "data/water_debris/checkpoints/sam/sam2.1_l.pt"
CAT_TO_IDX = {1: 0, 3: 1, 5: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7, 16: 8, 17: 9, 19: 10}
NUM_LARS, STUFF = 11, [0, 1, 2]
YOLO11_TO_LARS = {0: 0, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 1, 10: 2}
LARS_NAMES = ["Static Obstacle", "Water", "Sky", "Boat/ship", "Row boats", "Paddle board", "Buoy", "Swimmer", "Animal",
              "Float", "Other"]
OVERLAP = 0.8


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def priority(c):
    return 0 if c >= 3 else {0: 1, 1: 2, 2: 3}[c]          # objects, Static Obstacle, Water, Sky


def merge(segments, shape):
    """segments: [(lars_class, score, bool mask)] -> EoMT-format preds (H, W, 2): class, segment id (void: 11, -1)."""
    import torch

    cls = torch.full(shape, NUM_LARS, dtype=torch.long)
    seg = torch.full(shape, -1, dtype=torch.long)
    taken = np.zeros(shape, bool)
    stuff_id, sid = {}, 0
    for c, s, m in sorted(segments, key=lambda t: (priority(t[0]), -t[1])):
        orig = m.sum()
        new = m & ~taken
        if orig == 0 or new.sum() / orig < OVERLAP:
            continue
        if c in STUFF:
            if c not in stuff_id:
                stuff_id[c] = sid
                sid += 1
            k = stuff_id[c]
        else:
            k = sid
            sid += 1
        tn = torch.from_numpy(new)
        cls[tn], seg[tn] = c, k
        taken |= new
    return torch.stack([cls, seg], -1)


def load_target(split, stem, ann):
    import torch
    from PIL import Image

    rgb = np.asarray(Image.open(LARS / split / "panoptic_masks" / f"{stem}.png").convert("RGB")).astype(np.int64)
    ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    target = -torch.ones((*ids.shape, 2), dtype=torch.long)
    crowd, gt_region = [], {}
    for k, s in enumerate(ann["segments_info"]):
        m = ids == s["id"]
        target[:, :, 0][torch.from_numpy(m)] = CAT_TO_IDX[s["category_id"]]
        target[:, :, 1][torch.from_numpy(m)] = k
        crowd.append(bool(s.get("iscrowd", 0)))
        if s["category_id"] in (3, 5):
            gt_region[CAT_TO_IDX[s["category_id"]]] = gt_region.get(CAT_TO_IDX[s["category_id"]], np.zeros(ids.shape, bool)) | m
    return target, torch.tensor(crowd + [False]), gt_region


def eomt_kept_segments(model, img, device):
    """Segments EoMT keeps for PQ, as (class, score, box xyxy): a step-by-step copy of EoMT's to_per_pixel_preds_panoptic
    (training/lightning_module.py, same thresholds) that also records each segment's class confidence."""
    import torch

    from eval_eomt_fairness import forward

    ml, cl = forward(model, img, device)
    scores, classes = cl.softmax(-1).max(-1)
    keep = classes.ne(cl.shape[-1] - 1) & (scores > model.mask_thresh)
    if not keep.any():
        return []
    masks = ml.sigmoid()[keep]
    ks, kc = scores[keep], classes[keep]
    mask_ids = (ks[:, None, None] * masks).argmax(0)
    segs = {}                                          # key -> [class, score, mask]
    for q in range(len(kc)):
        c = int(kc[q])
        orig = masks[q] >= 0.5
        new = mask_ids == q
        final = orig & new
        oa, na = int(orig.sum()), int(new.sum())
        if oa == 0 or na == 0 or int(final.sum()) == 0 or na / oa < model.overlap_thresh:
            continue
        key = ("stuff", c) if c in model.stuff_classes else ("thing", q)
        if key in segs:
            segs[key][1] = max(segs[key][1], float(ks[q]))
            segs[key][2] |= final
        else:
            segs[key] = [c, float(ks[q]), final.clone()]
    out = []
    for c, sc, m in segs.values():
        ys, xs = torch.nonzero(m, as_tuple=True)
        out.append((c, sc, [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], m.cpu().numpy()))
    return out


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
    ap.add_argument("--source", choices=["yolo", "eomt"], required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", choices=["held10", "val"], required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.25])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-sam", action="store_true",
                    help="check: EoMT's own masks instead of SAM; must reproduce EoMT's panoptic PQ exactly")
    ap.add_argument("--export-png", type=Path, help="val, one threshold: also write the scored panoptic maps as LaRS evaluator PNGs")
    args = ap.parse_args()
    assert not args.export_png or (args.split == "val" and len(args.thresholds) == 1), "--export-png needs --split val and one threshold"

    import torch
    from PIL import Image

    from water_debris_seg.lars_panoptic_png import write_lars_png
    from torchvision import tv_tensors
    from training.lightning_module import LightningModule  # type: ignore[import-untyped]
    from ultralytics import SAM

    from eval_eomt_lars_panoptic_pq import MetricHost

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_dir = "train" if args.split == "held10" else "val"
    pan = json.loads((LARS / split_dir / "panoptic_annotations.json").read_text())
    anns = sorted(pan["annotations"], key=lambda a: a["file_name"])
    if args.split == "held10":
        keep = {p.stem for p in HELD10.glob("*.txt")}
        anns = [a for a in anns if Path(a["file_name"]).stem in keep]
    anns = anns[: args.limit or None]
    sam = SAM(str(SAM_W))
    if args.source == "yolo":
        from ultralytics import YOLO

        det = YOLO(str(args.weights))
        thresholds = sorted(args.thresholds)
    else:
        from eval_eomt_cansurf_mask2box import build_model_lars_panoptic

        det = build_model_lars_panoptic(args.weights, device, num_classes=NUM_LARS, stuff_classes=STUFF)
        thresholds = [None]

    hosts = {t: MetricHost() for t in thresholds}
    for h in hosts.values():
        h.metrics.to(device)
    conf = {t: np.zeros((4, 3), np.int64) for t in thresholds}
    region_iou = {1: [], 2: []}
    for k, a in enumerate(anns):
        stem = Path(a["file_name"]).stem
        path = LARS / split_dir / "images" / f"{stem}.jpg"
        im = np.asarray(Image.open(path).convert("RGB"))
        H, W = im.shape[:2]
        if args.source == "yolo":
            r = det.predict(str(path), imgsz=640, conf=thresholds[0], max_det=300, verbose=False)[0]
            dets = [(YOLO11_TO_LARS[int(c)], float(s), [float(v) for v in b]) for b, s, c in
                    zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        else:
            with torch.no_grad():
                full = eomt_kept_segments(det, tv_tensors.Image(torch.from_numpy(im.copy()).permute(2, 0, 1)), device)
            dets = [f[:3] for f in full]
        masks = []
        if dets and args.no_sam:
            masks = [f[3] for f in full]
        elif dets:
            masks = sam_masks(sam, im, [d[2] for d in dets])
        target, crowd, gt_region = load_target(split_dir, stem, a)
        for t in thresholds:
            segs = [(c, s, m) for (c, s, _), m in zip(dets, masks) if t is None or s >= t]
            preds = merge(segs, (H, W))
            LightningModule.update_metrics_panoptic(hosts[t], [preds.to(device)], [target.to(device)], [crowd.to(device)], 0)
            if args.export_png:
                write_lars_png(preds, args.export_png / f"{stem}.png")
            if args.split == "val":
                sem = np.asarray(Image.open(LARS / "val/semantic_masks" / f"{stem}.png"))
                c11 = preds[:, :, 0].numpy()
                p3 = np.full(c11.shape, 3)
                p3[(c11 == 0) | ((c11 >= 3) & (c11 < NUM_LARS))] = 0
                p3[c11 == 1], p3[c11 == 2] = 1, 2
                v = sem != 255
                np.add.at(conf[t], (p3[v], sem[v]), 1)
        for c in (1, 2):       # pilot: best SAM mask of each region class vs the GT region
            if c in gt_region:
                cand = [m for (cc, _, _), m in zip(dets, masks) if cc == c]
                g = gt_region[c]
                region_iou[c].append(max([(m & g).sum() / max((m | g).sum(), 1) for m in cand], default=0.0))
        if (k + 1) % 25 == 0:
            print(f"{k + 1}/{len(anns)}", flush=True)

    out = {"source": args.source, "masks": "EoMT own (check)" if args.no_sam else "SAM", "weights": str(args.weights), "weights_sha256": sha(args.weights), "split": args.split,
           "n_images": len(anns), "sam": {"weights": str(SAM_W), "sha256": sha(SAM_W)}, "merge_rule":
           "objects (score desc) > Static Obstacle > Water > Sky; drop segment if < 80% of its mask is kept", "by_threshold": {}}
    for t in thresholds:
        r = hosts[t].metrics[0].compute()[:-1]
        pq = r[:, 0]
        order = sorted(hosts[t].metrics[0].cat_id_to_continuous_id, key=hosts[t].metrics[0].cat_id_to_continuous_id.get)[:-1]
        entry = {"PQ": float(pq.mean()), "PQ_things": float(pq[[i for i, c in enumerate(order) if c >= 3]].mean()),
                 "PQ_stuff": float(pq[[i for i, c in enumerate(order) if c < 3]].mean()),
                 "PQ_per_class": {LARS_NAMES[c]: float(pq[i]) for i, c in enumerate(order)}}
        if args.split == "val":
            cm = conf[t]
            iou = [cm[c, c] / (cm[c, :].sum() + cm[:, c].sum() - cm[c, c]) for c in range(3)]
            entry["mIoU"] = float(np.mean(iou))
            entry["IoU_per_class"] = dict(zip(["obstacle", "water", "sky"], map(float, iou)))
        out["by_threshold"]["eomt_keep_rule" if t is None else f"{t:.2f}"] = entry
    out["pilot_region_iou"] = {LARS_NAMES[c]: {"median": float(np.median(v)) if v else None, "n_images": len(v),
                                                "share_below_0.8": float(np.mean(np.array(v) < 0.8)) if v else None}
                               for c, v in region_iou.items()}
    if args.source == "yolo" and len(thresholds) > 1:
        best = max(thresholds, key=lambda t: out["by_threshold"][f"{t:.2f}"]["PQ"])
        out["selected_threshold"] = best
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k in ("by_threshold", "pilot_region_iou", "selected_threshold")}, indent=1)[:2500])


if __name__ == "__main__":
    main()
