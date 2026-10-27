#!/usr/bin/env python3
"""Build YOLO box labels for LaRS from the panoptic instance masks (ablation 10).

Box = min/max of the mask pixels per coordinate:
    x_min, x_max = min/max{x : M(x,y)=1};  y_min, y_max = min/max{y : M(x,y)=1}
    COCO [x_min, y_min, w, h] with w = x_max - x_min + 1, h = y_max - y_min + 1
    YOLO "cls cx/W cy/H w/W h/H" with cx = x_min + w/2, cy = y_min + h/2 (W, H = that image's size)

Classes (9): Static Obstacle + the 8 LaRS object classes; Water and Sky are dropped; Float stays a
separate class. Crowd segments are left out of the YOLO labels and kept as iscrowd=1 in the COCO GT,
so pycocotools ignores detections on them.

Splits: train_all = official LaRS train; held10 = 10% of train *sequences* (seeded; LaRS frames come
from videos, so a frame-level split would leak near-duplicates); train90 = the rest; val = official
LaRS val (scoring only; LaRS test labels are not public).

Outputs
  <out>/{train_all,train90,held10,val}/{images (symlinks), labels}
  <out>/stageA.yaml (train90 -> held10), stageB.yaml (train_all; held10 monitored only), eval_val.yaml
  <eval>/sources/lars_{val,held10}_gt_coco_boxes.json, lars_yolo_label_stats.json, box_check_examples.{png,json}
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

KEEP = {1: "Static Obstacle", 11: "Boat/ship", 12: "Row boats", 13: "Paddle board", 14: "Buoy",
        15: "Swimmer", 16: "Animal", 17: "Float", 19: "Other"}
YOLO_ID = {cat: i for i, cat in enumerate(KEEP)}          # 0 Static Obstacle, 1 Boat/ship, ..., 7 Float, 8 Other
NAMES = list(KEEP.values())


def seq_group(stem: str) -> str:
    return re.sub(r"_\d+$", "", stem)


def trace(ids: np.ndarray, seg_id: int):
    ys, xs = np.nonzero(ids == seg_id)
    if xs.size == 0:
        return None, 0
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1], int(xs.size)


def load_split(lars: Path, split: str):
    pan = json.loads((lars / split / "panoptic_annotations.json").read_text())
    images = {im["id"]: im for im in pan["images"]}
    out = []
    for a in sorted(pan["annotations"], key=lambda a: a["file_name"]):
        im = images[a["image_id"]]
        rgb = np.asarray(Image.open(lars / split / "panoptic_masks" / a["file_name"]).convert("RGB")).astype(np.int64)
        ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
        objs = []
        for s in a["segments_info"]:
            if s["category_id"] not in KEEP:
                continue
            box, px = trace(ids, s["id"])
            if box is None:
                continue
            objs.append({"cat": s["category_id"], "seg_id": s["id"], "box": box, "stored": s["bbox"], "pixels": px,
                         "crowd": int(s.get("iscrowd", 0))})
        stem = Path(a["file_name"]).stem
        out.append({"stem": stem, "W": im["width"], "H": im["height"], "objs": objs, "mask": a["file_name"]})
    return out


def yolo_line(o, W, H) -> str:
    x, y, w, h = o["box"]
    return f"{YOLO_ID[o['cat']]} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}"


def write_split(name, recs, img_dir: Path, out: Path):
    (out / name / "images").mkdir(parents=True, exist_ok=True)
    (out / name / "labels").mkdir(parents=True, exist_ok=True)
    for r in recs:
        link = out / name / "images" / f"{r['stem']}.jpg"
        if not link.exists():
            link.symlink_to(img_dir / f"{r['stem']}.jpg")
        lines = [yolo_line(o, r["W"], r["H"]) for o in r["objs"] if not o["crowd"]]
        (out / name / "labels" / f"{r['stem']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def coco_gt(recs) -> dict:
    images, anns = [], []
    for i, r in enumerate(recs, 1):
        images.append({"id": i, "file_name": f"{r['stem']}.jpg", "width": r["W"], "height": r["H"]})
        for o in r["objs"]:
            x, y, w, h = o["box"]
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": YOLO_ID[o["cat"]], "bbox": [x, y, w, h],
                         "area": w * h, "iscrowd": o["crowd"]})
    return {"images": images, "annotations": anns,
            "categories": [{"id": YOLO_ID[c], "name": n} for c, n in KEEP.items()]}


def box_check(recs, lars: Path, split: str, out_png: Path) -> list[dict]:
    """Decode the written YOLO label lines back to pixels and draw them over the masks (5 examples)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    rules = [("Static Obstacle (whole-image region)", lambda o, r: o["cat"] == 1),
             ("large boat", lambda o, r: o["cat"] == 11 and o["box"][2] * o["box"][3] > 150 * 150),
             ("small buoy", lambda o, r: o["cat"] == 14 and o["box"][2] * o["box"][3] < 32 * 32 and min(o["box"][2:]) >= 6),
             ("float", lambda o, r: o["cat"] == 17),
             ("'Other' (debris-like)", lambda o, r: o["cat"] == 19)]
    picks, used = [], set()
    for tag, rule in rules:
        for r in recs:
            o = next((o for o in r["objs"] if not o["crowd"] and rule(o, r) and (r["stem"], o["seg_id"]) not in used), None)
            if o:
                picks.append((tag, r, o))
                used.add((r["stem"], o["seg_id"]))
                break
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 4.4))
    rows = []
    for ax, (tag, r, o) in zip(np.atleast_1d(axes), picks):
        img = np.asarray(Image.open(lars / split / "images" / f"{r['stem']}.jpg").convert("RGB"))
        rgb = np.asarray(Image.open(lars / split / "panoptic_masks" / r["mask"]).convert("RGB")).astype(np.int64)
        m = (rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]) == o["seg_id"]
        line = yolo_line(o, r["W"], r["H"])
        c, cx, cy, w, h = line.split()
        dw, dh = float(w) * r["W"], float(h) * r["H"]
        dx, dy = float(cx) * r["W"] - dw / 2, float(cy) * r["H"] - dh / 2
        err = max(abs(dx - o["box"][0]), abs(dy - o["box"][1]), abs(dw - o["box"][2]), abs(dh - o["box"][3]))
        rows.append({"example": tag, "image": f"{r['stem']}.jpg", "size": [r["W"], r["H"]], "class": KEEP[o["cat"]],
                     "traced_box_xywh": o["box"], "stored_box_xywh": o["stored"], "yolo_line": line,
                     "decoded_box_xywh": [round(dx, 2), round(dy, 2), round(dw, 2), round(dh, 2)], "max_roundtrip_error_px": round(err, 3)})
        ov = np.zeros((*m.shape, 4))
        ov[m] = (0, 0.72, 0.85, 0.5)
        ax.imshow(img)
        ax.imshow(ov)
        ax.add_patch(mp.Rectangle((dx - 0.5, dy - 0.5), dw, dh, fill=False, ec="#C81919", lw=2))
        pad = max(dw, dh) * 0.4 + 15
        ax.set_xlim(max(0, dx - pad), min(r["W"], dx + dw + pad))
        ax.set_ylim(min(r["H"], dy + dh + pad), max(0, dy - pad))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{tag}\n{r['stem']}\nYOLO: {line}", fontsize=8)
    fig.text(0.5, 0.01, "cyan = instance mask; red = box decoded back from the written YOLO label line", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_png, dpi=130)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lars", type=Path, default=Path("/scratch-shared/rcampanella/water-debris-seg/data/lars"))
    ap.add_argument("--out", type=Path, default=Path("/scratch-shared/rcampanella/water-debris-seg/data/lars/yolo_boxes"))
    ap.add_argument("--eval-dir", type=Path, default=Path(__file__).resolve().parents[2] / "eval_outputs/water_debris_seg/ablations/10_yolo_lars_boxes")
    ap.add_argument("--held-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--with-water-sky", action="store_true",
                    help="fairness fix 2 (YOLO + SAM): add Water (id 9) and Sky (id 10) boxes; ids 0-8 unchanged. "
                         "Outputs get the prefix lars11_ in --eval-dir/sources")
    args = ap.parse_args()
    global YOLO_ID, NAMES
    pre = ""
    if args.with_water_sky:
        KEEP.update({3: "Water", 5: "Sky"})
        YOLO_ID = {cat: i for i, cat in enumerate(KEEP)}
        NAMES = list(KEEP.values())
        pre = "lars11_"
    src = args.eval_dir / "sources"
    src.mkdir(parents=True, exist_ok=True)

    train, val = load_split(args.lars, "train"), load_split(args.lars, "val")
    groups = sorted({seq_group(r["stem"]) for r in train})
    random.Random(args.seed).shuffle(groups)
    held_groups = set(groups[: round(len(groups) * args.held_frac)])
    held = [r for r in train if seq_group(r["stem"]) in held_groups]
    train90 = [r for r in train if seq_group(r["stem"]) not in held_groups]

    for name, recs, split in [("train_all", train, "train"), ("train90", train90, "train"), ("held10", held, "train"), ("val", val, "val")]:
        write_split(name, recs, args.lars / split / "images", args.out)
    for name, tr, va in [("stageA", "train90", "held10"), ("stageB", "train_all", "held10"), ("eval_val", "train90", "val")]:
        (args.out / f"{name}.yaml").write_text(f"path: {args.out}\ntrain: {tr}/images\nval: {va}/images\nnc: {len(NAMES)}\nnames: {json.dumps(NAMES)}\n")
    (src / f"{pre}lars_val_gt_coco_boxes.json").write_text(json.dumps(coco_gt(val)))
    (src / f"{pre}lars_held10_gt_coco_boxes.json").write_text(json.dumps(coco_gt(held)))
    (src / f"{pre}lars_held10_sequences.txt").write_text("\n".join(sorted(held_groups)) + "\n")

    stats = {"splits": {}, "box_vs_stored_mismatch": 0, "max_diff_px": 0}
    for name, recs in [("train_all", train), ("train90", train90), ("held10", held), ("val", val)]:
        cnt, crowd, small, inside = Counter(), Counter(), 0, 0
        for r in recs:
            obs = [o for o in r["objs"] if o["cat"] == 1]
            for o in r["objs"]:
                (crowd if o["crowd"] else cnt)[KEEP[o["cat"]]] += 1
                if o["cat"] != 1 and not o["crowd"]:
                    small += o["box"][2] * o["box"][3] < 32 * 32
                    cx, cy = o["box"][0] + o["box"][2] / 2, o["box"][1] + o["box"][3] / 2
                    inside += any(b["box"][0] <= cx <= b["box"][0] + b["box"][2] and b["box"][1] <= cy <= b["box"][1] + b["box"][3] for b in obs)
                d = max(abs(a - b) for a, b in zip(o["box"], o["stored"]))
                stats["max_diff_px"] = max(stats["max_diff_px"], d)
                stats["box_vs_stored_mismatch"] += d > 0
        things = sum(v for k, v in cnt.items() if k != "Static Obstacle")
        stats["splits"][name] = {"images": len(recs), "sequences": len({seq_group(r['stem']) for r in recs}),
                                 "boxes_per_class": dict(cnt), "crowd_excluded_from_labels": dict(crowd),
                                 "object_boxes": things, "object_boxes_small_lt_32px": small,
                                 "object_boxes_centred_inside_a_static_obstacle_box": inside,
                                 "images_without_labels": sum(1 for r in recs if not any(not o['crowd'] for o in r['objs']))}
    stats["examples"] = box_check(val, args.lars, "val", src / f"{pre}box_check_examples.png")
    stats["class_index"] = {i: n for i, n in enumerate(NAMES)}
    (src / f"{pre}lars_yolo_label_stats.json").write_text(json.dumps(stats, indent=2))
    (src / f"{pre}box_check_examples.json").write_text(json.dumps(stats["examples"], indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "examples"}, indent=1))
    for e in stats["examples"]:
        print(e["example"], e["image"], e["traced_box_xywh"], "| stored", e["stored_box_xywh"], "| roundtrip err px", e["max_roundtrip_error_px"])


if __name__ == "__main__":
    main()
