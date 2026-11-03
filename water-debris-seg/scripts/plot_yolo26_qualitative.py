#!/usr/bin/env python3
"""Ablation 11 qualitative figures: LaRS panoptic and CANSURF box predictions of YOLO26 models and the main-table EoMT,
from saved outputs only.

No inference is run. Sources:
  LaRS     data/water_debris/lars_eval/preds/ab11_lars11_<model>/<image>.png (panoptic PNGs scored by the official
           evaluator; R = LaRS category id, G * 256 + B = instance id) and the GT panoptic masks; per-image PQ from
           <ablation>/lars_official/ab11_lars11_<model>/frames.csv; EoMT: preds/<EOMT_LARS>/ and
           fairness_fixes/lars_official/<EOMT_LARS>/frames.csv
  CANSURF  sources/cansurf_paper_ab11_<model>_<job>.preds.json (all boxes at conf >= 0.001; drawn at conf >= CONF,
           the CANSURF paper threshold) and our CANSURF val GT. Boxes are matched greedily (score order, IoU >= 0.5):
           green = matched prediction, red = false positive, yellow dashed = missed GT. EoMT: EOMT_CANSURF_PREDS (the
           boxes its CANSURF paper metric was computed from), drawn at the same CONF
EoMT = the main-table run (fix 4 full fine-tuning, 12 epochs, held-out epoch selection); YOLO26 picks best.pt on the
scored split (ablation 11), so the two do not share a selection protocol.
Images: --n images drawn with random.Random(--seed) (LaRS: among val images with at least one GT object; CANSURF:
among images whose file name matches --cansurf-scene, default the sea frames), so the selection is not hand-picked.
CANSURF has no scene labels; by file name its val split is: frame_*_zahra_cans (608, shallow sea at a beach; the last
frames of that video are on the sand), frame_*_IMG / frame_*_mostafa / IMG* (pools), sprite* and WhatsApp* (cans on a
table / floor), <n>_<theme>_jpg (synthetic composites on theme backgrounds). Writes <ablation>/qualitative/{lars,cansurf}_seed<seed>.png and a JSON with the image
list, models and source files.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
ABL = REPO / "eval_outputs/water_debris_seg/ablations/11_yolo26_family"
LARS = REPO / "data/water_debris/lars_eval"
CANSURF_GT = REPO / "eval_outputs/water_debris_seg/ablations/05_cansurf_small_object/sources/cansurf_val_gt_coco_yolo_aligned.json"
CANSURF_IMG = REPO / "data/water_debris/cansurf_home/val/images"
CONF = 0.25
CANSURF_SCENES = {"sea": r"^frame_\d+_zahra_cans", "all": r""}
FF = REPO / "eval_outputs/water_debris_seg/fairness_fixes"
EOMT_LARS = "eomt_full"
EOMT_CANSURF_PREDS = FF / "sources/fix1_box_instance_cansurf_eomt_full_28547307.preds.json"
STUFF = {1: (0.55, 0.45, 0.25), 3: (0.15, 0.45, 0.85), 5: (0.75, 0.85, 1.0)}   # Static Obstacle, Water, Sky
THING_NAMES = {11: "Boat/ship", 12: "Row boats", 13: "Paddle board", 14: "Buoy", 15: "Swimmer", 16: "Animal",
               17: "Float", 19: "Other"}


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(REPO))


def overlay(img: np.ndarray, pan: np.ndarray) -> np.ndarray:
    """Stuff tinted by class, each thing instance a distinct colour, void left as the image."""
    out = img.astype(np.float32) / 255
    cat, inst = pan[..., 0], pan[..., 1].astype(np.int64) * 256 + pan[..., 2]
    for c, col in STUFF.items():
        m = cat == c
        out[m] = 0.55 * out[m] + 0.45 * np.array(col)
    rng = np.random.default_rng(0)
    for c in THING_NAMES:
        for i in np.unique(inst[cat == c]):
            m = (cat == c) & (inst == i)
            out[m] = 0.35 * out[m] + 0.65 * rng.uniform(0.2, 1.0, 3) * np.array([1.0, 0.5, 0.5])   # warm colours
    return out


def lars(args) -> dict:
    # column label -> (prediction PNG dir, official-evaluator frames.csv)
    models = {(m if m.endswith("-seg") else f"{m} + SAM 2.1-L"): (LARS / f"preds/ab11_lars11_{m}", ABL / f"lars_official/ab11_lars11_{m}/frames.csv")
              for m in args.lars_models}
    models["EoMT (main table)"] = (LARS / f"preds/{EOMT_LARS}", FF / f"lars_official/{EOMT_LARS}/frames.csv")
    frames = {m: {r["image"]: r for r in csv.DictReader(open(f))} for m, (_, f) in models.items()}
    names = (LARS / "val/image_list.txt").read_text().split()
    with_things = [n for n in names if np.isin(np.array(Image.open(LARS / f"val/panoptic_masks/{n}.png"))[..., 0], list(THING_NAMES)).any()]
    pick = sorted(random.Random(args.seed).sample(with_things, args.n))
    cols = ["Image", "Ground truth"] + list(models)
    fig, axes = plt.subplots(len(pick), len(cols), figsize=(3.2 * len(cols), 2.0 * len(pick)))
    for r, n in enumerate(pick):
        img = np.array(Image.open(LARS / f"val/images/{n}.jpg").convert("RGB"))
        panels = [img / 255, overlay(img, np.array(Image.open(LARS / f"val/panoptic_masks/{n}.png").convert("RGB")))]
        panels += [overlay(img, np.array(Image.open(d / f"{n}.png").convert("RGB"))) for d, _ in models.values()]
        for c, p in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(p)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=10)
            if c >= 2:
                f = frames[cols[c]][n]
                ax.set_xlabel(f"PQ {float(f['PQ']):.0f}  things {float(f['PQ_th']):.0f}  stuff {float(f['PQ_st']):.0f}", fontsize=8)
        axes[r, 0].set_ylabel(n, fontsize=7)
    handles = [mpatches.Patch(color=STUFF[1], label="Static Obstacle"), mpatches.Patch(color=STUFF[3], label="Water"),
               mpatches.Patch(color=STUFF[5], label="Sky"), mpatches.Patch(color=(0.9, 0.4, 0.4), label="objects (one colour per instance)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"LaRS val: {args.n} random images with objects (seed {args.seed}); per-image official-evaluator PQ (%) under each prediction",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    dst = ABL / f"qualitative/lars_seed{args.seed}.png"
    fig.savefig(dst, dpi=args.dpi)
    plt.close(fig)
    return {"figure": rel(dst), "images": pick, "models": list(models),
            "sources": {m: {"pred_pngs": rel(d), "frames": rel(f)} for m, (d, f) in models.items()}}


def iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(a[0], b[0])), max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def match(preds, gts):
    used, ok = set(), []
    for p in sorted(preds, key=lambda p: -p["score"]):
        best = max(((iou(p["bbox"], g["bbox"]), j) for j, g in enumerate(gts) if j not in used), default=(0, None))
        hit = best[0] >= 0.5
        if hit:
            used.add(best[1])
        ok.append((p, hit))
    return ok, [g for j, g in enumerate(gts) if j not in used]


def cansurf(args) -> dict:
    gt = json.load(open(CANSURF_GT))
    gts = {}
    for a in gt["annotations"]:
        gts.setdefault(a["image_id"], []).append(a)
    ims = {i["id"]: i for i in gt["images"]}
    pool = sorted(i for i, im in ims.items() if re.match(CANSURF_SCENES[args.cansurf_scene], im["file_name"]))
    pick = sorted(random.Random(args.seed).sample(pool, args.n))
    preds, srcs = {}, {}
    files = {m: Path(sorted(glob.glob(str(ABL / f"sources/cansurf_paper_ab11_{m}_*.preds.json")))[-1]) for m in args.cansurf_models}
    files["EoMT (main table)"] = EOMT_CANSURF_PREDS
    for m, f in files.items():
        srcs[m] = rel(f)
        preds[m] = {}
        for p in json.load(open(f)):
            if p["score"] >= CONF:
                preds[m].setdefault(p["image_id"], []).append(p)
    cols = ["Ground truth"] + list(files)
    fig, axes = plt.subplots(len(pick), len(cols), figsize=(3.4 * len(cols), 3.4 * len(pick)))
    for r, iid in enumerate(pick):
        img = Image.open(CANSURF_IMG / ims[iid]["file_name"]).convert("RGB")
        g = gts.get(iid, [])
        for c, name in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=11)
            if c == 0:
                for a in g:
                    x, y, w, h = a["bbox"]
                    ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec="cyan", lw=1.2))
                ax.set_xlabel(f"{len(g)} GT objects", fontsize=8)
                ax.set_ylabel(f"image id {iid}", fontsize=8)
                continue
            ok, missed = match(preds[name].get(iid, []), g)
            for p, hit in ok:
                x, y, w, h = p["bbox"]
                ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec="lime" if hit else "red", lw=1.2))
            for a in missed:
                x, y, w, h = a["bbox"]
                ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec="yellow", lw=1.2, ls="--"))
            tp = sum(h for _, h in ok)
            ax.set_xlabel(f"TP {tp}  FP {len(ok) - tp}  missed {len(missed)}", fontsize=8)
    handles = [mpatches.Patch(fill=False, ec="cyan", label="GT"), mpatches.Patch(fill=False, ec="lime", label="prediction matched (IoU ≥ 0.5)"),
               mpatches.Patch(fill=False, ec="red", label="false positive"), mpatches.Patch(fill=False, ec="yellow", ls="--", label="missed GT")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    scene = "sea frames (frame_*_zahra_cans)" if args.cansurf_scene == "sea" else "all"
    fig.suptitle(f"CANSURF val, {scene}: {args.n} random images (seed {args.seed}); boxes at conf ≥ {CONF}", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    dst = ABL / f"qualitative/cansurf_{args.cansurf_scene}_seed{args.seed}.png"
    fig.savefig(dst, dpi=args.dpi)
    plt.close(fig)
    return {"figure": rel(dst), "image_ids": pick, "file_names": [ims[i]["file_name"] for i in pick], "models": list(files),
            "conf": CONF, "scene": args.cansurf_scene, "scene_regex": CANSURF_SCENES[args.cansurf_scene], "n_candidates": len(pool), "sources": {"gt": rel(CANSURF_GT), **srcs}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--lars-models", nargs="+", default=["yolo26l", "yolo26m-seg"])
    ap.add_argument("--cansurf-models", nargs="+", default=["yolo26n", "yolo26x"])
    ap.add_argument("--cansurf-scene", choices=sorted(CANSURF_SCENES), default="sea")
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()
    (ABL / "qualitative").mkdir(exist_ok=True)
    out = {"lars": lars(args), "cansurf": cansurf(args)}
    (ABL / f"qualitative/qualitative_seed{args.seed}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
