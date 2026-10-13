#!/usr/bin/env python3
"""Render mask overlays from qual_dump_masks.py output.

boxsets  For each selected FloW/CANSURF image: input + GT boxes | each EoMT run's mask with the
         boxes derived from it (box colour = error status from qual_error_analysis). Shows *why*
         a mask→box detection merged, fragmented or missed.
lars     LaRS val: per-image obstacle IoU (ab02 semantic) and thing recall (ab03 panoptic) vs GT,
         a per-image CSV, and panels for the worst images plus examples of each hard attribute
         (glitter, fog, night, heavy reflections, wakes, dirty lens, plants/debris).

Writes under eval_outputs/water_debris_seg/qualitative/:
  mask_panels/<dataset>/<NN>_<category>_<stem>.png
  lars_val_per_image.csv, MASKS.md (index with selection rule + provenance per panel)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "eval_outputs" / "water_debris_seg"
Q = EVAL / "qualitative"
LARS = Path("/scratch-shared/rcampanella/water-debris-seg/data/lars/val")
STATUS_COL = {"tp": "#2CA02C", "fp_bkg": "#D62728", "fp_loc": "#FF9896", "fp_dup": "#E377C2", "fp_merge": "#9467BD"}
# LaRS model class index → (name, colour); panoptic order from eomt_adapters/lars_panoptic.py
LARS_PAN = ["static obstacle", "water", "sky", "boat/ship", "row boat", "paddle board", "buoy", "swimmer",
            "animal", "float", "other"]
SEM = {0: ("obstacle", (1.0, 0.83, 0.1)), 1: ("water", (0.27, 0.96, 1.0)), 2: ("sky", (0.67, 0.0, 1.0))}
HARD_FLAGS = ["glitter", "fog", "extra_dark", "extra_bright", "dirty_lens", "plants_debris", "wakes"]


def seg_overlay(ax, img: np.ndarray, seg: np.ndarray, cls: np.ndarray, keep, alpha=0.55, seed=0):
    """Colour each kept segment (random colour per segment id) over the image."""
    ax.imshow(img)
    rgba = np.zeros((*seg.shape, 4))
    rng = np.random.default_rng(seed)
    for sid in np.unique(seg):
        if sid < 0:
            continue
        m = seg == sid
        c = int(cls[m][0])
        if not keep(c):
            continue
        rgba[m] = (*rng.uniform(0.2, 1.0, 3), alpha)
    ax.imshow(rgba)


def sem_overlay(ax, img, cls_map, classes=(0,), alpha=0.5):
    ax.imshow(img)
    rgba = np.zeros((*cls_map.shape, 4))
    for c in classes:
        rgba[cls_map == c] = (*SEM[c][1], alpha)
    ax.imshow(rgba)


def boxes(ax, rows, lw=1.4, color=None, ls="-"):
    for r in rows:
        x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, lw=lw, ls=ls,
                                        ec=color or STATUS_COL.get(r["status"], "white")))


def load_csv(path: Path) -> dict[int, list[dict]]:
    out = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            out[int(r["image_id"])].append(r)
    return out


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def render_boxsets(index: list[str], dpi: int) -> None:
    registry = json.loads((EVAL / "box_eval_registry.json").read_text())
    sel = json.loads((Q / "selection.json").read_text())
    runs = {r["id"]: r for r in registry["runs"]}
    by_ds = defaultdict(list)
    for p in sel["picks"]:
        by_ds[p["dataset"]].append(p)
    for ds, picks in by_ds.items():
        eomt = [r for r in sel["meta"]["runs"] if sel["meta"]["runs"][r]["dataset"] == ds and "model" in runs[r]]
        cons = {r: json.loads((Q / "masks" / ds / r / "consistency.json").read_text()) for r in eomt}
        dets = {r: load_csv(Q / "errors" / r / "detections.csv") for r in eomt}
        gts = load_csv(Q / "errors" / eomt[0] / "gt.csv")
        img_dir = Path(registry["ground_truth"][ds]["images"])
        out_dir = Q / "mask_panels" / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        index += [f"## {ds} — masks behind the boxes", "",
                  "Mask consistency (boxes re-derived from the dumped masks vs the scored prediction file): " +
                  ", ".join(f"`{r}` {cons[r]['n_identical_box_sets']}/{cons[r]['n_images']}" for r in eomt), "",
                  "| # | Category | Image | Panel |", "|---|---|---|---|"]
        for k, p in enumerate(picks, 1):
            stem = Path(p["file_name"]).stem
            img = np.asarray(Image.open(img_dir / p["file_name"]).convert("RGB"))
            H, W = img.shape[:2]
            cols = 1 + len(eomt)
            cw = 3.8
            fig, axes = plt.subplots(1, cols, figsize=(cw * cols, cw * H / W + 1.3), squeeze=False,
                                     gridspec_kw={"wspace": 0.03})
            ax = axes[0][0]
            ax.imshow(img)
            boxes(ax, gts.get(p["image_id"], []), color="#00B8D9")
            ax.set_title(f"input + GT ({len(gts.get(p['image_id'], []))})", fontsize=9)
            clean(ax)
            for c, r in enumerate(eomt, 1):
                ax = axes[0][c]
                kind = runs[r]["model"]["kind"]
                m = np.load(Q / "masks" / ds / r / f"{stem}.npz")
                if kind.startswith("panoptic"):
                    stuff = set(range(80, 133)) if kind == "panoptic_coco133" else {0, 1, 2}
                    seg_overlay(ax, img, m["seg_map"], m["class_map"], keep=lambda c, s=stuff: c >= 0 and c not in s)
                    what = "thing segments"
                else:
                    sem_overlay(ax, img, m["class_map"], classes=(0,))
                    what = "obstacle mask"
                boxes(ax, dets[r].get(p["image_id"], []))
                ok = next(x["identical"] for x in cons[r]["images"] if x["image_id"] == p["image_id"])
                ax.set_title(f"{runs[r]['method']}: {what}\nboxes = scored dets{'' if ok else ' (⚠ re-derived differ)'}",
                             fontsize=8)
                clean(ax)
            fig.suptitle(f"{ds} · {p['file_name']} · {p['category']}", fontsize=9)
            prov = "\n".join(f"{r}: {runs[r]['model']['ckpt']} → {runs[r]['pred']}" for r in eomt)
            fig.text(0.005, 0.005, prov, fontsize=4.8, color="#555555", va="bottom")
            fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.12)
            out = out_dir / f"{k:02d}_{p['category'].split('__')[0]}_{stem[:40]}.png"
            fig.savefig(out, dpi=dpi)
            plt.close(fig)
            index.append(f"| {k} | {p['category']} | `{img_dir / p['file_name']}` | [{out.name}](mask_panels/{ds}/{out.name}) |")
        index.append("")


def lars_gt(stem: str, pan_ann: dict, cat_to_idx: dict):
    """GT semantic map (0/1/2/255) and list of (mask, class_idx) for thing segments."""
    sem = np.asarray(Image.open(LARS / "semantic_masks" / f"{stem}.png"))
    rgb = np.asarray(Image.open(LARS / "panoptic_masks" / f"{stem}.png").convert("RGB")).astype(np.int64)
    ids = rgb[..., 0] + 256 * rgb[..., 1] + 256 * 256 * rgb[..., 2]
    things = [(ids == s["id"], cat_to_idx[s["category_id"]]) for s in pan_ann["segments_info"]
              if cat_to_idx.get(s["category_id"], 0) >= 3]
    return sem, things


def render_lars(index: list[str], dpi: int, n_worst: int) -> None:
    pan = json.loads((LARS / "panoptic_annotations.json").read_text())
    ann_by_file = {Path(a["file_name"]).stem: a for a in pan["annotations"]}
    attrs = {Path(a.get("file_name", a.get("image_name", ""))).stem: a.get("labels", {})
             for a in json.loads((LARS / "image_annotations.json").read_text())["annotations"]}
    cat_to_idx = {1: 0, 3: 1, 5: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7, 16: 8, 17: 9, 19: 10}
    sem_dir, pan_dir = Q / "masks" / "lars_val" / "semantic_lars", Q / "masks" / "lars_val" / "panoptic_lars"
    rows, thing_rows = [], []
    for npz in sorted(sem_dir.glob("*.npz")):
        stem = npz.stem
        pred_sem = np.load(npz)["class_map"]
        gt_sem, gt_things = lars_gt(stem, ann_by_file[stem], cat_to_idx)
        valid = gt_sem != 255
        iou = {}
        for c, (name, _) in SEM.items():
            p, g = (pred_sem == c) & valid, (gt_sem == c) & valid
            u = (p | g).sum()
            iou[name] = float((p & g).sum() / u) if u else float("nan")
        pm = np.load(pan_dir / f"{stem}.npz")
        seg, cls = pm["seg_map"], pm["class_map"]
        pred_things = [(seg == s, int(cls[seg == s][0])) for s in np.unique(seg) if s >= 0 and int(cls[seg == s][0]) >= 3]
        found = 0
        for gm, gc in gt_things:
            best = max((float((gm & pm_).sum() / max((gm | pm_).sum(), 1)) for pm_, _ in pred_things), default=0.0)
            found += best > 0.5
            area = int(gm.sum())
            thing_rows.append({"stem": stem, "class": LARS_PAN[gc], "area_px": area,
                               "size": "small" if area < 32**2 else "medium" if area < 96**2 else "large",
                               "best_iou": round(best, 3), "found_iou50": best > 0.5})
        lab = attrs.get(stem, {})
        flags = [f for f, v in (lab.get("special") or {}).items() if v]
        rows.append({"stem": stem, "obstacle_iou": iou["obstacle"], "water_iou": iou["water"], "sky_iou": iou["sky"],
                     "gt_things": len(gt_things), "things_found_iou50": found,
                     "thing_recall": found / len(gt_things) if gt_things else float("nan"),
                     "reflections": lab.get("reflections"), "waves": lab.get("waves"),
                     "lighting": lab.get("lighting"), "flags": ";".join(flags)})
    with (Q / "lars_val_per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (Q / "lars_val_things.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(thing_rows[0]))
        w.writeheader()
        w.writerows(thing_rows)

    picks = []
    for r in sorted(rows, key=lambda r: r["obstacle_iou"])[:n_worst]:
        picks.append((r, f"worst obstacle IoU (ab02 semantic): {r['obstacle_iou']:.3f}"))
    ranked_things = sorted((r for r in rows if r["gt_things"] >= 2), key=lambda r: (r["thing_recall"], -r["gt_things"]))
    for r in ranked_things[:n_worst]:
        picks.append((r, f"lowest thing recall (ab03 panoptic): {r['things_found_iou50']}/{r['gt_things']}"))
    for flag in HARD_FLAGS:
        cand = sorted((r for r in rows if flag in r["flags"].split(";")), key=lambda r: r["obstacle_iou"])
        if cand:
            picks.append((cand[0], f"attribute `{flag}`: worst obstacle IoU among {len(cand)} flagged images"))
    seen, uniq = set(), []
    for r, why in picks:
        if r["stem"] not in seen:
            seen.add(r["stem"])
            uniq.append((r, why))

    out_dir = Q / "mask_panels" / "lars_val"
    out_dir.mkdir(parents=True, exist_ok=True)
    pal = plt.get_cmap("tab20")
    index += ["## lars_val — LaRS fine-tuned models vs pixel GT", "",
              f"Per-image table: [`lars_val_per_image.csv`](lars_val_per_image.csv) (198 images). Masks: "
              f"`masks/lars_val/{{semantic_lars,panoptic_lars}}/*.npz` (models: see `provenance.json` there). "
              "GT: `/scratch-shared/rcampanella/water-debris-seg/data/lars/val/{semantic_masks,panoptic_masks}`.", "",
              "| # | Rule | Image | Obstacle IoU | Things found | Panel |", "|---|---|---|---|---|---|"]
    for k, (r, why) in enumerate(uniq, 1):
        stem = r["stem"]
        img = np.asarray(Image.open(LARS / "images" / f"{stem}.jpg").convert("RGB"))
        H, W = img.shape[:2]
        gt_sem, gt_things = lars_gt(stem, ann_by_file[stem], cat_to_idx)
        pred_sem = np.load(sem_dir / f"{stem}.npz")["class_map"]
        pm = np.load(pan_dir / f"{stem}.npz")
        fig, axes = plt.subplots(1, 4, figsize=(3.8 * 4, 3.8 * H / W + 1.3), gridspec_kw={"wspace": 0.03})
        sem_overlay(axes[0], img, np.where(gt_sem == 255, 99, gt_sem), classes=(0, 1, 2), alpha=0.45)
        axes[0].set_title("GT semantic (obstacle / water / sky)", fontsize=8)
        sem_overlay(axes[1], img, pred_sem, classes=(0, 1, 2), alpha=0.45)
        err = (pred_sem != gt_sem) & (gt_sem != 255)
        axes[1].contour(err, levels=[0.5], colors="red", linewidths=0.6)
        axes[1].set_title(f"ab02 semantic pred (red = wrong px) · obstacle IoU {r['obstacle_iou']:.2f}", fontsize=8)
        axes[2].imshow(img)
        for i, (gm, c) in enumerate(gt_things):
            axes[2].imshow(np.dstack([*([np.full(gm.shape, v) for v in pal(i % 20)[:3]]), gm * 0.6]))
        axes[2].set_title(f"GT things ({len(gt_things)})", fontsize=8)
        seg_overlay(axes[3], img, pm["seg_map"], pm["class_map"], keep=lambda c: c >= 3)
        for s in np.unique(pm["seg_map"]):
            if s < 0:
                continue
            m = pm["seg_map"] == s
            c = int(pm["class_map"][m][0])
            if c >= 3:
                ys, xs = np.nonzero(m)
                axes[3].text(xs.mean(), ys.min(), LARS_PAN[c], fontsize=5.5, color="white", ha="center", clip_on=True,
                             bbox={"facecolor": "black", "alpha": 0.5, "pad": 0.3, "lw": 0})
        axes[3].set_title(f"ab03 panoptic things · found {r['things_found_iou50']}/{r['gt_things']} (IoU>0.5)", fontsize=8)
        for ax in axes:
            clean(ax)
        meta = f"reflections={r['reflections']} waves={r['waves']} lighting={r['lighting']} flags={r['flags'] or '-'}"
        fig.suptitle(f"lars_val · {stem} · {why}\n{meta}", fontsize=9)
        fig.text(0.005, 0.01, f"image {LARS / 'images' / (stem + '.jpg')} | masks qualitative/masks/lars_val/*/{stem}.npz",
                 fontsize=5, color="#555555")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.06)
        out = out_dir / f"{k:02d}_{stem}.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)
        index.append(f"| {k} | {why} | `{stem}.jpg` | {r['obstacle_iou']:.3f} | {r['things_found_iou50']}/{r['gt_things']} | "
                     f"[{out.name}](mask_panels/lars_val/{out.name}) |")
    index.append("")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["boxsets", "lars", "all"], nargs="?", default="all")
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--n-worst", type=int, default=4)
    args = parser.parse_args()
    index = ["# Mask panels", "", "Generated by `water-debris-seg/scripts/qual_render_masks.py` from masks dumped by "
             "`qual_dump_masks.py` (benchmarked inference code, configs in `box_eval_registry.json` → `model`).", ""]
    if args.mode in ("boxsets", "all"):
        render_boxsets(index, args.dpi)
    if args.mode in ("lars", "all"):
        render_lars(index, args.dpi, args.n_worst)
    (Q / "MASKS.md").write_text("\n".join(index) + "\n")
    print(f"wrote {Q / 'MASKS.md'}")


if __name__ == "__main__":
    main()
