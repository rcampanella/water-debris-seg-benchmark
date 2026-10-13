#!/usr/bin/env python3
"""Slide-sized figures for the qualitative review deck, plus every number the deck shows.

Figures reuse the drawing code of water-debris-seg/scripts/qual_render_{panels,masks}.py on
the same audited data (errors/<run>/*.csv, masks/*.npz); they differ only in column choice and
font size. Numbers are read from the analysis outputs, never typed by hand.

Writes docs/presentation/qual_assets/:
  fig_*.png        one figure per slide
  deck_data.json   numbers + per-figure provenance (source files, sha256 prefixes)
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
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
OUT = Path(__file__).resolve().parent / "qual_assets"
sys.path.insert(0, str(REPO / "water-debris-seg" / "scripts"))
from qual_render_masks import LARS, SEM, boxes, lars_gt, seg_overlay, sem_overlay  # noqa: E402
from qual_render_panels import COL, LEGEND, draw, focus_window, load_rows, zoom_size  # noqa: E402

plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "font.family": "DejaVu Sans"})
REG = json.loads((EVAL / "box_eval_registry.json").read_text())
RUNS = {r["id"]: r for r in REG["runs"]}
SHORT = {"flow_eomt_zeroshot": "EoMT zero-shot", "flow_eomt_semantic": "EoMT semantic FT",
         "flow_eomt_panoptic": "EoMT panoptic FT", "flow_yolo11n": "YOLOv11n",
         "cansurf_eomt_zeroshot": "EoMT zero-shot", "cansurf_eomt_semantic_aligned": "EoMT semantic FT",
         "cansurf_eomt_panoptic": "EoMT panoptic FT", "cansurf_yolo11n": "YOLOv11n"}
CAT_TO_IDX = {1: 0, 3: 1, 5: 2, 11: 3, 12: 4, 13: 5, 14: 6, 15: 7, 16: 8, 17: 9, 19: 10}
manifest: dict[str, dict] = {}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:10]


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def image_of(ds: str, image_id: int) -> tuple[Path, str]:
    gt = json.loads((EVAL / REG["ground_truth"][ds]["path"]).read_text())
    fn = next(i["file_name"] for i in gt["images"] if i["id"] == image_id)
    return Path(REG["ground_truth"][ds]["images"]) / fn, fn


def counts(gr, dr) -> str:
    tp = sum(g["status"] == "tp" for g in gr)
    return f"TP {tp} · FP {sum(d['status'] != 'tp' for d in dr)} · FN {len(gr) - tp}"


def legend(fig, y=0.0):
    keep = ["tp", "fp_bkg", "fp_loc", "fp_merge", "fn_missed", "fn_loc", "fn_merged"]
    h = [mpatches.Patch(color=COL[s], label=t) for s, t in LEGEND if s in keep]
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=12, frameon=False, bbox_to_anchor=(0.5, y))


def box_figure(name: str, ds: str, image_id: int, runs: list[str], zoom: str | None, zoom_run: str | None = None):
    """GT + chosen runs; optional zoom row centred on `zoom` statuses of `zoom_run`."""
    img_path, fn = image_of(ds, image_id)
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    gts = {r: load_rows(Q / "errors" / r / "gt.csv").get(image_id, []) for r in runs}
    dets = {r: load_rows(Q / "errors" / r / "detections.csv").get(image_id, []) for r in runs}
    window = None
    if zoom:
        src = gts[zoom_run] if zoom.startswith("fn") else dets[zoom_run]
        focus = [b for b in src if b["status"] in zoom.split(",")]
        window = focus_window(focus, W, H, *zoom_size(focus, W, H))
    rows, cols = (2 if window else 1), 1 + len(runs)
    cw = 16 / cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, cw * H / W * rows + 1.4), squeeze=False,
                             gridspec_kw={"wspace": 0.03, "hspace": 0.06})
    g0 = gts[runs[0]]
    for r_i in range(rows):
        win = window if r_i == 1 else None
        draw(axes[r_i][0], img, g0, [], gt_only=True, window=win, lw=1.6 if r_i == 0 else 2.6)
        for c, r in enumerate(runs, 1):
            draw(axes[r_i][c], img, gts[r], dets[r], gt_only=False, window=win, lw=1.6 if r_i == 0 else 2.6,
                 labels=False)
            if r_i == 0:
                axes[r_i][c].set_title(f"{SHORT[r]}\n{counts(gts[r], dets[r])}", fontsize=14)
        if r_i == 0:
            axes[r_i][0].set_title(f"Ground truth\n{len(g0)} objects", fontsize=14)
    if window:
        x0, y0, x1, y1 = window
        axes[0][0].add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=2, ls="--"))
    legend(fig)
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.75 / fig.get_figheight(), bottom=0.45 / fig.get_figheight())
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)
    manifest[name] = {
        "image": rel(img_path), "dataset": ds, "image_id": image_id,
        "sources": [f"qualitative/errors/{r}/{{gt,detections}}.csv" for r in runs],
        "preds": {r: f"{RUNS[r]['pred']} (sha {json.loads((Q / 'errors' / r / 'summary.json').read_text())['provenance']['pred_sha256'][:10]})" for r in runs},
        "footer": f"{ds} · {fn} · boxes from qualitative/errors/<run>/*.csv (pycocotools matching, IoU 0.5)",
    }


def mask_figure(name: str, ds: str, image_id: int, runs: list[str], with_yolo: str | None = None):
    img_path, fn = image_of(ds, image_id)
    img = np.asarray(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    stem = Path(fn).stem
    cols = 1 + len(runs) + (1 if with_yolo else 0)
    cw = 16 / cols
    fig, axes = plt.subplots(1, cols, figsize=(16, cw * H / W + 1.4), squeeze=False, gridspec_kw={"wspace": 0.03})
    gts = load_rows(Q / "errors" / runs[0] / "gt.csv").get(image_id, [])
    ax = axes[0][0]
    ax.imshow(img)
    boxes(ax, gts, color="#00B8D9", lw=1.8)
    ax.set_title(f"Ground truth\n{len(gts)} objects", fontsize=14)
    for c, r in enumerate(runs, 1):
        ax = axes[0][c]
        kind = RUNS[r]["model"]["kind"]
        m = np.load(Q / "masks" / ds / r / f"{stem}.npz")
        if kind.startswith("panoptic"):
            stuff = set(range(80, 133)) if kind == "panoptic_coco133" else {0, 1, 2}
            seg_overlay(ax, img, m["seg_map"], m["class_map"], keep=lambda k, s=stuff: k >= 0 and k not in s)
            what = "object masks"
        else:
            sem_overlay(ax, img, m["class_map"], classes=(0,))
            what = "'obstacle' mask (yellow)"
        dr = load_rows(Q / "errors" / r / "detections.csv").get(image_id, [])
        boxes(ax, dr, lw=2.2)
        ax.set_title(f"{SHORT[r]}: {what}\n+ boxes derived from it", fontsize=14)
    if with_yolo:
        ax = axes[0][-1]
        gr = load_rows(Q / "errors" / with_yolo / "gt.csv").get(image_id, [])
        dr = load_rows(Q / "errors" / with_yolo / "detections.csv").get(image_id, [])
        draw(ax, Image.fromarray(img), gr, dr, gt_only=False, lw=1.8, labels=False)
        ax.set_title(f"{SHORT[with_yolo]} (reference)\n{counts(gr, dr)}", fontsize=14)
    for a in axes[0]:
        a.set_xticks([])
        a.set_yticks([])
    legend(fig)
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.8 / fig.get_figheight(), bottom=0.45 / fig.get_figheight())
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)
    cons = {r: next(x["identical"] for x in json.loads((Q / "masks" / ds / r / "consistency.json").read_text())["images"]
                    if x["image_id"] == image_id) for r in runs}
    manifest[name] = {"image": rel(img_path), "dataset": ds, "image_id": image_id,
                      "masks": [f"qualitative/masks/{ds}/{r}/{stem}.npz" for r in runs],
                      "masks_reproduce_scored_boxes": cons,
                      "footer": f"{ds} · {fn} · masks qualitative/masks/{ds}/<run>/{stem}.npz (re-derive scored boxes: {'yes' if all(cons.values()) else 'NO'})"}


def lars_figure(name: str, stem: str, zoom_things: bool, stacked: bool = False):
    img = np.asarray(Image.open(LARS / "images" / f"{stem}.jpg").convert("RGB"))
    H, W = img.shape[:2]
    pan = json.loads((LARS / "panoptic_annotations.json").read_text())
    ann = next(a for a in pan["annotations"] if Path(a["file_name"]).stem == stem)
    gt_sem, gt_things = lars_gt(stem, ann, CAT_TO_IDX)
    pred_sem = np.load(Q / "masks/lars_val/semantic_lars" / f"{stem}.npz")["class_map"]
    pm = np.load(Q / "masks/lars_val/panoptic_lars" / f"{stem}.npz")
    window = None
    if zoom_things and gt_things:
        bxs = []
        for gm, _ in gt_things:
            ys, xs = np.nonzero(gm)
            bxs.append({"x": xs.min(), "y": ys.min(), "w": xs.max() - xs.min() + 1, "h": ys.max() - ys.min() + 1})
        window = focus_window(bxs, W, H, *zoom_size(bxs, W, H))
    # two panels: semantic errors (full frame) and panoptic objects vs GT (zoomed on GT objects)
    if stacked:
        fig, axes = plt.subplots(2, 1, figsize=(8, 2 * 8 * H / W + 1.1), gridspec_kw={"hspace": 0.16})
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8 * H / W + 0.7), gridspec_kw={"wspace": 0.03})
    sem_overlay(axes[0], img, pred_sem, classes=(0, 1, 2), alpha=0.45)
    axes[0].contour((pred_sem != gt_sem) & (gt_sem != 255), levels=[0.5], colors="red", linewidths=1.0)
    axes[0].set_title("Semantic FT: obstacle / water / sky, red = wrong pixels", fontsize=15)
    axes[1].imshow(img)
    for gm, _ in gt_things:
        ys, xs = np.nonzero(gm)
        axes[1].add_patch(mpatches.Rectangle((xs.min() - 2, ys.min() - 2), xs.max() - xs.min() + 4, ys.max() - ys.min() + 4,
                                             fill=False, ec="#00B8D9", lw=1.8))
    seg_overlay(axes[1], img, pm["seg_map"], pm["class_map"], keep=lambda c: c >= 3, alpha=0.7)
    axes[1].set_title("Panoptic FT objects (filled) vs GT objects (cyan)" + (", zoomed" if window else ""), fontsize=15)
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    if window:
        x0, y0, x1, y1 = window
        axes[1].set_xlim(x0, x1)
        axes[1].set_ylim(y1, y0)
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.5 / fig.get_figheight(), bottom=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)
    manifest[name] = {"image": rel(LARS / "images" / f"{stem}.jpg"), "dataset": "lars_val",
                      "masks": [f"qualitative/masks/lars_val/{k}/{stem}.npz" for k in ("semantic_lars", "panoptic_lars")],
                      "footer": f"LaRS val · {stem}.jpg · masks qualitative/masks/lars_val/*/{stem}.npz · GT LaRS semantic/panoptic PNGs"
                                + (" · object panel zoomed on GT objects" if window else "")}


def numbers() -> dict:
    d: dict = {}
    es = {r["run_id"]: r for r in csv.DictReader((Q / "error_summary.csv").open())}
    outcome = {}
    for rid, r in es.items():
        n = int(r["n_gt"])
        sums = {k: sum(int(r[f"gt_{k}_{z}"]) for z in ("small", "medium", "large")) for k in
                ("tp", "fn_merged", "fn_frag", "fn_loc", "fn_missed")}
        outcome[rid] = {"dataset": r["dataset"], "method": SHORT[rid], "n_gt": n,
                        **{k: round(v / n, 4) for k, v in sums.items()}, "recall50": float(r["recall_iou50"])}
    d["outcome"] = outcome
    d["det"] = {rid: {k.removeprefix("det_"): int(v) for k, v in r.items() if k.startswith("det_")} for rid, r in es.items()}
    # small-object LaRS example: how much of the GT object area the semantic model calls obstacle
    stem = "inhouse_seq_191_00399"
    pan = json.loads((LARS / "panoptic_annotations.json").read_text())
    ann = next(a for a in pan["annotations"] if Path(a["file_name"]).stem == stem)
    _, things = lars_gt(stem, ann, CAT_TO_IDX)
    ps = np.load(Q / "masks/lars_val/semantic_lars" / f"{stem}.npz")["class_map"]
    px = sum(int(m.sum()) for m, _ in things)
    d["lars_small_example"] = {"stem": stem, "objects": len(things),
                               "obstacle_px_share": round(sum(int((m & (ps == 0)).sum()) for m, _ in things) / px, 3),
                               "objects_touched": sum(int((m & (ps == 0)).sum() > 0) for m, _ in things),
                               "found_panoptic": next(int(r["things_found_iou50"]) for r in csv.DictReader((Q / "lars_val_per_image.csv").open()) if r["stem"] == stem)}
    rs = {r["id"]: r["new"] for r in json.loads((EVAL / "RESCORE.json").read_text())["runs"]}
    d["box_metrics"] = {rid: {k: round(rs[rid][k], 4) for k in ("mAP", "AP50", "AP75", "AP_small", "AR100")} for rid in SHORT}
    z = json.loads((EVAL / "ablations/01_zeroshot_lars/sources/pq_zeroshot_predremap_28011322.json").read_text())
    f = json.loads((EVAL / "ablations/01_zeroshot_lars/sources/pq_native_ab03_check_28011322.json").read_text())
    d["lars_pq"] = {"zeroshot": {k: round(v["pq"], 3) for k, v in z["per_class"].items()},
                    "finetuned": {k: round(v["pq"], 3) for k, v in f["per_class"].items()},
                    "all": {"zeroshot": round(z["pq_all"], 3), "finetuned": round(f["pq_all"], 3)}}
    things = list(csv.DictReader((Q / "lars_val_things.csv").open()))
    by = defaultdict(lambda: [0, 0])
    byc = defaultdict(lambda: [0, 0])
    for t in things:
        by[t["size"]][0] += 1
        by[t["size"]][1] += t["found_iou50"] == "True"
        byc[t["class"]][0] += 1
        byc[t["class"]][1] += t["found_iou50"] == "True"
    d["lars_things_by_size"] = {k: {"n": v[0], "found": v[1]} for k, v in by.items()}
    d["lars_things_by_class"] = {k: {"n": v[0], "found": v[1]} for k, v in byc.items()}
    rows = list(csv.DictReader((Q / "lars_val_per_image.csv").open()))

    def agg(sel):
        s = [r for r in rows if sel(r)]
        oi = [float(r["obstacle_iou"]) for r in s if r["obstacle_iou"] != "nan"]
        g = sum(int(r["gt_things"]) for r in s)
        fd = sum(int(r["things_found_iou50"]) for r in s)
        return {"n": len(s), "obstacle_iou": round(statistics.mean(oi), 3), "things": g, "found": fd}

    cond = {"all images": agg(lambda r: True)}
    for flag in ["glitter", "fog", "wakes", "extra_dark", "extra_bright"]:
        cond[flag.replace("_", " ")] = agg(lambda r, fl=flag: fl in r["flags"].split(";"))
    cond["dusk"] = agg(lambda r: r["lighting"] == "dusk_like")
    cond["night"] = agg(lambda r: r["lighting"] == "night_like")
    d["lars_conditions"] = cond
    gt_sizes = {}
    for ds in ("flow_test", "cansurf_val_aligned"):
        a = [x["bbox"][2] * x["bbox"][3] for x in json.loads((EVAL / REG["ground_truth"][ds]["path"]).read_text())["annotations"]]
        gt_sizes[ds] = {"n": len(a), "small": sum(x < 1024 for x in a)}
    d["gt_sizes"] = gt_sizes
    d["consistency"] = sum(json.loads(p.read_text())["n_identical_box_sets"] for p in Q.glob("masks/*/*/consistency.json"))
    d["consistency_total"] = sum(json.loads(p.read_text())["n_images"] for p in Q.glob("masks/*/*/consistency.json"))
    d["sources"] = {
        "outcome": "eval_outputs/water_debris_seg/qualitative/error_summary.csv",
        "box_metrics": "eval_outputs/water_debris_seg/RESCORE.json (sha " + sha(EVAL / "RESCORE.json") + ")",
        "lars_pq": "ablations/01_zeroshot_lars/sources/pq_{zeroshot_predremap,native_ab03_check}_28011322.json",
        "lars_things": "eval_outputs/water_debris_seg/qualitative/lars_val_things.csv",
        "lars_conditions": "eval_outputs/water_debris_seg/qualitative/lars_val_per_image.csv",
    }
    return d


def main() -> None:
    OUT.mkdir(exist_ok=True)
    F, C = "flow_test", "cansurf_val_aligned"
    fl = ["flow_eomt_zeroshot", "flow_eomt_semantic", "flow_eomt_panoptic", "flow_yolo11n"]
    ca = ["cansurf_eomt_zeroshot", "cansurf_eomt_semantic_aligned", "cansurf_eomt_panoptic", "cansurf_yolo11n"]
    box_figure("fig_random_flow", F, 42, fl, "fn_missed,fn_loc,tp", "flow_eomt_zeroshot")
    box_figure("fig_missed_flow", F, 657, ["flow_eomt_zeroshot", "flow_eomt_panoptic", "flow_yolo11n"],
               "fn_missed", "flow_eomt_zeroshot")
    mask_figure("fig_merge_flow_mask", F, 212, ["flow_eomt_semantic"], with_yolo="flow_yolo11n")
    box_figure("fig_merge_flow_zoom", F, 212, ["flow_eomt_semantic", "flow_yolo11n"], "fn_merged", "flow_eomt_semantic")
    cans = json.loads((EVAL / REG["ground_truth"][C]["path"]).read_text())
    cid = {i["file_name"]: i["id"] for i in cans["images"]}
    dog = next(k for k in cid if k.startswith("375_dog_jpg.rf.9ebf22f3"))
    mask_figure("fig_merge_cansurf_mask", C, cid[dog], ["cansurf_eomt_zeroshot", "cansurf_eomt_semantic_aligned"],
                with_yolo="cansurf_yolo11n")
    mask_figure("fig_shore_flow_mask", F, 778, ["flow_eomt_semantic"], with_yolo="flow_yolo11n")
    pool = next(k for k in cid if k.startswith("frame_0000_IMG_3619"))
    box_figure("fig_yolo_fail_cansurf", C, cid[pool], ["cansurf_yolo11n", "cansurf_eomt_semantic_aligned"], None)
    lars_figure("fig_lars_small", "inhouse_seq_191_00399", zoom_things=True)
    lars_figure("fig_lars_glitter", "inhouse_seq_344_00113", zoom_things=True, stacked=True)
    data = numbers()
    data["figures"] = manifest
    (OUT / "deck_data.json").write_text(json.dumps(data, indent=2))
    print("figures:", ", ".join(manifest))


if __name__ == "__main__":
    main()
