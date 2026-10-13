#!/usr/bin/env python3
"""Render GT-vs-methods error panels for the images in qualitative/selection.json.

Each panel: one column for GT, one per method (TP / FP / FN colour-coded, FP types labelled),
plus a zoomed row on the region that triggered the pick. A provenance footer names the image
file, GT file and every prediction file with its sha256 prefix, so each figure can be traced
back to the exact outputs it shows.

Writes under eval_outputs/water_debris_seg/qualitative/:
  panels/<dataset>/<NN>_<category>_<stem>.png
  error_breakdown_<dataset>.png     GT outcome shares per method (stacked bars)
  INDEX.md                          every panel with its selection rule, stats and provenance
Run qual_error_analysis.py first.
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
from PIL import Image  # noqa: E402

COL = {
    "gt": "#00B8D9", "tp": "#2CA02C", "fp_bkg": "#D62728", "fp_loc": "#FF9896", "fp_dup": "#E377C2",
    "fp_merge": "#9467BD", "fn_missed": "#FF7F0E", "fn_loc": "#FFBB78", "fn_frag": "#8C564B", "fn_merged": "#C5B0D5",
}
LABEL = {"fp_bkg": "bkg", "fp_loc": "loc", "fp_dup": "dup", "fp_merge": "merge"}
LEGEND = [
    ("tp", "TP (IoU≥0.5)"), ("fp_bkg", "FP background"), ("fp_loc", "FP poor loc."), ("fp_dup", "FP duplicate"),
    ("fp_merge", "FP merge (≥2 objects)"), ("fn_missed", "FN missed"), ("fn_loc", "FN poor loc."),
    ("fn_merged", "FN merged"), ("fn_frag", "FN fragmented"),
]
FOCUS = {"merge": ("gt", {"fn_merged"}), "missed_small": ("gt", {"fn_missed"}), "bkg_fp": ("det", {"fp_bkg"}),
         "fragment": ("gt", {"fn_frag"})}


ZOOM = 4  # smallest zoom window = 1/ZOOM of the image; grows to fit ~4× the focus boxes


def zoom_size(boxes: list[dict], img_w: int, img_h: int) -> tuple[int, int]:
    """Window keeps the image aspect; wide enough for ~4× the median focus box, max half the image."""
    if not boxes:
        return img_w // ZOOM, img_h // ZOOM
    sides = sorted(max(b["w"], b["h"] * img_w / img_h) for b in boxes)
    w = min(max(img_w / ZOOM, 4 * sides[len(sides) // 2]), img_w / 2)
    return int(w), int(w * img_h / img_w)


def load_rows(path: Path) -> dict[int, list[dict]]:
    out = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            for k in ("x", "y", "w", "h"):
                r[k] = float(r[k])
            out[int(r["image_id"])].append(r)
    return out


def focus_window(boxes: list[dict], img_w: int, img_h: int, win_w: int, win_h: int):
    """Window of size win_w × win_h containing the most box centres (coarse grid search)."""
    if not boxes:
        return None
    cs = [(b["x"] + b["w"] / 2, b["y"] + b["h"] / 2) for b in boxes]
    best, best_n = None, -1
    step_x, step_y = max(win_w // 4, 1), max(win_h // 4, 1)
    for x0 in range(0, max(img_w - win_w, 0) + 1, step_x):
        for y0 in range(0, max(img_h - win_h, 0) + 1, step_y):
            n = sum(x0 <= cx < x0 + win_w and y0 <= cy < y0 + win_h for cx, cy in cs)
            if n > best_n:
                best, best_n = (x0, y0, x0 + win_w, y0 + win_h), n
    # re-centre on the centres the best window captured, clamped to the image
    x0, y0, x1, y1 = best
    inside = [(cx, cy) for cx, cy in cs if x0 <= cx < x1 and y0 <= cy < y1]
    mx = sum(c[0] for c in inside) / len(inside)
    my = sum(c[1] for c in inside) / len(inside)
    x0 = min(max(mx - win_w / 2, 0), img_w - win_w)
    y0 = min(max(my - win_h / 2, 0), img_h - win_h)
    return (x0, y0, x0 + win_w, y0 + win_h)


def draw(ax, img, gts, dets, *, gt_only: bool, window=None, lw=1.6, labels=True):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    if gt_only:
        for g in gts:
            ax.add_patch(mpatches.Rectangle((g["x"], g["y"]), g["w"], g["h"], fill=False, ec=COL["gt"], lw=lw))
    else:
        for g in gts:
            if g["status"] != "tp":
                ax.add_patch(mpatches.Rectangle((g["x"], g["y"]), g["w"], g["h"], fill=False,
                                                ec=COL[g["status"]], lw=lw, ls=(0, (3, 2))))
        for d in dets:
            ax.add_patch(mpatches.Rectangle((d["x"], d["y"]), d["w"], d["h"], fill=False, ec=COL[d["status"]], lw=lw))
            if labels and d["status"] in LABEL:
                ax.text(d["x"], d["y"] - 2, LABEL[d["status"]], color="white", fontsize=6, clip_on=True,
                        bbox={"facecolor": COL[d["status"]], "pad": 0.5, "lw": 0})
    if window:
        x0, y0, x1, y1 = window
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)


def counts(gts, dets) -> str:
    tp = sum(g["status"] == "tp" for g in gts)
    fp = sum(d["status"] != "tp" for d in dets)
    return f"TP {tp} · FP {fp} · FN {len(gts) - tp}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = Path(__file__).resolve().parents[2] / "eval_outputs" / "water_debris_seg"
    parser.add_argument("--eval-root", type=Path, default=default_root)
    parser.add_argument("--dpi", type=int, default=110)
    args = parser.parse_args()
    root = args.eval_root
    qroot = root / "qualitative"
    registry = json.loads((root / "box_eval_registry.json").read_text())
    sel = json.loads((qroot / "selection.json").read_text())
    run_ids = list(sel["meta"]["runs"])
    summaries = {r: json.loads((qroot / "errors" / r / "summary.json").read_text()) for r in run_ids}
    dets = {r: load_rows(qroot / "errors" / r / "detections.csv") for r in run_ids}
    gts = {r: load_rows(qroot / "errors" / r / "gt.csv") for r in run_ids}

    index = ["# Qualitative error panels", "",
             "Generated by `water-debris-seg/scripts/qual_render_panels.py` from [`selection.json`](selection.json) "
             "(picked by `qual_error_analysis.py`). Error definitions: `water-debris-seg/water_debris_seg/qualitative/errors.py`; "
             "per-box tables: `errors/<run_id>/{detections,gt}.csv`; counts: [`error_summary.csv`](error_summary.csv).", "",
             "Colours: TP green · FP background red · FP poor localisation pink · FP duplicate magenta · FP merge purple · "
             "FN (dashed) missed orange · poorly localised light orange · merged lilac · fragmented brown. "
             f"Operating point: YOLO score ≥ {sel['meta']['yolo_min_score']}; EoMT all boxes (fixed score 0.5).", ""]

    by_ds = defaultdict(list)
    for p in sel["picks"]:
        by_ds[p["dataset"]].append(p)

    for ds, picks in by_ds.items():
        ds_runs = [r for r in run_ids if sel["meta"]["runs"][r]["dataset"] == ds]
        gt_entry = registry["ground_truth"][ds]
        img_dir = Path(gt_entry["images"])
        out_dir = qroot / "panels" / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        prov = [f"GT {gt_entry['path']} ({summaries[ds_runs[0]]['provenance']['gt_sha256'][:10]})"]
        prov += [f"{r}: {summaries[r]['provenance']['pred']} ({summaries[r]['provenance']['pred_sha256'][:10]}, "
                 f"min_score {summaries[r]['provenance']['min_score']})" for r in ds_runs]
        index += [f"## {ds}", "", "**Provenance (all panels below):**", ""] + [f"- `{x}`" for x in prov] + [""]
        index += ["| # | Category | Rule | Stats | Image | Panel |", "|---|---|---|---|---|---|"]

        for k, p in enumerate(picks, 1):
            img_id, fname = p["image_id"], p["file_name"]
            img = Image.open(img_dir / fname).convert("RGB")
            W, H = img.size
            cat = p["category"].split("__")[0]
            trig = p["stats"].get("run") if "run" in p["stats"] else None
            window = None
            if cat in FOCUS and trig:
                kind, statuses = FOCUS[cat]
                rows = (gts if kind == "gt" else dets)[trig].get(img_id, [])
                focus = [b for b in rows if b["status"] in statuses]
                window = focus_window(focus, W, H, *zoom_size(focus, W, H))
            elif cat != "random":
                focus = [g for g in gts[ds_runs[0]].get(img_id, []) if float(g["area"]) < 32 * 32]
                window = focus_window(focus, W, H, *zoom_size(focus, W, H))

            n_rows = 2 if window else 1
            cols = 1 + len(ds_runs)
            cell_w = 3.6
            cell_h = cell_w * H / W
            fig, axes = plt.subplots(n_rows, cols, figsize=(cell_w * cols, cell_h * n_rows + 1.6), squeeze=False,
                                     gridspec_kw={"wspace": 0.03, "hspace": 0.12})
            g0 = gts[ds_runs[0]].get(img_id, [])
            for row in range(n_rows):
                win = window if row == 1 else None
                draw(axes[row][0], img, g0, [], gt_only=True, window=win, lw=1.2 if row == 0 else 2)
                if row == 0:
                    axes[row][0].set_title(f"GT ({len(g0)} boxes)", fontsize=9)
                for c, r in enumerate(ds_runs, 1):
                    gr, dr = gts[r].get(img_id, []), dets[r].get(img_id, [])
                    draw(axes[row][c], img, gr, dr, gt_only=False, window=win, lw=1.2 if row == 0 else 2,
                         labels=row == 1 or len(dr) < 25)
                    if row == 0:
                        star = " ★" if r == trig else ""
                        axes[row][c].set_title(f"{sel['meta']['runs'][r]['method']}{star}\n{counts(gr, dr)}", fontsize=8)
                if row == 1:
                    axes[row][0].set_ylabel("zoom", fontsize=8)
            if window:
                x0, y0, x1, y1 = window
                axes[0][0].add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=1, ls="--"))
            handles = [mpatches.Patch(color=COL[s], label=t) for s, t in LEGEND]
            fig_h = cell_h * n_rows + 1.6
            fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.75 / fig_h, bottom=0.6 / fig_h)
            fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=7, frameon=False,
                       bbox_to_anchor=(0.5, 0.28 / fig_h))
            fig.suptitle(f"{ds} · {fname} (image_id {img_id}) · {p['category']}\n{p['rule']}", fontsize=9,
                         y=1 - 0.08 / fig_h)
            half = (len(prov) + 1) // 2
            fig.text(0.005, 0.03 / fig_h, " | ".join(prov[:half]) + "\n" + " | ".join(prov[half:]),
                     fontsize=5, color="#555555", ha="left", va="bottom")
            out = out_dir / f"{k:02d}_{p['category'].split('__')[0]}_{Path(fname).stem[:40]}.png"
            fig.savefig(out, dpi=args.dpi)
            plt.close(fig)
            stats = ", ".join(f"{a}={b}" for a, b in p["stats"].items())
            index.append(f"| {k} | {p['category']} | {p['rule']} | {stats} | `{img_dir / fname}` | "
                         f"[{out.name}](panels/{ds}/{out.name}) |")
            print(out.relative_to(root))
        index.append("")

        # GT outcome shares per method
        statuses = ["tp", "fn_merged", "fn_frag", "fn_loc", "fn_missed"]
        fig, ax = plt.subplots(figsize=(8, 0.6 + 0.55 * len(ds_runs)))
        for y, r in enumerate(ds_runs):
            g = summaries[r]["gt"]
            tot = sum(sum(v.values()) for v in g.values())
            left = 0.0
            for s in statuses:
                v = sum(g[s].values()) / tot
                ax.barh(y, v, left=left, color=COL[s], edgecolor="white")
                if v > 0.06:
                    ax.text(left + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=8, color="black")
                left += v
        ax.set_yticks(range(len(ds_runs)), [sel["meta"]["runs"][r]["method"] for r in ds_runs], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of GT boxes")
        ax.set_title(f"{ds}: what happens to each GT box (IoU 0.5)", fontsize=10)
        ax.legend(handles=[mpatches.Patch(color=COL[s], label=s) for s in statuses], ncol=5, fontsize=7,
                  loc="upper center", bbox_to_anchor=(0.5, -0.35), frameon=False)
        fig.tight_layout()
        fig.savefig(qroot / f"error_breakdown_{ds}.png", dpi=args.dpi)
        plt.close(fig)
        index += [f"![{ds} error breakdown](error_breakdown_{ds}.png)", ""]

    (qroot / "INDEX.md").write_text("\n".join(index) + "\n")
    print(f"wrote {qroot / 'INDEX.md'}")


if __name__ == "__main__":
    main()
