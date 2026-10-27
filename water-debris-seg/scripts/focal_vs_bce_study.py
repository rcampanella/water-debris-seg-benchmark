#!/usr/bin/env python3
"""Focal loss vs BCE on LaRS val: where each loss puts its weight, and what that does to detections.

Implements ablations/10_yolo_lars_boxes/FOCAL_STUDY_PLAN.md (approved 2026-10-20). Inference only.

* 4.0  toy BCE vs focal numbers (natural log; the Medium post's log10 numbers reproduced) and loss curves
* 4.1  positive vs negative samples per image under YOLOv11's TaskAlignedAssigner, positives per class
* 4.2  per sample category (easy/hard x positive/negative): count, BCE and focal loss share, gradient share,
       mean loss, on the same predictions, for 4 stage-B checkpoints
* 4.3  TP/FP/FN at conf 0.25, TP vs FP score histograms, detection images and loss heatmaps for a fixed
       image selection (Float images, most small objects, most objects inside Static Obstacle, random seed 0)

A "sample" is one (grid location, class) pair: 8,400 x 9 per 640-px image. Targets t come from Ultralytics'
assigner run on each model's own predictions, exactly as in training. Outputs: <out>/study.json + figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "water-debris-seg"))
import water_debris_seg.yolo_focal  # noqa: E402,F401  (unpickle focal checkpoints)
import water_debris_seg.yolo_lora  # noqa: E402,F401  (unpickle LoRA checkpoints)
from water_debris_seg.yolo_focal import FocalBCE  # noqa: E402

ABL = REPO / "eval_outputs/water_debris_seg/ablations/10_yolo_lars_boxes"
DATA = Path("/scratch-shared/rcampanella/water-debris-seg/data/lars")
MODELS = {"full_bce": "lars_full_stageB", "full_focal": "lars_full_focal_stageB",
          "lora_bce": "lars_lora_stageB", "lora_focal": "lars_lora_focal_stageB"}
GAMMA, ALPHA = 1.5, 0.25
CATS = ["trivial_neg", "easy_neg", "hard_neg", "easy_pos", "hard_pos"]
CAT_LABEL = {"trivial_neg": "easy negative, p<0.01", "easy_neg": "easy negative, 0.01≤p<thr",
             "hard_neg": "hard negative (p≥thr)", "easy_pos": "easy positive (p≥thr)", "hard_pos": "hard positive (p<thr)"}
CAT_COLOR = {"trivial_neg": "#c6dbef", "easy_neg": "#6baed6", "hard_neg": "#d62728", "easy_pos": "#2ca02c", "hard_pos": "#ff7f0e"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 4.0 toy numbers
def toy_table() -> dict:
    def ce(p, y, log):
        return -log(p) if y == 1 else -log(1 - p)

    def fl(p, y, g, a, log):
        pt, at = (p, a) if y == 1 else (1 - p, 1 - a)
        return at * (1 - pt) ** g * ce(p, y, log)

    cases = [("easy positive", 0.9, 1), ("hard positive", 0.1, 1), ("easy negative", 0.1, 0),
             ("hard negative", 0.9, 0), ("trivial negative", 0.001, 0)]
    rows = []
    for name, p, y in cases:
        r = {"case": name, "p": p, "y": y, "bce": ce(p, y, math.log)}
        for g in (2.0, 1.5):
            r[f"focal_g{g}"] = fl(p, y, g, ALPHA, math.log)
            r[f"ratio_g{g}"] = r["bce"] / r[f"focal_g{g}"]
        r["post_log10_bce"] = ce(p, y, math.log10)
        r["post_log10_focal_g2"] = fl(p, y, 2.0, ALPHA, math.log10)
        rows.append(r)
    return {"alpha": ALPHA, "rows": rows,
            "note": "BCE/focal ratio = 1/(alpha_t (1-p_t)^gamma), independent of the log base. The post quotes focal "
                    "0.000125 and ratio 360 for the easy positive (computed: 0.000114, 400) and ratio 40 for the hard "
                    "positive (computed: 1/0.2025 = 4.9)."}


def plot_curves(out: Path) -> None:
    import matplotlib.pyplot as plt

    p = np.linspace(1e-3, 1 - 1e-3, 500)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y in zip(axes, (1, 0)):
        pt = p if y == 1 else 1 - p
        at = ALPHA if y == 1 else 1 - ALPHA
        bce = -np.log(pt)
        ax.plot(p, bce, "k-", lw=2, label="BCE")
        for g, ls in ((2.0, "--"), (1.5, "-")):
            ax.plot(p, at * (1 - pt) ** g * bce, ls, lw=2, label=f"focal γ={g}, α={ALPHA}" + (" (our runs)" if g == 1.5 else " (post)"))
        easy = (p >= 0.5) if y == 1 else (p < 0.5)
        ax.fill_between(p, 0, 5, where=easy, color="#2ca02c" if y == 1 else "#6baed6", alpha=0.12, label="easy region")
        ax.set_ylim(0, 5)
        ax.set_xlabel("predicted probability p")
        ax.set_ylabel("loss")
        ax.set_title("positive sample (target 1)" if y == 1 else "negative sample (target 0)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "loss_curves.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- data and models
class Recorder(nn.Module):
    """Stands in for v8DetectionLoss.bce: keeps the class logits and the assigner's soft targets."""

    def forward(self, x, t):
        self.x, self.t = x.detach().float(), t.detach().float()
        return F.binary_cross_entropy_with_logits(x.float(), t.float(), reduction="none")


def load(name: str, device):
    """Stage-B checkpoint, or 'init': the model as training starts (COCO yolo11n.pt + Ultralytics' fresh 9-class head)."""
    if name == "init":
        from ultralytics.nn.tasks import DetectionModel

        p = ABL / "runs/yolo11n.pt"
        m = DetectionModel("yolo11n.yaml", nc=9, verbose=False)   # Detect.bias_init() sets the class-bias prior here
        m.load(torch.load(p, map_location="cpu", weights_only=False)["model"])   # same transfer as the trainer
        return m.float().to(device).eval(), p
    p = ABL / "runs" / MODELS[name] / "weights/last.pt"
    m = torch.load(p, map_location="cpu", weights_only=False)["model"].float().to(device).eval()
    return m, p


def samples(m, batch, cfg):
    """Run the model and Ultralytics' assigner on one batch; return logits, targets (A, nc) and level shapes."""
    m.args = cfg
    m.criterion = m.init_criterion() if getattr(m, "criterion", None) is None else m.criterion
    rec = Recorder()
    m.criterion.bce = rec
    head = m.model[-1]
    head.training = True
    with torch.no_grad():
        preds = m(batch["img"])
        head.training = False
        m.criterion(preds, batch)
    feats = preds["feats"] if isinstance(preds, dict) else preds
    shapes = [tuple(f.shape[2:]) for f in feats]
    return rec.x[0].cpu(), rec.t[0].cpu(), shapes


def categorise(p, t, thr):
    pos = t > 0
    c = torch.empty(p.shape, dtype=torch.long)
    c[~pos & (p < 0.01)] = 0
    c[~pos & (p >= 0.01) & (p < thr)] = 1
    c[~pos & (p >= thr)] = 2
    c[pos & (p >= thr)] = 3
    c[pos & (p < thr)] = 4
    return c


def per_sample(x, t):
    """Element-wise BCE, focal, and |dL/dlogit| for both."""
    x = x.clone().requires_grad_(True)
    fl = FocalBCE(GAMMA, ALPHA)(x, t)
    (g_fl,) = torch.autograd.grad(fl.sum(), x)
    with torch.no_grad():
        bce = F.binary_cross_entropy_with_logits(x, t, reduction="none")
        g_bce = (x.sigmoid() - t).abs()
    return bce.detach(), fl.detach(), g_bce, g_fl.abs()


def level_map(v, shapes):
    """Per-anchor values (A,) -> stride-8 grid map, coarser levels upsampled and added."""
    out, i = None, 0
    for h, w in shapes:
        lv = v[i:i + h * w].view(1, 1, h, w)
        i += h * w
        lv = F.interpolate(lv, size=shapes[0], mode="nearest") / ((shapes[0][0] // h) * (shapes[0][1] // w))
        out = lv if out is None else out + lv
    return out[0, 0]


# ---------------------------------------------------------------- 4.3 helpers
def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def match(dets, gts):
    """Greedy per class at IoU 0.5. dets: [(bbox xywh, score, cls)]; gts: annotations. Detections on crowd GT are ignored."""
    status, used = [], set()
    for d in sorted(range(len(dets)), key=lambda i: -dets[i][1]):
        box, _, c = dets[d]
        best, bi = 0.5, None
        for gi, g in enumerate(gts):
            if g["category_id"] == c and not g["iscrowd"] and gi not in used:
                v = iou(box, g["bbox"])
                if v >= best:
                    best, bi = v, gi
        if bi is not None:
            used.add(bi)
            status.append((d, "TP"))
        elif any(g["category_id"] == c and g["iscrowd"] and iou(box, g["bbox"]) >= 0.5 for g in gts):
            status.append((d, "ignore"))
        else:
            status.append((d, "FP"))
    fn = [gi for gi, g in enumerate(gts) if not g["iscrowd"] and gi not in used]
    return dict(status), fn


def select_images(gt) -> list[dict]:
    anns = defaultdict(list)
    for a in gt["annotations"]:
        anns[a["image_id"]].append(a)
    ids = [im["id"] for im in gt["images"]]
    obj = lambda a: a["category_id"] != 0 and not a["iscrowd"]
    small = {i: sum(1 for a in anns[i] if obj(a) and a["bbox"][2] * a["bbox"][3] < 32 * 32) for i in ids}

    def inside(i):
        obs = [a["bbox"] for a in anns[i] if a["category_id"] == 0]
        n = 0
        for a in anns[i]:
            if obj(a):
                cx, cy = a["bbox"][0] + a["bbox"][2] / 2, a["bbox"][1] + a["bbox"][3] / 2
                n += any(o[0] <= cx <= o[0] + o[2] and o[1] <= cy <= o[1] + o[3] for o in obs)
        return n

    ins = {i: inside(i) for i in ids}
    chosen, picks = set(), []

    def take(group, ordered, k):
        n = 0
        for i in ordered:
            if n == k:
                break
            if i not in chosen:
                chosen.add(i)
                picks.append({"image_id": i, "group": group})
                n += 1

    take("float", [i for i in ids if any(a["category_id"] == 7 and not a["iscrowd"] for a in anns[i])], 99)
    take("most_small_objects", sorted(ids, key=lambda i: (-small[i], i)), 3)
    take("most_objects_inside_static_obstacle", sorted(ids, key=lambda i: (-ins[i], i)), 3)
    rest = [i for i in ids if i not in chosen]
    random.Random(0).shuffle(rest)
    take("random_seed0", rest, 3)
    for p in picks:
        p["n_small_objects"], p["n_objects_inside_obstacle"] = small[p["image_id"]], ins[p["image_id"]]
    return picks


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ABL / "focal_study")
    ap.add_argument("--limit", type=int, default=0, help="first N val images only (smoke test)")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    out = args.out
    (out / "examples").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gt_path = ABL / "sources/lars_val_gt_coco_boxes.json"
    gt = json.loads(gt_path.read_text())
    names = {c["id"]: c["name"] for c in gt["categories"]}
    nc = len(names)
    by_file = {im["file_name"]: im for im in gt["images"]}
    anns = defaultdict(list)
    for a in gt["annotations"]:
        anns[a["image_id"]].append(a)
    picks = select_images(gt)
    pick_ids = {p["image_id"] for p in picks}

    res = {"plan": "FOCAL_STUDY_PLAN.md", "device": str(device), "gamma": GAMMA, "alpha": ALPHA,
           "gt": str(gt_path), "gt_sha256": sha(gt_path), "toy": toy_table(), "checkpoints": {}, "selection": picks}
    plot_curves(out)

    data = check_det_dataset(str(DATA / "yolo_boxes/eval_val.yaml"))
    cfg = get_cfg()
    cfg.imgsz = 640
    ds = build_yolo_dataset(cfg, data["val"], 1, data, mode="val", rect=False, stride=32)
    dl = build_dataloader(ds, 1, 0, shuffle=False)

    imbalance = {"per_image_pos": [], "per_image_neg": [], "pos_per_class": defaultdict(int), "gt_boxes_per_class": defaultdict(int)}
    for a in gt["annotations"]:
        if not a["iscrowd"]:
            imbalance["gt_boxes_per_class"][names[a["category_id"]]] += 1
    tables, heat, letterboxed = {}, defaultdict(dict), {}
    for mname in ["init", *MODELS]:
        m, ck = load(mname, device)
        res["checkpoints"][mname] = {"path": str(ck), "sha256": sha(ck)}
        acc = {thr: {k: defaultdict(float) for k in CATS} for thr in (0.5, 0.25)}
        near_perfect_hard_pos = 0
        n_img = 0
        for idx, b in enumerate(dl):
            im = by_file.get(Path(b["im_file"][0]).name)
            if args.limit and idx >= args.limit and not (im and im["id"] in pick_ids):
                continue
            b["img"] = b["img"].to(device).float() / 255
            assert tuple(b["img"].shape[2:]) == (640, 640), b["img"].shape
            for k in ("cls", "bboxes", "batch_idx"):
                b[k] = b[k].to(device)
            x, t, shapes = samples(m, b, cfg)
            p = x.sigmoid()
            bce, fl, gb, gf = per_sample(x, t)
            for thr in (0.5, 0.25):
                c = categorise(p, t, thr)
                for k, cat in enumerate(CATS):
                    s = c == k
                    a = acc[thr][cat]
                    a["n"] += int(s.sum())
                    a["bce"] += float(bce[s].sum())
                    a["focal"] += float(fl[s].sum())
                    a["grad_bce"] += float(gb[s].sum())
                    a["grad_focal"] += float(gf[s].sum())
            hp = (t > 0) & (p < 0.5)
            near_perfect_hard_pos += int((hp & ((p - t).abs() < 0.1)).sum())
            if mname == "full_bce":
                npos = int((t > 0).sum())
                imbalance["per_image_pos"].append(npos)
                imbalance["per_image_neg"].append(t.numel() - npos)
                for ci in range(nc):
                    imbalance["pos_per_class"][names[ci]] += int((t[:, ci] > 0).sum())
            if im and im["id"] in pick_ids:
                heat[im["id"]][mname] = {"bce": level_map(bce.sum(1), shapes).numpy(), "focal": level_map(fl.sum(1), shapes).numpy()}
                letterboxed[im["id"]] = (b["img"][0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            n_img += 1
        tab = {}
        for thr in (0.5, 0.25):
            tot = {q: sum(acc[thr][k][q] for k in CATS) for q in ("n", "bce", "focal", "grad_bce", "grad_focal")}
            tab[str(thr)] = {k: {"n": int(acc[thr][k]["n"]),
                                 "pct_samples": 100 * acc[thr][k]["n"] / tot["n"],
                                 "pct_bce_loss": 100 * acc[thr][k]["bce"] / tot["bce"],
                                 "pct_focal_loss": 100 * acc[thr][k]["focal"] / tot["focal"],
                                 "pct_bce_grad": 100 * acc[thr][k]["grad_bce"] / tot["grad_bce"],
                                 "pct_focal_grad": 100 * acc[thr][k]["grad_focal"] / tot["grad_focal"],
                                 "mean_bce": acc[thr][k]["bce"] / max(acc[thr][k]["n"], 1),
                                 "mean_focal": acc[thr][k]["focal"] / max(acc[thr][k]["n"], 1),
                                 "bce_over_focal": acc[thr][k]["bce"] / max(acc[thr][k]["focal"], 1e-12)} for k in CATS}
        tab["n_images"] = n_img
        tab["hard_pos_within_0.1_of_target"] = near_perfect_hard_pos
        tables[mname] = tab
        print(f"[4.2] {mname}: {n_img} images", json.dumps({k: round(v['pct_bce_loss'], 2) for k, v in tab['0.5'].items()}), flush=True)
        del m
    pos, neg = np.array(imbalance["per_image_pos"]), np.array(imbalance["per_image_neg"])
    res["imbalance"] = {"model": "full_bce", "images": int(len(pos)), "samples_per_image": int(pos[0] + neg[0]),
                        "positives_per_image_mean": float(pos.mean()), "positives_per_image_median": float(np.median(pos)),
                        "negatives_per_image_mean": float(neg.mean()), "neg_to_pos_ratio": float(neg.sum() / max(pos.sum(), 1)),
                        "positives_per_class": dict(imbalance["pos_per_class"]), "gt_boxes_per_class": dict(imbalance["gt_boxes_per_class"])}
    res["loss_tables"] = tables

    # ---- figures: imbalance and loss share
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = ["negatives", "positives (all)"] + [names[i] for i in range(nc)]
    vals = [neg.sum(), pos.sum()] + [res["imbalance"]["positives_per_class"].get(names[i], 0) for i in range(nc)]
    ax.bar(labels, [max(v, 0.5) for v in vals], color=["#6baed6", "#2ca02c"] + ["#74c476"] * nc)
    ax.set_yscale("log")
    ax.set_ylabel("samples over LaRS val (log)")
    ax.set_title(f"Foreground vs background samples, YOLOv11n full-BCE ({len(pos)} images): 1 positive per {res['imbalance']['neg_to_pos_ratio']:.0f} negatives")
    for i, v in enumerate(vals):
        ax.text(i, max(v, 0.5) * 1.15, f"{int(v):,}", ha="center", fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out / "fg_bg_counts.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, mname in zip(axes, ("full_bce", "full_focal")):
        tab = tables[mname]["0.5"]
        cols = [("samples", "pct_samples"), ("BCE\nloss", "pct_bce_loss"), ("focal\nloss", "pct_focal_loss"),
                ("BCE\ngradient", "pct_bce_grad"), ("focal\ngradient", "pct_focal_grad")]
        bottom = np.zeros(len(cols))
        for k in CATS:
            v = np.array([tab[k][c] for _, c in cols])
            ax.bar([c for c, _ in cols], v, bottom=bottom, color=CAT_COLOR[k], label=CAT_LABEL[k].replace("thr", "0.5"))
            bottom += v
        ax.set_title(f"predictions of the {mname.replace('_', ' ').upper()} model")
        ax.set_ylabel("% of total")
    axes[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Where the classification loss and gradient come from (same predictions scored with BCE and with focal)")
    fig.tight_layout()
    fig.savefig(out / "loss_share.png", dpi=150)
    plt.close(fig)

    # ---- 4.3 detections
    det = {}
    for mname in MODELS:
        model = YOLO(str(ABL / "runs" / MODELS[mname] / "weights/last.pt"))
        counts = {names[i]: {"TP": 0, "FP": 0, "FN": 0} for i in range(nc)}
        scores = {"TP": [], "FP": []}
        per_img = {}
        ims = [im for k, im in enumerate(gt["images"]) if not args.limit or k < args.limit or im["id"] in pick_ids]
        for im in ims:
            r = model.predict(str(DATA / "val/images" / im["file_name"]), imgsz=640, conf=0.05, max_det=300, verbose=False)[0]
            dets = [((x1, y1, x2 - x1, y2 - y1), s, int(c)) for (x1, y1, x2, y2), s, c in
                    zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
            st, _ = match(dets, anns[im["id"]])
            for d, s in st.items():
                if s in ("TP", "FP"):
                    scores[s].append(dets[d][1])
            d25 = [d for d in dets if d[1] >= 0.25]
            st25, fn25 = match(d25, anns[im["id"]])
            for d, s in st25.items():
                if s in ("TP", "FP"):
                    counts[names[d25[d][2]]][s] += 1
            for gi in fn25:
                counts[names[anns[im["id"]][gi]["category_id"]]]["FN"] += 1
            if im["id"] in pick_ids:
                per_img[im["id"]] = (d25, st25, fn25)
        tot = {k: sum(v[k] for v in counts.values()) for k in ("TP", "FP", "FN")}
        det[mname] = {"conf": 0.25, "iou": 0.5, "per_class": counts, "total": tot,
                      "precision": tot["TP"] / max(tot["TP"] + tot["FP"], 1), "recall": tot["TP"] / max(tot["TP"] + tot["FN"], 1),
                      "score_quantiles": {s: [float(q) for q in np.quantile(v, [0.25, 0.5, 0.75])] if v else [] for s, v in scores.items()},
                      "_scores": scores, "_per_img": per_img}
        print(f"[4.3] {mname}: {tot}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    bins = np.linspace(0.05, 1, 39)
    for ax, mname in zip(axes, ("full_bce", "full_focal")):
        for s, col in (("TP", "#2ca02c"), ("FP", "#d62728")):
            ax.hist(det[mname]["_scores"][s], bins=bins, color=col, alpha=0.6, label=f"{s} ({len(det[mname]['_scores'][s])})")
        ax.axvline(0.25, color="k", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("detection confidence")
        ax.set_title(mname.replace("_", " ").upper())
        ax.legend()
    axes[0].set_ylabel("detections (log)")
    fig.suptitle("Confidence of correct (TP) vs false (FP) detections, LaRS val, IoU ≥ 0.5")
    fig.tight_layout()
    fig.savefig(out / "score_hist.png", dpi=150)
    plt.close(fig)

    def draw(ax, img, d25, st, fn, gts, title):
        import matplotlib.patches as pa

        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        for i, (bx, s, c) in enumerate(d25):
            col = {"TP": "#2ca02c", "FP": "#d62728"}.get(st.get(i), "#999999")
            ax.add_patch(pa.Rectangle(bx[:2], bx[2], bx[3], fill=False, ec=col, lw=1.2))
            ax.text(bx[0], bx[1] - 2, f"{names[c]} {s:.2f}", color=col, fontsize=5)
        for gi in fn:
            g = gts[gi]["bbox"]
            ax.add_patch(pa.Rectangle(g[:2], g[2], g[3], fill=False, ec="#ff7f0e", lw=1.2, ls="--"))

    def _example(pk, iid):
        im = next(i for i in gt["images"] if i["id"] == iid)
        img = np.array(Image.open(DATA / "val/images" / im["file_name"]).convert("RGB"))
        gts = anns[iid]
        fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
        import matplotlib.patches as pa

        ax = axes[0, 0]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"ground truth ({pk['group']})", fontsize=9)
        for g in gts:
            b_ = g["bbox"]
            ax.add_patch(pa.Rectangle(b_[:2], b_[2], b_[3], fill=False, ec="#1f77b4", lw=1.2, ls=":" if g["iscrowd"] else "-"))
            ax.text(b_[0], b_[1] - 2, names[g["category_id"]], color="#1f77b4", fontsize=5)
        for ax, mname in ((axes[0, 1], "full_bce"), (axes[0, 2], "full_focal")):
            d25, st, fn = det[mname]["_per_img"][iid]
            ntp = sum(1 for v in st.values() if v == "TP")
            nfp = sum(1 for v in st.values() if v == "FP")
            draw(ax, img, d25, st, fn, gts, f"{mname.replace('_', ' ').upper()}: TP {ntp}  FP {nfp}  missed {len(fn)}  (conf ≥ 0.25)")
        lb = letterboxed[iid]
        hb = heat[iid]["full_bce"]
        for ax, key, title in ((axes[1, 0], "bce", "BCE loss, full-BCE predictions"),
                               (axes[1, 1], "focal", "focal loss, same predictions")):
            h = hb[key] / hb[key].sum()
            ax.imshow(lb)
            ax.imshow(np.kron(h, np.ones((8, 8))), cmap="inferno", alpha=0.6,
                      norm=matplotlib.colors.LogNorm(vmin=max(h.max() * 1e-6, 1e-12), vmax=h.max()))
            ax.axis("off")
            ax.set_title(f"{title}: share per location (log)", fontsize=9)
        hf = heat[iid]["full_focal"]["focal"]
        h = hf / hf.sum()
        ax = axes[1, 2]
        ax.imshow(lb)
        ax.imshow(np.kron(h, np.ones((8, 8))), cmap="inferno", alpha=0.6,
                  norm=matplotlib.colors.LogNorm(vmin=max(h.max() * 1e-6, 1e-12), vmax=h.max()))
        ax.axis("off")
        ax.set_title("focal loss, full-FOCAL predictions: share per location (log)", fontsize=9)
        fig.suptitle(f"{im['file_name']}   green TP · red FP · orange dashed = missed GT", fontsize=10)
        fig.tight_layout()
        fname = f"{pk['group']}_{Path(im['file_name']).stem}.png"
        fig.savefig(out / "examples" / fname, dpi=130)
        plt.close(fig)
        pk["figure"] = f"examples/{fname}"
        pk["counts"] = {m_: {"TP": sum(1 for v in det[m_]["_per_img"][iid][1].values() if v == "TP"),
                             "FP": sum(1 for v in det[m_]["_per_img"][iid][1].values() if v == "FP"),
                             "missed": len(det[m_]["_per_img"][iid][2])} for m_ in MODELS}
        pk["file_name"] = im["file_name"]

    def _numbers_only():
        d = {m_: {k: v for k, v in det[m_].items() if not k.startswith("_")} for m_ in det}
        return {**res, "detections": d}

    (out / "study.json").write_text(json.dumps(_numbers_only(), indent=2))   # numbers first; figures may still fail
    for pk in picks:
        iid = pk["image_id"]
        if iid not in heat or iid not in det["full_bce"]["_per_img"]:
            pk["figure"] = None
            continue
        try:
            _example(pk, iid)
        except Exception as e:  # keep going; record the failure in study.json
            import traceback

            traceback.print_exc()
            pk["figure_error"] = repr(e)

    for mname in det:
        det[mname].pop("_scores")
        det[mname].pop("_per_img")
    res["detections"] = det
    (out / "study.json").write_text(json.dumps(res, indent=2))
    print("wrote", out / "study.json")


if __name__ == "__main__":
    main()
