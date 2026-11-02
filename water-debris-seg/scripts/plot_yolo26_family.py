#!/usr/bin/env python3
"""Ablation 11 plots: accuracy vs end-to-end latency for the YOLO26 / YOLO26-seg family, one figure per dataset.

Reads, for the newest job of each (dataset, model), the run JSON sources/ab11_<dataset>_<model>_<job>.json
(train_yolo26.py: selected epoch, latency) and its score JSON from the same job:
  LaRS     pq_lars11_<model>_{best,last}_<job>.json (eval_sam_panoptic.py; 11-class models with Water / Sky)
           y = LaRS PQ on val at confidence PQ_CONF (the YOLOv11s + SAM operating point, chosen on held10 in fix 2);
           detection: boxes -> SAM 2.1-L masks; segmentation: the model's own masks
  FloW     coco_eval_flow_<model>_<job>.json      y = pycocotools mAP50-95 on the main-table test GT
  CANSURF  coco_eval_cansurf_<model>_<job>.json   y = pycocotools mAP50-95 on the main-table val GT (2,209 images)
           (mAP50-95 matches the box mAP50-95 that picks best.pt; AP50 is kept in ab11_results.json)
x = median end-to-end latency of the YOLO model alone (Ultralytics preprocess + inference + postprocess, batch 1, imgsz
640, A100): TensorRT FP16 from the newest sources/ab11_latency_<job>.json (bench_yolo26_latency.py, every model timed in
one job), else PyTorch FP16 from the training job. For LaRS detection the SAM step is not in x; its time is in the legend
and the table.
Writes <ablation>/ab11_{lars,flow,cansurf}_latency.png and <ablation>/ab11_results.json (every value with its source).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ABL = REPO / "eval_outputs/water_debris_seg/ablations/11_yolo26_family"
SRC = ABL / "sources"
SIZES = "nsmlx"
PQ_CONF = "0.20"
COLORS = {"detect": "#1f2fa0", "segment": "#e8a25a"}
PLOTS = {  # plot name -> (run-JSON dataset, y label, title)
    "lars": ("lars11", "LaRS PQ (%)", "LaRS val (198 images, 11 classes)"),
    "flow": ("flow", "FloW mAP 50-95 (%)", "FloW test (800 images)"),
    "cansurf": ("cansurf", "CANSURF mAP 50-95 (%)", "CANSURF val (2,209 images)"),
}
PAT = re.compile(r"ab11_(lars11|flow|cansurf)_(yolo26([nslmx])(-seg)?)_(\d+)\.json$")


def rel(p: Path) -> str:
    return str(p.relative_to(REPO))


def score(ds: str, model: str, job: str) -> dict:
    if ds == "lars11":
        out = {}
        for ck in ("best", "last"):
            p = SRC / f"pq_lars11_{model}_{ck}_{job}.json"
            d = json.loads(p.read_text())
            bt = d["by_threshold"]
            out[ck] = {"PQ": bt[PQ_CONF]["PQ"], "PQ_things": bt[PQ_CONF]["PQ_things"], "PQ_stuff": bt[PQ_CONF]["PQ_stuff"],
                       "best_conf": d.get("selected_threshold"), "PQ_at_best_conf": bt[f"{d['selected_threshold']:.2f}"]["PQ"]
                       if d.get("selected_threshold") is not None else None, "source": rel(p)}
            if ck == "best":
                out["sam_ms"] = (d["timing_ms"]["sam"] or {}).get("median")
        return {"y": out["best"]["PQ"], "y_metric": f"PQ (conf {PQ_CONF})", "y_last": out["last"]["PQ"], **out}
    p = SRC / f"coco_eval_{ds}_{model}_{job}.json"
    c = json.loads(p.read_text())["checkpoint"]
    return {"y": c["pycocotools_all_9_classes"]["mAP"], "y_metric": "pycocotools mAP50-95",
            "AP50": c["pycocotools_all_9_classes"]["AP50"], "ultralytics_mAP50": c["ultralytics"]["mAP50"], "source": rel(p)}


def main() -> None:
    lat_files = sorted(SRC.glob("ab11_latency_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1]) if p.stem.rsplit("_", 1)[1].isdigit() else 0)
    lat = json.loads(lat_files[-1].read_text()) if lat_files else None
    backend = "tensorrt" if lat and "tensorrt" in next(iter(lat["models"].values())) else "pytorch"
    latest = {}
    for p in sorted(SRC.glob("ab11_*.json")):
        m = PAT.match(p.name)
        if m and ((m[1], m[2]) not in latest or int(m[5]) > int(PAT.match(latest[(m[1], m[2])].name)[5])):
            latest[(m[1], m[2])] = p
    table = defaultdict(list)
    for (ds, model), p in sorted(latest.items()):
        r = json.loads(p.read_text())
        try:
            sc = score(ds, model, r["job"])
        except FileNotFoundError as e:
            print("missing score for", ds, model, e.filename)
            continue
        lm = lat["models"].get(f"{ds}/{model}") if lat else None
        if lat and lm is None:
            print("no common-job latency for", ds, model)
            continue
        if lm:
            assert lm["weights_sha256"] == r["best"]["sha256"], (ds, model)
        table[ds].append({"model": model, "size": model[6], "task": r["task"], **sc,
                          "selected_epoch": r["best"]["epoch"], "epochs_run": r["curve"]["epochs_run"],
                          "latency_ms": lm[backend]["end_to_end_ms"]["median"] if lm else r["latency"]["end_to_end_ms"]["median"],
                          "latency_backend": f"{backend} FP16" if lm else "pytorch FP16 (training job)",
                          "latency_pytorch_ms": lm["pytorch"]["end_to_end_ms"]["median"] if lm and "pytorch" in lm else None,
                          "latency_source": rel(lat_files[-1]) if lm else rel(p),
                          "params_M": r["params"] / 1e6, "gflops": r["gflops"], "gpu": lat["gpu"] if lm else r["latency"]["gpu"],
                          "wandb": r.get("wandb_train_run"), "run_json": rel(p)})
    for name, (ds, ylabel, title) in PLOTS.items():
        rows = table.get(ds)
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for task in ("detect", "segment"):
            pts = sorted((r for r in rows if r["task"] == task), key=lambda r: SIZES.index(r["size"]))
            if not pts:
                continue
            label = "YOLO26" if task == "detect" else "YOLO26-seg"
            if ds == "lars11":
                if task == "detect":
                    sam = [r["sam_ms"] for r in pts if r.get("sam_ms")]
                    label += " + SAM 2.1-L masks" + (f" (SAM not in x: ~{np.median(sam):.0f} ms/img more)" if sam else "")
                else:
                    label += " (own masks)"
            xs, ys = [r["latency_ms"] for r in pts], [100 * r["y"] for r in pts]
            ax.plot(xs, ys, "-o", color=COLORS[task], lw=2.5 if task == "detect" else 1.6, ms=7, label=label)
            for i, (r, x, y) in enumerate(zip(pts, xs, ys)):
                dip = 0 < i < len(ys) - 1 and y < ys[i - 1] and y < ys[i + 1]   # local minimum: label below the line
                ax.annotate(f"26{r['size']}", (x, y), textcoords="offset points", xytext=(-4, -16 if dip else 7),
                            ha="right", color=COLORS[task], fontsize=11)
        bk = "TensorRT FP16" if rows[0]["latency_backend"].startswith("tensorrt") else "PyTorch FP16"
        ax.set_xlabel(f"End-to-End Latency {rows[0]['gpu'].replace('NVIDIA ', '')} {bk}, batch 1 (ms/img)")
        ax.set_ylabel(ylabel)
        sel = ("box mAP50-95 (-seg: box + mask)" if any(r["task"] == "segment" for r in rows) else "box mAP50-95")
        ax.set_title(f"{title}, 50 epochs\nEach point: the epoch with the best {sel} on this same split (best.pt; optimistic)",
                     fontsize=10)
        ax.margins(x=0.08, y=0.25)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        fig.savefig(ABL / f"ab11_{name}_latency.png", dpi=150)
        plt.close(fig)
    (ABL / "ab11_results.json").write_text(json.dumps(table, indent=2))
    for ds, rows in table.items():
        for r in rows:
            print(f"{ds:8s} {r['model']:12s} {100 * r['y']:5.1f} {r['y_metric']:18s} (ep {r['selected_epoch']:2d}) "
                  f"{r['latency_ms']:6.2f} ms  {r['params_M']:5.1f}M")


if __name__ == "__main__":
    main()
