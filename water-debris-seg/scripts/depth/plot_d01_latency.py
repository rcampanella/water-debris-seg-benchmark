#!/usr/bin/env python3
"""D01 (RGB) plots: point-cloud F-Score vs end-to-end latency per YOLO26-depth size, one figure per source.

y = F-Score at 10 cm (MDEC, per-image least-squares scale + shift), mean over the pooled day + night test images with
    its 95 % bootstrap CI, from ablations/D01_rgb/sources/tables/d01_summary.json (make_d01_tables.py), i.e. the values
    of the pooled LaTeX table.
x = median end-to-end latency (Ultralytics preprocess + inference + postprocess), TensorRT FP16, batch 1, imgsz 640,
    A100, from the latency file named in the summary (bench_depth_latency.py, every model timed in one job):
    fine-tuned = that source's best.pt (SHA-256 checked against the scored checkpoint); zero-shot = released weights
    (timed on Pohang day-test images).
Writes ablations/D01_rgb/d01_{pohang,multiaqua}_f_latency.png and d01_f_latency.json (every value with its source).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
ABL = REPO / "eval_outputs/water_debris_depth/ablations/D01_rgb"
SUMJ = ABL / "sources/tables/d01_summary.json"
SIZES = "nsmlx"
SRCS = {"pohang": "Pohang Canal", "multiaqua": "MULTIAQUA"}
ARMS = {"fine-tuned/best/refit": ("fine-tuned", "#1f2fa0", "-", 2.5),
        "zero-shot": ("zero-shot (released weights)", "#e8a25a", "--", 1.6)}


def main() -> None:
    S = json.loads(SUMJ.read_text())
    lat_file = S["latency"]["file"]
    lat = json.loads((REPO / lat_file).read_text())
    out = {"summary": str(SUMJ.relative_to(REPO)), "latency_file": lat_file, "gpu": lat["gpu"],
           "latency_setting": lat["setting"], "points": []}
    for src, name in SRCS.items():
        fig, ax = plt.subplots(figsize=(9, 4.6))
        n_img = None
        for arm, (label, color, ls, lw) in ARMS.items():
            xs, ys, lo, hi = [], [], [], []
            for s in SIZES:
                row = S["rows"][f"{src}/{s}/{arm}"]
                f = row["pooled"]["relative"]["F"]
                n_img = f["n_images"]
                key = f"{src}/yolo26{s}-depth" if arm != "zero-shot" else f"released/yolo26{s}-depth"
                lm = lat["models"][key]
                if arm != "zero-shot":  # latency measured on the same checkpoint that was scored
                    shas = {sp["weights_sha256"] for sp in row["splits"].values()}
                    assert shas == {lm["weights_sha256"]}, (src, s, shas, lm["weights_sha256"])
                x = lm["tensorrt"]["end_to_end_ms"]["median"]
                xs.append(x); ys.append(f["mean"]); lo.append(f["mean"] - f["ci95"][0]); hi.append(f["ci95"][1] - f["mean"])
                out["points"].append({"source": src, "size": s, "arm": label, "F_pooled": f["mean"], "F_ci95": f["ci95"],
                                      "n_images": f["n_images"], "latency_trt_ms": x, "latency_model": key,
                                      "score_files": sorted(row["files"].values())})
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ls=ls, color=color, lw=lw, ms=7, capsize=3, label=label)
            for s, x, y in zip(SIZES, xs, ys):
                ax.annotate(f"26{s}", (x, y), textcoords="offset points", xytext=(-9, 4), ha="right", color=color, fontsize=11)
        ax.set_xlabel(f"End-to-End Latency {lat['gpu'].replace('NVIDIA ', '')} TensorRT FP16, batch 1 (ms/img)")
        ax.set_ylabel("F-Score at 10 cm (%)")
        ax.set_title(f"{name}, RGB input, test day + night pooled ({n_img:,} images), raw LiDAR ground truth $\\leq$ 20 m\n"
                     "Fine-tuned: 50 epochs, checkpoint and scale chosen on the day test split (optimistic); bars: 95 % CI",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.15 * ax.get_ylim()[1])
        ax.legend(loc="upper right")
        ax.margins(x=0.07)
        fig.tight_layout()
        p = ABL / f"d01_{src}_f_latency.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print("written", p)
    (ABL / "d01_f_latency.json").write_text(json.dumps(out, indent=1))
    print("written", ABL / "d01_f_latency.json")


if __name__ == "__main__":
    main()
