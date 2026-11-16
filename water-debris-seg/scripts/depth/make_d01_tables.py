#!/usr/bin/env python3
"""D01 tables from the scorer outputs (eval_depth.py) of the zero-shot and fine-tuned arms.

Inputs (eval_outputs/water_debris_depth/ablations/D01_rgb/sources/):
  zeroshot/zs_<src>_yolo26<s>_<split>_<job>.json(l)            newest job per (src, size, split); variant 'shipped'
  finetune/d01_<src>_yolo26<s>-depth_<job>.json                 newest full run (50 epochs, fraction 1) per (src, size)
  finetune/ft_<src>_yolo26<s>-depth_<best|last>_<split>_<job>.json(l)   variant 'refit' (D12 main) and 'shipped'
  latency/d01_latency_<job>.json                                newest complete file, TensorRT FP16 end-to-end median
Pooled (day + night) = mean over all test images of both splits, bootstrap 95 % CI over images (1000 resamples, seed 0),
from the per-image .jsonl; day / night / band values are the scorer's own .json aggregates.
Outputs: ablations/D01_rgb/d01_results.md, ablations/D01_rgb/sources/tables/d01_summary.json,
         latex/depth_d01_tables.tex (filled tab:depth_d01 + D01 rows of tab:depth_benchmark, % source: comments).
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EV = REPO / "eval_outputs/water_debris_depth"
SRC = EV / "ablations/D01_rgb/sources"
SIZES = "nsmlx"
SOURCES = ("pohang", "multiaqua")
SPLITS = ("test_day", "test_night")
BANDS = ("0-5", "5-10", "10-15", "15-20")
REL = ("F", "AbsRel"); ABS = ("SILog", "absErrorRel")


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(REPO))


def newest(pattern: str, rx: str) -> str | None:
    c = [(int(re.search(rx, p)[1]), p) for p in glob.glob(pattern) if re.search(rx, p)]
    return max(c)[1] if c else None


def pooled(jsonls: list[str], variant: str, fam: str, key: str) -> dict:
    v = []
    for f in jsonls:
        for line in open(f):
            r = json.loads(line)
            if r["variant"] == variant and r["all"] is not None: v.append(r["all"][fam][key])
    v = np.array(v); rng = np.random.default_rng(0)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_images": int(len(v))}


def collect() -> dict:
    S = {"rows": {}, "latency": None}
    lat = None
    for p in sorted(glob.glob(str(SRC / "latency/d01_latency_*.json"))):
        d = json.load(open(p))
        if d.get("complete"): lat = (p, d)
    if lat: S["latency"] = {"file": rel(lat[0]), "gpu": lat[1]["gpu"], "tensorrt": lat[1].get("tensorrt")}
    for src in SOURCES:
        for s in SIZES:
            # zero-shot
            files = {sp: newest(str(SRC / f"zeroshot/zs_{src}_yolo26{s}_{sp}_*.json"), rf"_{sp}_(\d+)\.json$") for sp in SPLITS}
            S["rows"][f"{src}/{s}/zero-shot"] = arm(files, "shipped", None)
            # fine-tuned
            runs = []
            for p in glob.glob(str(SRC / f"finetune/d01_{src}_yolo26{s}-depth_*.json")):
                d = json.load(open(p)); ta = d["train_args"]
                if ta.get("epochs") == 50 and float(ta.get("fraction", 1)) == 1.0: runs.append((int(d["job"]), p, d))
            if not runs: continue
            job, rp, d = max(runs)
            meta = {"run_json": rel(rp), "params_M": d["params"] / 1e6, "gflops": d["gflops"],
                    "best_epoch": d["curve"]["selected_epoch"], "epochs_run": d["curve"]["epochs_run"]}
            if lat:
                m = lat[1]["models"].get(f"{src}/yolo26{s}-depth", {})
                if "tensorrt" in m: meta["latency_trt_ms"] = m["tensorrt"]["end_to_end_ms"]["median"]
                if "pytorch" in m: meta["latency_pt_ms"] = m["pytorch"]["end_to_end_ms"]["median"]
            for ck in ("best", "last"):
                files = {sp: str(SRC / f"finetune/ft_{src}_yolo26{s}-depth_{ck}_{sp}_{job}.json") for sp in SPLITS}
                for var in ("refit", "shipped"):
                    S["rows"][f"{src}/{s}/fine-tuned/{ck}/{var}"] = arm(files, var, meta)
    return S


def arm(files: dict, variant: str, meta) -> dict:
    out = {"variant": variant, "files": {sp: rel(f) for sp, f in files.items()}, "meta": meta, "splits": {}, "pooled": {}}
    for sp, f in files.items():
        d = json.load(open(f)); v = d["variants"][variant]
        out["splits"][sp] = {"all": {fam: {k: v["all"][fam][k] for k in keys} for fam, keys in (("relative", REL + ("delta1",)), ("absolute", ABS))},
                             "bands": {b: {fam: {k: v["bands"][b][fam][k] for k in keys} for fam, keys in (("relative", REL), ("absolute", ABS))} for b in BANDS},
                             "n_images": d["n_images"], "refit": d.get("refit"), "weights_sha256": d["weights_sha256"]}
    jl = [f[:-5] + ".jsonl" for f in files.values()]
    out["pooled"] = {fam: {k: pooled(jl, variant, fam, k) for k in keys} for fam, keys in (("relative", REL), ("absolute", ABS))}
    return out


def fmt(x, ci=False):
    m = x["mean"]
    return f"{m:.2f} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]" if ci else f"{m:.2f}"


def markdown(S) -> str:
    L = ["# D01 (RGB) results", "", "Generated by `water-debris-seg/scripts/depth/make_d01_tables.py` from "
         "`sources/zeroshot`, `sources/finetune`, `sources/latency`; full values with sources in `sources/tables/d01_summary.json`.",
         "", "Standard split (README D6); raw ground truth (unadapted), ≤20 m. Zero-shot = released weights, shipped calibration. "
         "Fine-tuned = best.pt (epoch with best Ultralytics δ1 on the day test split), metric scale refitted on the day test split "
         "(D12, main); both choices use the test split, so fine-tuned values are optimistic. Values: mean over test images "
         "[bootstrap 95 % CI]. F = point-cloud F-Score at 10 cm after per-image LS alignment (MDEC); AbsRel % (aligned, capped 20 m); "
         "SILog log(m)·100 and absErrRel % (KITTI, unaligned, uncapped).", ""]
    if S["latency"]: L += [f"Latency: TensorRT FP16 end-to-end median, {S['latency']['gpu']}, batch 1, imgsz 640 (`{S['latency']['file']}`).", ""]
    for src in SOURCES:
        L += [f"## {src} — pooled day + night", "",
              "| Model | Arm | Params (M) | Latency (ms) | Best ep. | F ↑ | AbsRel % ↓ | SILog ↓ | absErrRel % ↓ |", "|---|---|---|---|---|---|---|---|---|"]
        for s in SIZES:
            for a in ("zero-shot", "fine-tuned/best/refit"):
                r = S["rows"].get(f"{src}/{s}/{a}")
                if not r: continue
                m = S["rows"].get(f"{src}/{s}/fine-tuned/best/refit", {}).get("meta") or {}
                p = r["pooled"]
                L.append(f"| YOLO26{s}-depth | {a.split('/')[0]} | {m.get('params_M', float('nan')):.1f} | "
                         f"{m['latency_trt_ms']:.2f} | " if "latency_trt_ms" in m else f"| YOLO26{s}-depth | {a.split('/')[0]} | {m.get('params_M', float('nan')):.1f} | – | ")
                L[-1] += (f"{m.get('best_epoch', '') if a != 'zero-shot' else ''} | {fmt(p['relative']['F'], 1)} | {fmt(p['relative']['AbsRel'], 1)} | "
                          f"{fmt(p['absolute']['SILog'], 1)} | {fmt(p['absolute']['absErrorRel'], 1)} |")
        L.append("")
        for sp in SPLITS:
            L += [f"### {src} — {sp}", "", "| Model | Arm | F | AbsRel % | δ1 | SILog | absErrRel % | n images |", "|---|---|---|---|---|---|---|---|"]
            for s in SIZES:
                for a in ("zero-shot", "fine-tuned/best/refit", "fine-tuned/best/shipped", "fine-tuned/last/refit"):
                    r = S["rows"].get(f"{src}/{s}/{a}")
                    if not r: continue
                    x = r["splits"][sp]["all"]
                    L.append(f"| YOLO26{s} | {a} | {fmt(x['relative']['F'], 1)} | {fmt(x['relative']['AbsRel'])} | {x['relative']['delta1']['mean']:.3f} | "
                             f"{fmt(x['absolute']['SILog'], 1)} | {fmt(x['absolute']['absErrorRel'], 1)} | {r['splits'][sp]['n_images']} |")
            L.append("")
            L += [f"#### {src} — {sp}, per distance band (absErrRel % / F)", "", "| Model | Arm | " + " | ".join(f"{b} m" for b in BANDS) + " |",
                  "|---|---|" + "---|" * len(BANDS)]
            for s in SIZES:
                for a in ("zero-shot", "fine-tuned/best/refit"):
                    r = S["rows"].get(f"{src}/{s}/{a}")
                    if not r: continue
                    bb = r["splits"][sp]["bands"]
                    L.append(f"| YOLO26{s} | {a.split('/')[0]} | " + " | ".join(
                        f"{bb[b]['absolute']['absErrorRel']['mean']:.1f} / {bb[b]['relative']['F']['mean']:.1f}" if bb[b]['absolute']['absErrorRel'].get('mean') is not None else "–" for b in BANDS) + " |")
            L.append("")
    L += ["## Sources", ""]
    for k, r in S["rows"].items():
        L.append(f"- {k}: " + ", ".join(f"`{f}`" for f in r["files"].values()) + (f", run `{r['meta']['run_json']}`" if r["meta"] else ""))
    return "\n".join(L) + "\n"


def latex(S) -> str:
    def cell(src, s, a, fam, k):
        r = S["rows"].get(f"{src}/{s}/{a}")
        return f"{r['pooled'][fam][k]['mean']:.2f}" if r else "--"
    def meta(s, key, f="{:.1f}"):
        m = (S["rows"].get(f"pohang/{s}/fine-tuned/best/refit") or {}).get("meta") or {}
        return f.format(m[key]) if key in m else r"\tbd"
    def lat(src, s):
        m = (S["rows"].get(f"{src}/{s}/fine-tuned/best/refit") or {}).get("meta") or {}
        return f"{m['latency_trt_ms']:.2f}" if "latency_trt_ms" in m else r"\tbd"
    L = [r"% D01 tables, filled by water-debris-seg/scripts/depth/make_d01_tables.py; values = pooled day+night means (bootstrap CIs in",
         r"% ablations/D01_rgb/d01_results.md). Sources: ablations/D01_rgb/sources/tables/d01_summary.json (per cell: files listed there).",
         r"% Pohang Uhrig GT column: not built yet (\tbd). Fine-tuned = best.pt, refit metric scale (D12 main).",
         r"\providecommand{\tbd}{\textit{tbd}}", ""]
    # Table B
    L += [r"\begin{table*}[t]", r"\centering", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{ll|rr|cccc|cccc|cccc}", r"\toprule",
          r"\multirow{3}{*}{Model} & \multirow{3}{*}{Arm} & \multirow{3}{*}{Params (M)} & \multirow{3}{*}{Latency (ms)} & \multicolumn{4}{c|}{Pohang, Uhrig GT} & \multicolumn{4}{c|}{Pohang, raw GT} & \multicolumn{4}{c}{MULTIAQUA, raw GT} \\",
          r"\cmidrule(lr){5-8}\cmidrule(lr){9-12}\cmidrule(lr){13-16}",
          r" & & & & \multicolumn{2}{c}{Relative} & \multicolumn{2}{c|}{Absolute} & \multicolumn{2}{c}{Relative} & \multicolumn{2}{c|}{Absolute} & \multicolumn{2}{c}{Relative} & \multicolumn{2}{c}{Absolute} \\",
          r" & & & & F$\uparrow$ & AbsRel$\downarrow$ & SILog$\downarrow$ & absErrRel$\downarrow$ & F$\uparrow$ & AbsRel$\downarrow$ & SILog$\downarrow$ & absErrRel$\downarrow$ & F$\uparrow$ & AbsRel$\downarrow$ & SILog$\downarrow$ & absErrRel$\downarrow$ \\",
          r"\midrule"]
    for i, s in enumerate(SIZES):
        for a, lab in (("zero-shot", "zero-shot"), ("fine-tuned/best/refit", "fine-tuned")):
            vals = [cell(src, s, a, fam, k) for src in SOURCES for fam, k in (("relative", "F"), ("relative", "AbsRel"), ("absolute", "SILog"), ("absolute", "absErrorRel"))]
            first = lab == "zero-shot"
            L.append((rf"\multirow{{2}}{{*}}{{YOLO26{s}-depth}} & {lab} & \multirow{{2}}{{*}}{{{meta(s, 'params_M')}}} & " if first else rf" & {lab} & & ")
                     + rf"{lat('pohang', s)} & \tbd & \tbd & \tbd & \tbd & " + " & ".join(vals) + r" \\")
        L.append(r"\midrule" if i < len(SIZES) - 1 else r"\bottomrule")
    L += [r"\end{tabular}}",
          r"\caption{D01, RGB input: YOLO26-depth n--x, zero-shot (released weights, built-in calibration) and fine-tuned per source (50 epochs). "
          r"Standard split (Pohang: train sequences 00, 02, 03, test 04 by day and 01, 05 by night; MULTIAQUA: official split), ground truth $\leq$20\,m, "
          r"day and night pooled (mean over test images); per-band (0--5, 5--10, 10--15, 15--20\,m), day / night results and 95\,\% bootstrap intervals in the D01 report. "
          r"\emph{Relative}: \href{https://arxiv.org/abs/2504.17787}{MDEC 2025} evaluator, per-image least-squares scale and shift, predictions capped at 20\,m; "
          r"F = point-cloud F-Score at 10\,cm (\href{https://arxiv.org/abs/2203.08122}{\"Ornek et al.}), AbsRel in \%. "
          r"\emph{Absolute}: \href{https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction}{KITTI depth benchmark} metrics, no alignment, no capping; "
          r"SILog in log(m)$\cdot$100, absErrRel in \%. Scores are comparable only within one ground-truth column. "
          r"Fine-tuned: checkpoint (best Ultralytics $\delta_1$) and metric scale (our refit) are chosen on the day test split, so fine-tuned values are optimistic. "
          r"Latency: TensorRT FP16, A100, batch 1, imgsz 640, end to end, Pohang images (the zero-shot model of the same size has the same architecture).}",
          r"\label{tab:depth_d01}", r"\end{table*}", ""]
    # Table A rows (best fine-tuned size on Pohang raw F, pooled, until the Uhrig column exists)
    best = max(SIZES, key=lambda s: S["rows"][f"pohang/{s}/fine-tuned/best/refit"]["pooled"]["relative"]["F"]["mean"] if f"pohang/{s}/fine-tuned/best/refit" in S["rows"] else -1)
    L += [rf"% tab:depth_benchmark, D01 rows (best fine-tuned size on Pohang raw-GT F-Score, pooled: YOLO26{best}; replace the two D01 rows):"]
    for a, lab in (("zero-shot", "zero-shot"), ("fine-tuned/best/refit", "fine-tuned")):
        v = [cell(src, best, a, fam, k) for src in SOURCES for fam, k in (("relative", "F"), ("absolute", "SILog"))]
        pre = r"\multirow{2}{*}{D01} & \multirow{2}{*}{RGB} & " if lab == "zero-shot" else " & & "
        L.append(pre + rf"YOLO26{best}-depth, {lab} & {meta(best, 'params_M')} & {lat('pohang', best)} & \tbd & \tbd & " + " & ".join(v) + r" \\")
    return "\n".join(L) + "\n"


def main():
    S = collect()
    (SRC / "tables").mkdir(exist_ok=True)
    json.dump(S, open(SRC / "tables/d01_summary.json", "w"), indent=1)
    (EV / "ablations/D01_rgb/d01_results.md").write_text(markdown(S))
    (EV / "latex/depth_d01_tables.tex").write_text(latex(S))
    print("rows", len(S["rows"]), "latency", S["latency"])


if __name__ == "__main__":
    main()
