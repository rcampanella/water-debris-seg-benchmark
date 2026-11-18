#!/usr/bin/env python3
"""Overleaf depth section, results only: writes eval_outputs/water_debris_depth/latex/depth_benchmark.tex from
ablations/D01_rgb/sources/tables/d01_summary.json (make_d01_tables.py). Every value carries a % source: comment.

Tables: (1) D01 pooled day + night, all sizes, zero-shot vs fine-tuned, both metric families, params, latency;
(2a, 2b) day vs night, relative (F, AbsRel) and absolute (SILog, absErrRel) in separate tables; (3) absErrRel per distance band. Fine-tuned = best.pt with our metric-scale refit (D12 main);
zero-shot = released weights with shipped calibration. Raw ground truth only (Pohang Uhrig not built).
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EV = REPO / "eval_outputs/water_debris_depth"
SUMJ = EV / "ablations/D01_rgb/sources/tables/d01_summary.json"
S = json.load(open(SUMJ))
R = S["rows"]
SIZES = "nsmlx"
SRCS = (("pohang", "Pohang Canal"), ("multiaqua", "MULTIAQUA"))
ARMS = (("zero-shot", "zero-shot"), ("fine-tuned/best/refit", "fine-tuned"))
BANDS = ("0-5", "5-10", "10-15", "15-20")
SUMREL = SUMJ.relative_to(REPO)


def src_comment(keys):
    files = sorted({f for k in keys for f in R[k]["files"].values()})
    return "% source: " + "; ".join(files)


def bold_best(vals, higher):
    """vals: list of floats -> list of strings with the best one bold (per column)."""
    b = max(vals) if higher else min(vals)
    return [rf"\textbf{{{v:.2f}}}" if v == b else f"{v:.2f}" for v in vals]


def table_pooled():
    cols = [(s, fam, k, hi) for s, _ in SRCS for fam, k, hi in
            (("relative", "F", True), ("relative", "AbsRel", False), ("absolute", "SILog", False), ("absolute", "absErrorRel", False))]
    keys = [(sz, a) for sz in SIZES for a, _ in ARMS]
    grid = {c: bold_best([R[f"{c[0]}/{sz}/{a}"]["pooled"][c[1]][c[2]]["mean"] for sz, a in keys], c[3]) for c in cols}
    L = [r"\begin{table*}[!htb]", r"\centering", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{ll|rr|cccc|cccc}", r"\toprule",
         r"\multirow{3}{*}{Model} & \multirow{3}{*}{Arm} & \multirow{3}{*}{Params (M)} & \multirow{3}{*}{Latency (ms)} & \multicolumn{4}{c|}{Pohang Canal} & \multicolumn{4}{c}{MULTIAQUA} \\",
         r"\cmidrule(lr){5-8}\cmidrule(lr){9-12}",
         r" & & & & \multicolumn{2}{c}{Relative} & \multicolumn{2}{c|}{Absolute} & \multicolumn{2}{c}{Relative} & \multicolumn{2}{c}{Absolute} \\",
         r" & & & & F$\uparrow$ & AbsRel$\downarrow$ & SILog$\downarrow$ & absErrRel$\downarrow$ & F$\uparrow$ & AbsRel$\downarrow$ & SILog$\downarrow$ & absErrRel$\downarrow$ \\",
         r"\midrule"]
    lat = S["latency"]["file"]
    for i, sz in enumerate(SIZES):
        m = R[f"pohang/{sz}/fine-tuned/best/refit"]["meta"]
        L.append(src_comment([f"{s}/{sz}/{a}" for s, _ in SRCS for a, _ in ARMS]) + f"; params: {m['run_json']}; latency: {lat}")
        for j, (a, lab) in enumerate(ARMS):
            idx = keys.index((sz, a))
            pre = (rf"\multirow{{2}}{{*}}{{YOLO26{sz}-depth}} & {lab} & \multirow{{2}}{{*}}{{{m['params_M']:.1f}}} & \multirow{{2}}{{*}}{{{m['latency_trt_ms']:.2f}}} & "
                   if j == 0 else rf" & {lab} & & & ")
            L.append(pre + " & ".join(grid[c][idx] for c in cols) + r" \\")
        L.append(r"\midrule" if i < len(SIZES) - 1 else r"\bottomrule")
    L += [r"\end{tabular}}",
          r"\caption{RGB input, YOLO26-depth n--x: zero-shot (released weights, shipped calibration) vs fine-tuned per source (50 epochs, best checkpoint, metric scale refitted). "
          r"Test split, day and night pooled (mean over test images: Pohang 3,487, MULTIAQUA 215), raw LiDAR ground truth $\leq$20\,m. "
          r"\emph{Relative}: \href{https://arxiv.org/abs/2504.17787}{MDEC 2025} evaluator after per-image least-squares scale and shift; F = point-cloud F-Score at 10\,cm (\href{https://arxiv.org/abs/2203.08122}{\"Ornek et al.}), AbsRel in \%. "
          r"\emph{Absolute}: \href{https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction}{KITTI} metrics, no alignment; SILog in log(m)$\cdot$100, absErrRel in \%. "
          r"Latency: TensorRT FP16, A100, batch 1, imgsz 640, end to end. Bold: best per column. Fine-tuned checkpoint and scale are chosen on the day test split (optimistic).}",
          r"\label{tab:depth_rgb}", r"\end{table*}"]
    return L


DAYNIGHT = {  # family -> (metrics (key, header, higher is better), label suffix, caption title, caption metric text)
    "relative": ((("F", r"F$\uparrow$", True), ("AbsRel", r"AbsRel$\downarrow$", False)), "rel", "relative metrics",
                 r"\href{https://arxiv.org/abs/2504.17787}{MDEC 2025} evaluator after per-image least-squares scale and shift (shape): "
                 r"F = point-cloud F-Score at 10\,cm, AbsRel in \%."),
    "absolute": ((("SILog", r"SILog$\downarrow$", False), ("absErrorRel", r"absErrRel$\downarrow$", False)), "abs", "absolute metrics",
                 r"\href{https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction}{KITTI} metrics, no alignment (metres): "
                 r"SILog in log(m)$\cdot$100 (forgives one scale factor per image), absErrRel in \%."),
}


def table_daynight(fam):
    """Day vs night, one metric family per table (split 2026-11-17: four metrics did not fit one table)."""
    metrics, suffix, title, desc = DAYNIGHT[fam]
    hdr = " & ".join(h for _, h, _ in metrics)
    L = [r"\begin{table*}[!htb]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{ll|cc|cc|cc|cc}", r"\toprule",
         r"\multirow{3}{*}{Model} & \multirow{3}{*}{Arm} & \multicolumn{4}{c|}{Pohang Canal} & \multicolumn{4}{c}{MULTIAQUA} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r" & & \multicolumn{2}{c|}{Day} & \multicolumn{2}{c|}{Night} & \multicolumn{2}{c|}{Day} & \multicolumn{2}{c}{Night} \\",
         r" & & " + " & ".join([hdr] * 4) + r" \\",
         r"\midrule"]
    cols = [(s, sp, fam, k, hi) for s, _ in SRCS for sp in ("test_day", "test_night") for k, _, hi in metrics]
    keys = [(sz, a) for sz in SIZES for a, _ in ARMS]
    grid = {c: bold_best([R[f"{c[0]}/{sz}/{a}"]["splits"][c[1]]["all"][c[2]][c[3]]["mean"] for sz, a in keys], c[4]) for c in cols}
    for i, sz in enumerate(SIZES):
        L.append(src_comment([f"{s}/{sz}/{a}" for s, _ in SRCS for a, _ in ARMS]))
        for j, (a, lab) in enumerate(ARMS):
            idx = keys.index((sz, a))
            L.append((rf"\multirow{{2}}{{*}}{{YOLO26{sz}-depth}} & {lab} & " if j == 0 else rf" & {lab} & ") + " & ".join(grid[c][idx] for c in cols) + r" \\")
        L.append(r"\midrule" if i < len(SIZES) - 1 else r"\bottomrule")
    n = {s: {sp: R[f"{s}/n/zero-shot"]["splits"][sp]["n_images"] for sp in ("test_day", "test_night")} for s, _ in SRCS}
    L += [r"\end{tabular}",
          rf"\caption{{RGB input, {title}, day vs night test images (Pohang: day sequence 04, {n['pohang']['test_day']:,} images; night sequences 01 and 05, {n['pohang']['test_night']:,}; "
          rf"MULTIAQUA: {n['multiaqua']['test_day']} day, {n['multiaqua']['test_night']} night). {desc} Arms as Table~\ref{{tab:depth_rgb}}; bold: best per column.}}",
          rf"\label{{tab:depth_rgb_daynight_{suffix}}}", r"\end{table*}"]
    return L


def table_bands():
    L = [r"\begin{table*}[!htb]", r"\centering", r"\small", r"\begin{tabular}{ll|cccc|cccc}", r"\toprule",
         r"\multirow{2}{*}{Model} & \multirow{2}{*}{Arm} & \multicolumn{4}{c|}{Pohang Canal, day} & \multicolumn{4}{c}{Pohang Canal, night} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r" & & " + " & ".join(f"{b.replace('-', '--')}\\,m" for b in BANDS) + " & " + " & ".join(f"{b.replace('-', '--')}\\,m" for b in BANDS) + r" \\",
         r"\midrule"]
    cols = [(sp, b) for sp in ("test_day", "test_night") for b in BANDS]
    keys = [(sz, a) for sz in SIZES for a, _ in ARMS]
    def v(sz, a, sp, b):
        return R[f"pohang/{sz}/{a}"]["splits"][sp]["bands"][b]["absolute"]["absErrorRel"]["mean"]
    grid = {c: bold_best([v(sz, a, *c) for sz, a in keys], False) for c in cols}
    for i, sz in enumerate(SIZES):
        L.append(src_comment([f"pohang/{sz}/{a}" for a, _ in ARMS]))
        for j, (a, lab) in enumerate(ARMS):
            idx = keys.index((sz, a))
            L.append((rf"\multirow{{2}}{{*}}{{YOLO26{sz}-depth}} & {lab} & " if j == 0 else rf" & {lab} & ") + " & ".join(grid[c][idx] for c in cols) + r" \\")
        L.append(r"\midrule" if i < len(SIZES) - 1 else r"\bottomrule")
    L += [r"\end{tabular}",
          r"\caption{RGB input, Pohang Canal: absErrRel (\%, KITTI, no alignment) per ground-truth distance band; mean over images with ground truth in the band. "
          r"Arms as Table~\ref{tab:depth_rgb}; bold: best per column. The 0--5\,m band holds few points (under 2\,\% of ground truth) and is not yet inspected.}",
          r"\label{tab:depth_rgb_bands}", r"\end{table*}"]
    return L


def main():
    L = [r"% Depth benchmark section for the PROACTIF Overleaf document (results only). Paste before \end{document}.",
         r"% Needs booktabs, multirow, graphicx (\resizebox), hyperref. Generated by water-debris-seg/scripts/depth/make_depth_section_tex.py",
         rf"% from {SUMREL}; protocol, decisions and evidence: eval_outputs/water_debris_depth/README.md.", "",
         r"\clearpage", r"\section{Depth benchmark}", r"\label{sec:depth}", ""]
    # data per source + frame filter (hand-written, counts from standard_split_summary.json)
    data = (EV / "latex/depth_data_tables.tex").read_text().splitlines() + [""]
    tables = (table_pooled() + [""] + table_daynight("relative") + [""] + table_daynight("absolute") + [""]
              + table_bands() + [""])
    L += data + [r"\subsection{RGB input}", r"\label{sec:depth_rgb}", ""] + tables
    (EV / "latex/depth_benchmark.tex").write_text("\n".join(L))
    print("written", EV / "latex/depth_benchmark.tex")
    # standalone document: same tables, own preamble, compiles on its own
    D = [r"% Depth benchmark results, standalone document (compiles on its own). Generated by",
         rf"% water-debris-seg/scripts/depth/make_depth_section_tex.py from {SUMREL}.",
         r"\documentclass[10pt]{article}", r"\usepackage[a4paper,margin=1.5cm]{geometry}",
         r"\usepackage[T1]{fontenc}", r"\usepackage{booktabs,multirow,graphicx}", r"\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{hyperref}",
         r"\title{Depth benchmark: results}", r"\author{}", r"\date{RGB input, runs of 15 November 2026}",
         r"\begin{document}", r"\maketitle", ""]
    D += [ln.replace(r"\subsection{", r"\section*{") for ln in data] + [r"\section*{RGB input}", ""] + tables + [r"\end{document}", ""]
    (EV / "latex/depth_results_standalone.tex").write_text("\n".join(D))
    print("written", EV / "latex/depth_results_standalone.tex")


if __name__ == "__main__":
    main()
