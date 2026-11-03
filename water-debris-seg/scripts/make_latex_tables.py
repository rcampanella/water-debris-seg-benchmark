#!/usr/bin/env python3
"""Benchmark tables in LaTeX and Markdown: results on LaRS, FloW and CANSURF (one table each), class sets, ablation codes.

Each results table has three row groups: published results copied from the dataset's paper (LaRS paper Tables 4-5,
test set; FloW paper Table 2, same test split as ours; CANSURF paper Table II, its curated 900-image test set), the
in-domain rows of our main comparison table (fairness_fixes/README.md, 12-epoch runs), and every YOLO26 / YOLO26-seg
model of ablation 11 (dagger: checkpoint picked on the evaluation split). Every value of ours is read here from its
JSON: LaRS PQ / mIoU from the official LaRS evaluator, box AP from the pycocotools scorer, CANSURF P / R / F1 from the
CANSURF paper protocol. Sizes: parameters counted from the checkpoints (EoMT ViT-B 93.6 M, LoRA adapters 2.4 M, SAM
2.1-L 224.5 M, YOLOv11s 9.5 M; YOLO26 from the ablation 11 run JSONs).

Writes
  eval_outputs/water_debris_seg/latex/benchmark_tables.tex   metric headers link to the class-set table, model names to
                                                              the code table (hyperref); each row's sources as a comment
  eval_outputs/water_debris_seg/BENCHMARK_TABLES.md          the same tables for auditing: every value links to its
                                                              JSON, every model row to its W&B training run(s), every
                                                              code to its ablation / fix page
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EO = REPO / "eval_outputs/water_debris_seg"
ABL11 = EO / "ablations/11_yolo26_family"
FF = "eval_outputs/water_debris_seg/fairness_fixes/sources"
AB10 = "eval_outputs/water_debris_seg/ablations/10_yolo_lars_boxes/sources"
TEX = EO / "latex/benchmark_tables.tex"
MD = EO / "BENCHMARK_TABLES.md"
WB = "https://wandb.ai/riccardo-campanella/water-debris-seg/runs/"

EOMT, LORA, SAM, Y11S = 93.6, 2.4, 224.5, 9.5

CLASS_SETS = [  # code, classes (fairness_fixes/README.md "Class codes")
    ("COCO-133", "80 COCO objects + 53 COCO regions"),
    ("L11", "Static Obstacle, Water, Sky, Boat/ship, Row boats, Paddle board, Buoy, Swimmer, Animal, Float, Other"),
    ("L9", "Static Obstacle + the 8 objects of L8"),
    ("L8", "Boat/ship, Row boats, Paddle board, Buoy, Swimmer, Animal, Float, Other"),
    ("L3", "obstacle (Static Obstacle + L8 merged), water, sky"),
    ("D-LaRS", "debris: Float, Other"),
    ("D-COCO", "debris-like COCO: bottle, cup, frisbee, sports ball, kite, backpack, handbag, suitcase, umbrella"),
    ("waste", "FloW's single class"),
    ("can", "CANSURF's single class"),
]

ABLATIONS = [  # code, description, page (relative to EO)
    ("A01", "EoMT zero-shot on LaRS (COCO panoptic weights, COCO → LaRS class map)", "ablations/01_zeroshot_lars/README.md"),
    ("A02", "EoMT fine-tuned on LaRS, semantic (3 classes)", "ablations/02_finetune_lars_semantic/README.md"),
    ("A03", "EoMT fine-tuned on LaRS, panoptic (11 classes)", "ablations/03_finetune_lars_panoptic/README.md"),
    ("A04", "FloW-Img: EoMT mask→box mAP (evaluation only)", "ablations/04_flow_mask2box/README.md"),
    ("A05", "CANSURF: EoMT mask→box mAP (evaluation only)", "ablations/05_cansurf_small_object/README.md"),
    ("A06", "WSODD rubbish vs distractors (out of scope)", "ablations/06_wsodd_distractors/README.md"),
    ("A07", "VidEoMT temporal model on LaRS sequences", "ablations/07_videomt_temporal/README.md"),
    ("A08", "YOLOv11 specialised baselines on FloW and CANSURF", "ablations/08_specialized_yolo/README.md"),
    ("A09", "Cross-dataset transfer (LaRS-trained models → FloW, CANSURF)", "ablations/09_cross_dataset_transfer/README.md"),
    ("A10", "YOLOv11n/s on LaRS boxes traced from the instance masks", "ablations/10_yolo_lars_boxes/README.md"),
    ("A11", "YOLO26 / YOLO26-seg family (n–x), accuracy vs end-to-end latency; checkpoint picked on the evaluation split",
     "ablations/11_yolo26_family/README.md"),
    ("F1", "One EoMT model, three LaRS metrics in separate columns (PQ, mIoU, box AP); EoMT + SAM control",
     "fairness_fixes/FIX1_LARS_SPLIT_METRICS.md"),
    ("F2", "YOLO trained and scored on LaRS; YOLOv11s + SAM 2.1-L for LaRS PQ / mIoU", "fairness_fixes/FIX2_YOLO_ON_LARS.md"),
    ("F3", "EoMT trained on FloW / CANSURF with SAM pseudo-masks; YOLOv11s with the same protocol",
     "fairness_fixes/FIX3_SAM_BOX_DATASETS.md"),
    ("F4", "EoMT full fine-tuning vs LoRA, held-out epoch selection", "fairness_fixes/FIX4_EOMT_LORA.md"),
]

LARS_PAPER = "https://arxiv.org/abs/2308.09618"
FLOW_PAPER = ("https://openaccess.thecvf.com/content/ICCV2021/papers/"
              "Cheng_FloW_A_Dataset_and_Benchmark_for_Floating_Waste_Detection_in_ICCV_2021_paper.pdf")
CANSURF_PAPER = "https://arxiv.org/html/2605.16774v1"

# Published results, copied from the papers (LaRS: Table 4 PQ, Table 5 mIoU, test set via the evaluation server;
# FloW: Table 2 mmAP = mAP@[.50:.05:.95], FloW-Img test; CANSURF: Table II P / R / F1, curated 900-image test set)
LARS_PUB = [  # model, PQ, PQ_th, PQ_st, mIoU
    ("Panoptic DeepLab (ResNet-50)", 34.7, 13.4, 91.4, 95.4), ("Panoptic FPN (ResNet-50)", 40.1, 21.7, 89.3, 93.5),
    ("Panoptic FPN (ResNet-101)", 38.7, 19.7, 89.4, 93.3), ("MaX-DeepLab (MaX-S)", 31.9, 9.5, 91.7, 95.4),
    ("Mask2Former (ResNet-50)", 37.6, 17.0, 92.4, 95.4), ("Mask2Former (ResNet-101)", 37.2, 16.3, 92.8, 95.6),
    ("Mask2Former (Swin-T)", 39.2, 18.8, 93.7, 96.8), ("Mask2Former (Swin-B)", 41.7, 21.8, 94.7, 97.6),
]
FLOW_PUB = [("DSSD", 27.5), ("RetinaNet", 24.9), ("YOLOv3", 12.8), ("Faster R-CNN", 18.4), ("FPN", 33.4), ("Cascade R-CNN", 43.4)]
CANSURF_PUB = [  # model, params (M), P, R, F1
    ("YOLOv11s (baseline)", "9.4", 31, 4, 7), ("YOLOv8s", "11.2", 89, 89, 89), ("YOLOv11s", "9.4", 90, 90, 90),
    ("SAHI + YOLOv11s", "9.4", 67, 93, 78), ("YOLO-World", "179", 46, 20, 28), ("Grounding DINO", "172", 9, 20, 12),
]


def row(model, code, size, vals, srcs, wandb=()):
    """One table row: values with one source each (repo path, URL or None) and the W&B runs [(label, url)]."""
    assert len(vals) == len(srcs)
    return {"model": model, "code": code, "size": size, "vals": vals, "srcs": srcs, "wandb": list(wandb)}


def runs(b, a=None):
    """W&B training runs of a held-out model: B = the scored stage-B run, A = stage A (epoch selection)."""
    return [("B", WB + b)] + ([("A", WB + a)] if a else [])


def jget(path: str, *keys):
    d = json.loads((REPO / path).read_text())
    for k in keys:
        d = d[k]
    return d


def official(path: str) -> list:
    s = jget(path, "summary")
    return [s["PQ"], s["PQ_th"], s["PQ_st"], s["sem_mIoU"]]


def paper_prf(path: str) -> list:
    p = jget(path, "class_agnostic", "paper_protocol")
    return [100 * p["precision"], 100 * p["recall"], 100 * p["f1"]]


def ab11_runs(key: str):
    t = json.loads((ABL11 / "ab11_results.json").read_text())
    for task in ("detect", "segment"):
        yield from sorted((r for r in t[key] if r["task"] == task), key=lambda r: "nsmlx".index(r["size"]))


def rel11(pattern: str) -> str:
    files = [p for p in sorted(ABL11.glob(f"sources/{pattern}")) if not p.name.endswith(".preds.json")]
    return str(files[-1].relative_to(REPO))


def lars_rows():
    """PQ, PQ_th, PQ_st, mIoU: official LaRS evaluator (val); box AP: pycocotools on the 9 classes of L9."""
    main = []
    for model, code, size, off, apf, wb in [
        ("EoMT zero-shot", "F1", f"{EOMT}", "lars_official_eomt_zeroshot_tightened_28617624.json",
         f"{FF}/fix1_coco133_lars_map-tightened_28511590.json", []),
        ("EoMT, full fine-tuning", "F4", f"{EOMT}", "lars_official_eomt_full_28617625.json",
         f"{FF}/fix1_lars_panoptic_lars_eomt_full_28511607.json", runs("yphvg0cr", "zxokqelg")),
        ("EoMT, LoRA", "F4", f"{EOMT} + {LORA}", "lars_official_eomt_lora_28617626.json",
         f"{FF}/fix1_lars_panoptic_lars_eomt_lora_28511609.json", runs("8xdisbps", "7wrw1rkt")),
        ("EoMT + SAM 2.1-L (control)", "F1", f"{EOMT} + {SAM}", "lars_official_eomt_ab03_sam_28617629.json", None,
         [("ab03", WB + "67kk33hk")]),
        ("YOLOv11s + SAM 2.1-L", "F2", f"{Y11S} + {SAM}", "lars_official_yolo11s_sam_t020_28617630.json",
         f"{AB10}/eval_lars_full_11s_11cls_stageB_28515717.json",
         runs("lars_full_11s_11cls_stageB_20261023_210948", "lars_full_11s_11cls_stageA_20261023_205324")),
    ]:
        off = f"{FF}/{off}"
        ap = None
        if apf:
            d = json.loads((REPO / apf).read_text())
            ap = 100 * (d["checkpoint"]["pycocotools_all_9_classes"]["mAP"] if "checkpoint" in d else d["box_AP"]["all_9_classes"]["mAP"])
        main.append(row(model, code, size, official(off) + [ap], [off] * 4 + [apf], wb))
    apf = f"{AB10}/eval_lars_full_11s_stageB_28378811.json"
    main.append(row("YOLOv11s", "A10", f"{Y11S}", [None] * 4 + [100 * jget(apf, "checkpoint", "pycocotools_all_9_classes", "mAP")],
                    [None] * 4 + [apf], runs("lars_full_11s_stageB_20261020_170700", "lars_full_11s_stageA_20261020_161452")))
    a11 = []
    for r in ab11_runs("lars11"):
        off = rel11(f"lars_official_ab11_lars11_{r['model']}_*.json")
        det = r["task"] == "detect"
        name = f"YOLO26{r['size']}" + (" + SAM 2.1-L" if det else "-seg")
        a11.append(row(name, "A11", f"{r['params_M']:.1f}" + (f" + {SAM}" if det else ""), official(off) + [None],
                       [off] * 4 + [None], [("train", r["wandb"])]))
    pub = [row(m, None, "--", [pq, th, st, miou, None], [LARS_PAPER] * 4 + [None]) for m, pq, th, st, miou in LARS_PUB]
    return [("LaRS paper", LARS_PAPER, "‡", pub), ("Ours", None, "", main), ("Ours (A11)", None, "†", a11)]


def flow_rows():
    main = []
    for model, code, size, f, path, wb in [
        ("EoMT (SAM masks), full fine-tuning", "F3", f"{EOMT}", "fix1_box_instance_flow_eomt_full_28547302.json", "box", runs("whcbnw3e", "tdxrxg6w")),
        ("EoMT (SAM masks), LoRA", "F4", f"{EOMT} + {LORA}", "fix1_box_instance_flow_eomt_lora_28547305.json", "box", runs("eg9pabqv", "5oz1099z")),
        ("YOLOv11s", "F3", f"{Y11S}", "eval_flow_full_11s_stageB_28511601.json", "ckpt",
         runs("flow_full_11s_stageB_20261023_195008", "flow_full_11s_stageA_20261023_194420")),
    ]:
        f = f"{FF}/{f}"
        ap = jget(f, "box_AP", "class_agnostic", "mAP") if path == "box" else jget(f, "checkpoint", "pycocotools_all_9_classes", "mAP")
        main.append(row(model, code, size, [100 * ap], [f], wb))
    a11 = [row(f"YOLO26{r['size']}", "A11", f"{r['params_M']:.1f}", [100 * r["y"]], [r["source"]], [("train", r["wandb"])])
           for r in ab11_runs("flow")]
    pub = [row(m, None, "--", [ap], [FLOW_PAPER]) for m, ap in FLOW_PUB]
    return [("FloW paper", FLOW_PAPER, "", pub), ("Ours", None, "", main), ("Ours (A11)", None, "†", a11)]


def cansurf_rows():
    main = []
    for model, code, size, f, path, pf, wb in [
        ("EoMT (SAM masks), full fine-tuning", "F3", f"{EOMT}", "fix1_box_instance_cansurf_eomt_full_28547307.json", "box",
         "cansurf_paper_eomt_cansurf_full_28630629.json", runs("8188sdjb", "p8mi8h0k")),
        ("EoMT (SAM masks), LoRA", "F4", f"{EOMT} + {LORA}", "fix1_box_instance_cansurf_eomt_lora_28547309.json", "box",
         "cansurf_paper_eomt_cansurf_lora_28630629.json", runs("ip8at1js", "75g0ocvd")),
        ("YOLOv11s", "F3", f"{Y11S}", "eval_cansurf_full_11s_stageB_28511603.json", "ckpt",
         "cansurf_paper_yolo11s_cansurf_28630998.json",
         runs("cansurf_full_11s_stageB_20261023_200247", "cansurf_full_11s_stageA_20261023_194423")),
    ]:
        f, pf = f"{FF}/{f}", f"{FF}/{pf}"
        ap = jget(f, "box_AP", "class_agnostic", "mAP") if path == "box" else jget(f, "checkpoint", "pycocotools_all_9_classes", "mAP")
        main.append(row(model, code, size, [100 * ap] + paper_prf(pf), [f] + [pf] * 3, wb))
    a11 = []
    for r in ab11_runs("cansurf"):
        pf = rel11(f"cansurf_paper_ab11_{r['model']}_*.json")
        a11.append(row(f"YOLO26{r['size']}", "A11", f"{r['params_M']:.1f}", [100 * r["y"]] + paper_prf(pf),
                       [r["source"]] + [pf] * 3, [("train", r["wandb"])]))
    pub = [row(m, None, size, [None, pr, rc, f], [None] + [CANSURF_PAPER] * 3) for m, size, pr, rc, f in CANSURF_PUB]
    return [("CANSURF paper", CANSURF_PAPER, "§", pub), ("Ours", None, "", main), ("Ours (A11)", None, "†", a11)]


TABLES = [  # key, title, rows(), metric headers [(name, class-set code or None)], LaTeX star / fit, caption (plain text)
    ("lars", "LaRS", lars_rows, [("PQ", "L11"), ("PQ_Th", None), ("PQ_St", None), ("mIoU", "L3"), ("Box AP", "L9")], True, True,
     "LaRS. PQ, PQ_Th / PQ_St (things / stuff) and mIoU: official LaRS evaluator; ours on LaRS val (198 images, YOLO26 "
     "at confidence 0.20). ‡Published baselines from the LaRS paper (Table 4: PQ; Table 5: mIoU under the semantic "
     "setup) on the LaRS test set via the evaluation server: a different split from ours, so the comparison is "
     "indicative. Box AP: pycocotools mAP@[.50:.95] on boxes traced from the val masks."),
    ("flow", "FloW", flow_rows, [("Box AP", "waste")], False, False,
     "FloW-Img test split (800 images). Box AP: mAP@[.50:.05:.95] (pycocotools for ours; the paper's mmAP, its Table 2, "
     "same split)."),
    ("cansurf", "CANSURF", cansurf_rows, [("Box AP", "can"), ("P", None), ("R", None), ("F1", None)], True, True,
     "CANSURF. Ours on CANSURF val (2,209 images): box AP = pycocotools mAP@[.50:.95]; P / R / F1 with the CANSURF "
     "paper's protocol (its script; confidence > 0.25, counts summed over IoU 0.50:0.05:0.95). §Published results from "
     "the CANSURF paper (Table II) on its curated 900-image test set (cans < 5% of the frame): a different, harder image "
     "set, so the comparison is indicative; the paper reports no box AP."),
]
COMMON = ("Params: + SAM 2.1-L where its masks are scored, + LoRA adapters. †A11: checkpoint picked on the evaluation "
          "split (optimistic); other rows of ours use held-out epoch selection. --: not reported or not applicable.")


def fmt(x) -> str:
    """One decimal; published values given as integers (CANSURF paper: two-decimal fractions) keep their precision."""
    return "--" if x is None else str(x) if isinstance(x, int) else f"{x:.1f}"


# ---------------------------------------------------------------- LaTeX

def tex(s: str) -> str:
    for a, b in (("%", "\\%"), ("→", "$\\to$"), ("n–x", "n--x"), ("< ", "$<$ "), ("> ", "$>$ "), ("‡", "$^\\ddagger$"),
                 ("†", "$^\\dagger$"), ("§", "$^\\S$"), ("PQ_Th", "PQ$_\\mathrm{Th}$"), ("PQ_St", "PQ$_\\mathrm{St}$")):
        s = s.replace(a, b)
    return s


def link(kind: str, code: str, text: str | None = None) -> str:
    return f"\\hyperlink{{{kind}:{code}}}{{{text or code}}}"


def anchor(kind: str, code: str) -> str:
    """Empty link target raised to the top of the row (a target on the baseline scrolls the row out of view), followed
    by the visible bold code; the code is printed once, outside the target."""
    return f"\\raisebox{{\\ht\\strutbox}}{{\\hypertarget{{{kind}:{code}}}{{}}}}\\textbf{{{code}}}"


def tex_results(key, title, rows_fn, head, star, fit, caption) -> str:
    """Grouped first column (source of the row), vertical rule, model (linked to its code), params, metrics."""
    n, env = len(head), "table*" if star else "table"
    cols = [tex(m) + (f" ({link('cls', c)})" if c else "") for m, c in head]
    L = [f"\\begin{{{env}}}[t]", "\\centering"] + (["\\resizebox{\\textwidth}{!}{%"] if fit else ["\\small"]) + [
        f"\\begin{{tabular}}{{l|lr{'c' * n}}}", "\\toprule",
        f"\\multirow{{2}}{{*}}{{Source}} & \\multirow{{2}}{{*}}{{Model}} & \\multirow{{2}}{{*}}{{Params (M)}} & "
        f"\\multicolumn{{{n}}}{{c}}{{Metric (\\%)}} \\\\",
        f"\\cmidrule(lr){{4-{3 + n}}}", " & & & " + " & ".join(cols) + " \\\\", "\\midrule"]
    groups = rows_fn()
    for gi, (glabel, url, mark, rows) in enumerate(groups):
        gl = (f"\\href{{{url}}}{{{glabel}}}" if url else glabel) + tex(mark)
        for j, r in enumerate(rows):
            L.append("% source: " + "; ".join(dict.fromkeys(s for s in r["srcs"] if s)))
            g = f"\\multirow{{{len(rows)}}}{{*}}{{{gl}}}" if j == 0 else ""
            name = link("abl", r["code"], r["model"]) if r["code"] else r["model"]
            L.append(f"{g} & {name} & {r['size']} & " + " & ".join(fmt(v) for v in r["vals"]) + " \\\\")
        L.append("\\midrule" if gi < len(groups) - 1 else "\\bottomrule")
    cap = tex(caption) + (" Model names link to their ablation or fix (Table~\\ref{tab:ablations}); class-set codes link to "
                          "Table~\\ref{tab:classsets}. ") + tex(COMMON)
    L += ["\\end{tabular}" + ("}" if fit else ""), f"\\caption{{{cap}}}", f"\\label{{tab:results_{key}}}", f"\\end{{{env}}}"]
    return "\n".join(L)


def tex_classsets() -> str:
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{lp{0.7\\linewidth}}", "\\toprule",
         "Code & Classes \\\\", "\\midrule"]
    L += [f"{anchor('cls', c)} & {tex(d)} \\\\" for c, d in CLASS_SETS]
    L += ["\\bottomrule", "\\end{tabular}", "\\caption{Class sets scored.}", "\\label{tab:classsets}", "\\end{table}"]
    return "\n".join(L)


def tex_ablations() -> str:
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{lp{0.75\\linewidth}}", "\\toprule",
         "Code & Description \\\\", "\\midrule"]
    for i, (c, d, _) in enumerate(ABLATIONS):
        if i and c.startswith("F") and not ABLATIONS[i - 1][0].startswith("F"):
            L.append("\\midrule")
        L.append(f"{anchor('abl', c)} & {tex(d)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}",
          "\\caption{Ablations (A) and fairness fixes (F) referenced in Tables~\\ref{tab:results_lars}--\\ref{tab:results_cansurf}.}",
          "\\label{tab:ablations}", "\\end{table}"]
    return "\n".join(L)


# ---------------------------------------------------------------- Section 2: A11 figures

A11_DIR = "ablations/11_yolo26_family"  # relative to EO
A11_COMMON = ("YOLO26 n/s/m/l/x, 50 epochs; each point is best.pt, the epoch with the best box mAP50-95 "
              "on this same split, so accuracy is optimistic. x: median end-to-end latency of the YOLO model alone "
              "(Ultralytics pre-processing + inference + post-processing), TensorRT FP16, batch 1, imgsz 640, A100, all "
              "models timed in one job.")
FIGURES = [  # key, png (uploaded flat to Overleaf; generated by plot_yolo26_family.py), caption
    ("lars", "ab11_lars_latency.png",
     "LaRS val (198 images, 11 classes): PQ vs latency. Detection: boxes $\\rightarrow$ SAM 2.1-L masks (SAM time not in x); "
     "-seg: the model's own masks, checkpoint picked on box + mask mAP50-95. "
     "PQ is from eval\\_sam\\_panoptic.py at confidence 0.20 (the YOLOv11s + SAM operating "
     "point), 0.2--0.6 lower than the official LaRS PQ in Table~\\ref{tab:results_lars}."),
    ("flow", "ab11_flow_latency.png", "FloW test (800 images): pycocotools box mAP50-95 vs latency."),
    ("cansurf", "ab11_cansurf_latency.png", "CANSURF val (2,209 images): pycocotools box mAP50-95 vs latency."),
]


def tex_figures() -> str:
    L = []
    for key, png, cap in FIGURES:
        L += [f"% source: eval_outputs/water_debris_seg/{A11_DIR}/{png}; values: eval_outputs/water_debris_seg/{A11_DIR}/ab11_results.json",
              "\\begin{figure}[!htb]", "\\centering", f"\\includegraphics[width=0.85\\linewidth]{{{png}}}",
              f"\\caption[{cap.split(':')[0]}]{{{cap} {tex(A11_COMMON)} Ablation {link('abl', 'A11')}.}}", f"\\label{{fig:a11_{key}}}",
              "\\end{figure}", ""]
    return "\n".join(L)


TEX_PRE = ("% Generated by water-debris-seg/scripts/make_latex_tables.py (Markdown with every source and W&B link:\n"
           "% eval_outputs/water_debris_seg/BENCHMARK_TABLES.md). Sources of every results row are in the comment above it.\n"
           "% Preamble: \\usepackage{booktabs,multirow,graphicx} and \\usepackage{hyperref} (load hyperref last).\n"
           "% Without them LaTeX prints the macro arguments as text (e.g. '2*Dataset', 'cls:L11L11'); stop with a clear\n"
           "% error instead.\n"
           "\\makeatletter\n"
           "\\@for\\bt@pkg:={booktabs,multirow,graphicx,hyperref}\\do{\\@ifundefined{ver@\\bt@pkg.sty}{%  (\\@ifpackageloaded is preamble-only)\n"
           "  \\PackageError{benchmark_tables}{Package \\bt@pkg\\space not loaded}"
           "{Add \\protect\\usepackage{booktabs,multirow,graphicx} and \\protect\\usepackage{hyperref} to the preamble.}}{}}\n"
           "\\makeatother\n")


# ---------------------------------------------------------------- Markdown

def md_link(src: str | None) -> str | None:
    if not src:
        return None
    return src if src.startswith("http") else os.path.relpath(REPO / src, EO)


def md_anchor(kind: str, code: str) -> str:
    return f"{kind}-{code.lower()}"


def md_results(key, title, rows_fn, head, star, fit, caption) -> str:
    cols = [m.replace("_Th", "<sub>Th</sub>").replace("_St", "<sub>St</sub>") + (f" ([{c}](#{md_anchor('cls', c)}))" if c else "")
            for m, c in head]
    L = [f"### {title}", "", "| Source | Model | Code | Params (M) | " + " | ".join(cols) + " | W&B |",
         "|---|---|---|---:|" + "---:|" * len(head) + "---|"]
    for glabel, url, mark, rows in rows_fn():
        g = (f"[{glabel}]({url})" if url else glabel) + mark
        for r in rows:
            vals = []
            for v, s in zip(r["vals"], r["srcs"]):
                href = md_link(s)
                vals.append(f"[{fmt(v)}]({href})" if href and v is not None else fmt(v))
            code = f"[{r['code']}](#{md_anchor('abl', r['code'])})" if r["code"] else ""
            wb = " · ".join(f"[{lab}]({u})" for lab, u in r["wandb"]) or ("paper" if url else "no training (COCO weights)")
            L.append(f"| {g} | {r['model']} | {code} | {r['size']} | " + " | ".join(vals) + f" | {wb} |")
    L += ["", caption + " " + COMMON, ""]
    return "\n".join(L)


def md() -> str:
    L = ["# Benchmark tables", "",
         "Generated by `water-debris-seg/scripts/make_latex_tables.py` together with the LaTeX version "
         "([latex/benchmark_tables.tex](latex/benchmark_tables.tex)); do not edit by hand.", "",
         "**Auditing.** Every value of ours links to the JSON it was read from; published values link to their paper. "
         "**Code** links to the ablation / fix table below, which links to that page. **W&B**: B = the scored stage-B "
         "training run (held-out protocol), A = its stage-A run (epoch-count selection), train = the ablation 11 training "
         "run (its best.pt is scored), ab03 = the ablation 03 run whose segments the EoMT + SAM control uses. Values are "
         "% (fractions × 100).", "", "## 1. Benchmark results", ""]
    for t in TABLES:
        L.append(md_results(*t))
    L += ["### Class sets", "", "| Code | Classes |", "|---|---|"]
    L += [f"| <a id=\"{md_anchor('cls', c)}\"></a>**{c}** | {d} |" for c, d in CLASS_SETS]
    L += ["", "Source: [fairness_fixes/README.md, class codes](fairness_fixes/README.md#main-comparison-table-decisions-enforced).", "",
          "### Ablations and fixes", "", "| Code | Description | Page |", "|---|---|---|"]
    L += [f"| <a id=\"{md_anchor('abl', c)}\"></a>**{c}** | {d} | [{Path(p).parent.name if c[0] == 'A' else Path(p).stem}]({p}) |"
          for c, d, p in ABLATIONS]
    L += ["", "## 2. YOLO26 family: accuracy vs latency", "",
          f"Ablation [A11](#{md_anchor('abl', 'A11')}); generated by `water-debris-seg/scripts/plot_yolo26_family.py`, "
          f"values with sources in [ab11_results.json]({A11_DIR}/ab11_results.json), W&B runs in the A11 rows above. "
          + A11_COMMON, ""]
    for key, png, cap in FIGURES:
        cap = (cap.replace("$\\rightarrow$", "→").replace("\\_", "_").replace("--", "–")
               .replace("Table~\\ref{tab:results_lars}", "the LaRS table above"))
        L += [f"![{key}]({A11_DIR}/{png})", "", cap, ""]
    return "\n".join(L) + "\n"


def main() -> None:
    TEX.parent.mkdir(exist_ok=True)
    TEX.write_text(TEX_PRE + "\\section{Benchmark results}\n\\label{sec:results}\n\n"
                   + "\n\n".join([tex_results(*t) for t in TABLES] + [tex_classsets(), tex_ablations()])
                   + "\n\n\\clearpage\n\\section{YOLO26 family: accuracy vs latency}\n\\label{sec:a11}\n"
                   "% Upload the three PNGs to the Overleaf project root (or set \\graphicspath).\n"
                   "Figures~\\ref{fig:a11_lars}--\\ref{fig:a11_cansurf} show YOLO26 n--x on the three datasets "
                   f"(ablation {link('abl', 'A11')}; values in the A11 rows of Tables~\\ref{{tab:results_lars}}--\\ref{{tab:results_cansurf}}).\n\n"
                   + tex_figures())
    MD.write_text(md())
    for p in (TEX, MD):
        print(p)


if __name__ == "__main__":
    main()
