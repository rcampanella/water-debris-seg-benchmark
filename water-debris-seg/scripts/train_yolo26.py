#!/usr/bin/env python3
"""Ablation 11: YOLO26 / YOLO26-seg family, accuracy vs latency on LaRS, FloW and CANSURF.

One run = one pretrained COCO model (yolo26{n,s,m,l,x}[-seg].pt, data/water_debris/checkpoints/yolo) fine-tuned for
--epochs (default 50) on the full train split, with the TEST split as Ultralytics' val set. There is no held-out split
(user decision, 2026-10-30): the test split is scored after every epoch and best.pt is the epoch with the highest
Ultralytics fitness on it (8.4: box mAP50-95; seg: box mAP50-95 + mask mAP50-95). Scores are therefore optimistic and
not comparable with the held-out main table; last.pt and the per-epoch curve are recorded next to them.

  lars     9 classes (ablation 10 boxes / prepare_lars_yolo_seg.py polygons), train (2,605) -> LaRS val (198)
  flow     1 class, train_all (1,200) -> FloW test (800)
  cansurf  1 class, train_all (4,950) -> CANSURF val (2,221 YOLO label files; our COCO GT drops 12 of them)
  lars11   as lars + Water and Sky (11 classes; prepare_lars_yolo.py / prepare_lars_yolo_seg.py --with-water-sky), so
           predictions cover the whole image and LaRS PQ can be scored (eval_sam_panoptic.py; 2026-11-02)
  seg models on LaRS only (FloW / CANSURF have no GT masks; user decision).

After training: best.pt and last.pt validated on the test split; parameters / GFLOPs; latency = Ultralytics
end-to-end (preprocess + inference + postprocess) per image, PyTorch FP16, batch 1, imgsz 640, on --latency-images
test images after --warmup, reported as median / mean / p90. Writes <ablation>/sources/ab11_<name>_<job>.json (name defaults to <dataset>_<model>).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ABL = REPO / "eval_outputs/water_debris_seg/ablations/11_yolo26_family"
CKPT = REPO / "data/water_debris/checkpoints/yolo"
SCRATCH_LARS = Path("/scratch-shared/rcampanella/water-debris-seg/data/lars")
SPLITS = REPO / "data/water_debris/yolo_splits"
LARS_NAMES = ["Static Obstacle", "Boat/ship", "Row boats", "Paddle board", "Buoy", "Swimmer", "Animal", "Float", "Other"]
LARS11_NAMES = LARS_NAMES + ["Water", "Sky"]
# dataset -> (root, train dir, test dir, class names); the test dir goes in the yaml's `val` key
DATASETS = {
    ("lars", "detect"): (SCRATCH_LARS / "yolo_boxes", "train_all", "val", LARS_NAMES),
    ("lars", "segment"): (SCRATCH_LARS / "yolo_seg", "train_all", "val", LARS_NAMES),
    ("lars11", "detect"): (REPO / "data/water_debris/lars_yolo_11cls", "train_all", "val", LARS11_NAMES),
    ("lars11", "segment"): (SCRATCH_LARS / "yolo_seg11", "train_all", "val", LARS11_NAMES),
    ("flow", "detect"): (SPLITS / "flow", "train_all", "test", ["waste"]),
    ("cansurf", "detect"): (SPLITS / "cansurf", "train_all", "test", ["debris"]),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def metrics(m, seg: bool) -> dict:
    out = {"box_mAP50_95": float(m.box.map), "box_mAP50": float(m.box.map50), "box_P": float(m.box.mp),
           "box_R": float(m.box.mr), "fitness": float(m.fitness)}
    if seg:
        out.update(mask_mAP50_95=float(m.seg.map), mask_mAP50=float(m.seg.map50))
    out["per_class_box_mAP50_95"] = {m.names[c]: float(v) for c, v in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index])}
    return out


def curve(results_csv: Path, seg: bool) -> dict:
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(results_csv.open())]
    keys = ["metrics/mAP50-95(B)", "metrics/mAP50(B)"] + (["metrics/mAP50-95(M)", "metrics/mAP50(M)"] if seg else [])
    per = [{"epoch": int(float(r["epoch"])), **{k: float(r[k]) for k in keys if k in r}} for r in rows]
    plotted = "metrics/mAP50-95(M)" if seg else "metrics/mAP50-95(B)"
    b = max(per, key=lambda r: r[plotted])
    # epoch behind best.pt (strip_optimizer resets its stored epoch to -1): first max of Ultralytics' fitness
    fit = [r["metrics/mAP50-95(B)"] + (r["metrics/mAP50-95(M)"] if seg else 0.0) for r in per]
    sel = per[int(np.argmax(fit))]
    return {"source": str(results_csv), "epochs_run": len(per), "selected_epoch": sel["epoch"], "selected_fitness": max(fit),
            "best_epoch_on_plotted_metric": b["epoch"],
            "best_plotted_metric": b[plotted], "plotted_metric": plotted, "per_epoch": per}


def latency(weights: Path, images: list[Path], warmup: int) -> dict:
    import torch
    from ultralytics import YOLO

    model = YOLO(str(weights))
    pre, inf, post = [], [], []
    for i, p in enumerate(images):
        r = model.predict(str(p), imgsz=640, half=True, device=0, batch=1, verbose=False)[0]
        if i >= warmup:
            pre.append(r.speed["preprocess"]), inf.append(r.speed["inference"]), post.append(r.speed["postprocess"])
    tot = np.array(pre) + np.array(inf) + np.array(post)
    st = lambda a: {"median": float(np.median(a)), "mean": float(np.mean(a)), "p90": float(np.percentile(a, 90))}
    return {"end_to_end_ms": st(tot), "preprocess_ms": st(pre), "inference_ms": st(inf), "postprocess_ms": st(post),
            "n_timed": len(tot), "warmup": warmup, "setting": "PyTorch FP16 (half=True), batch 1, imgsz 640, Ultralytics Results.speed",
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cpu": platform.processor() or platform.machine()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["lars", "lars11", "flow", "cansurf"], required=True)
    ap.add_argument("--model", required=True, help="e.g. yolo26s or yolo26s-seg")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0, help="train fraction (smoke tests only)")
    ap.add_argument("--latency-images", type=int, default=220)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO, settings
    settings.update({"wandb": True})
    task = "segment" if args.model.endswith("-seg") else "detect"
    root, tr, te, names = DATASETS[(args.dataset, task)]
    name = args.name or f"{args.dataset}_{args.model}"
    job = os.environ.get("SLURM_JOB_ID", "local")
    runs, src = ABL / "runs", ABL / "sources"
    for d in (runs, src, ABL / "data"):
        d.mkdir(parents=True, exist_ok=True)
    data_yaml = ABL / "data" / f"{args.dataset}_{task}.yaml"
    data_yaml.write_text(f"path: {root}\ntrain: {tr}/images\nval: {te}/images\nnc: {len(names)}\nnames: {json.dumps(names)}\n")
    weights0 = CKPT / f"{args.model}.pt"
    os.chdir(runs)
    os.environ.setdefault("WANDB_RUN_GROUP", "ablation11_yolo26")

    model = YOLO(str(weights0))
    wb = {}

    def keep_wandb_url(trainer) -> None:   # on_train_start: the W&B run exists; Ultralytics finishes it on train end
        import wandb
        wb["url"] = wandb.run.url if wandb.run else None

    model.add_callback("on_train_start", keep_wandb_url)
    model.train(data=str(data_yaml), epochs=args.epochs, patience=args.epochs, imgsz=args.imgsz, batch=args.batch,
                seed=args.seed, deterministic=True, device=0, project="water-debris-seg", name=name, exist_ok=True,
                fraction=args.fraction, plots=True, cache="ram")
    actual = Path(model.trainer.save_dir).resolve()
    out = runs / name
    if out.is_symlink() or out.exists():
        out.unlink()
    out.symlink_to(actual, target_is_directory=True)
    seg = task == "segment"
    res = {"dataset": args.dataset, "model": args.model, "task": task, "job": job, "run_dir": str(out.relative_to(REPO)),
           "data_yaml": str(data_yaml.relative_to(REPO)), "train_split": str(root / tr), "test_split": str(root / te),
           "pretrained": str(weights0.relative_to(REPO)), "pretrained_sha256": sha(weights0),
           "selection": "best.pt = epoch with the highest Ultralytics fitness on the TEST split "
                        "(detect: box mAP50-95; segment: box + mask mAP50-95); no held-out split",
           "train_args": {k: v for k, v in vars(model.trainer.args).items() if isinstance(v, (int, float, str, bool, type(None)))},
           "wandb_train_run": wb.get("url"), "curve": curve(out / "results.csv", seg)}
    for ck in ("best", "last"):
        w = out / "weights" / f"{ck}.pt"
        m = YOLO(str(w))
        v = m.val(data=str(data_yaml), split="val", imgsz=args.imgsz, batch=16, device=0, plots=False, verbose=False,
                  project=str(actual), name=f"val_{ck}", exist_ok=True)
        res[ck] = {"weights": str(w.relative_to(REPO)), "sha256": sha(w),
                   "epoch": res["curve"]["selected_epoch"] if ck == "best" else res["curve"]["epochs_run"],
                   **metrics(v, seg)}
    from ultralytics.utils.torch_utils import get_flops, get_num_params
    net = YOLO(str(out / "weights" / "best.pt")).model   # unfused, as the model summary printed at load
    params, gflops = get_num_params(net), get_flops(net, args.imgsz)
    res.update(params=int(params), gflops=float(gflops))
    imgs = sorted((root / te / "images").iterdir())[: args.latency_images]
    res["latency"] = latency(out / "weights" / "best.pt", imgs, args.warmup)
    res["plot_y"] = {"metric": "mask_mAP50_95" if seg else "box_mAP50_95", "value": res["best"]["mask_mAP50_95" if seg else "box_mAP50_95"]}
    dst = src / f"ab11_{name}_{job}.json"   # smoke runs (NAME=smoke_...) stay out of the plots
    dst.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("dataset", "model", "plot_y", "params", "gflops")}),
          "best epoch", res["best"]["epoch"], "latency", res["latency"]["end_to_end_ms"], "->", dst, flush=True)


if __name__ == "__main__":
    main()
