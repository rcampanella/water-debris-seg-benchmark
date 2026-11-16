#!/usr/bin/env python3
"""D01 fine-tuning: one released YOLO26-depth model (n..x) fine-tuned on one D01 source (README D2-D5, D12).

As ablation 11 (train_yolo26.py): 50 epochs, imgsz 640, batch 16, seed 0, deterministic, Ultralytics 8.4.173 defaults
(depth loss dlog 1.0 / dgrad 0.5 / dlam 1.0), the TEST split (day test) as Ultralytics' val set, so best.pt = epoch
with the best Ultralytics depth fitness on it (D4: optimistic, labelled); Ultralytics' post-training calibration is
also fitted on it. Dataset: data/water_debris/depth_d01/<source>.yaml (prepare_depth_dataset.py; max_depth 20).
After training: eval_depth.py (fine-tuned arm: shipped calibration + D12 refit on test_day) for best.pt and last.pt
on test_day and test_night. Writes eval_outputs/water_debris_depth/ablations/D01_rgb/sources/finetune/d01_<source>_<model>_<job>.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "eval_outputs/water_debris_depth/ablations/D01_rgb/sources/finetune"
DATA = REPO / "data/water_debris/depth_d01"
CKPT = REPO / "data/water_debris/checkpoints/yolo/depth"
K = {"pohang": "1814.506142073043,1814.506142073043,1026.197713069134,524.7984412358129",
     "multiaqua": "1365.43274,1365.43274,1094.44934,624.207886"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def curve(results_csv: Path) -> dict:
    rows = list(csv.DictReader(open(results_csv)))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    fit = next((k for k in rows[0] if "delta1" in k), None)   # Ultralytics depth fitness = metrics/delta1 (median-aligned)
    vals = [float(r[fit]) for r in rows] if fit else []
    best = int(max(range(len(vals)), key=lambda i: vals[i])) + 1 if vals else None
    return {"fitness_column": fit, "per_epoch": rows, "epochs_run": len(rows), "selected_epoch": best}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["pohang", "multiaqua"], required=True)
    ap.add_argument("--model", required=True, help="yolo26{n,s,m,l,x}-depth")
    ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=1.0); ap.add_argument("--name", default=None)
    a = ap.parse_args()
    from ultralytics import YOLO, settings
    settings.update({"wandb": True})
    job = os.environ.get("SLURM_JOB_ID", "local")
    name = a.name or f"d01_{a.source}_{a.model}"
    data_yaml = DATA / f"{a.source}.yaml"
    runs = REPO / "eval_outputs/water_debris_depth/ablations/D01_rgb/runs"; runs.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(runs)
    os.environ.setdefault("WANDB_RUN_GROUP", "depth_D01")
    w0 = CKPT / f"{a.model}.pt"
    model = YOLO(str(w0)); wb = {}

    def keep_wandb_url(trainer) -> None:
        import wandb
        wb["url"] = wandb.run.url if wandb.run else None
    model.add_callback("on_train_start", keep_wandb_url)
    model.train(data=str(data_yaml), epochs=a.epochs, patience=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed,
                deterministic=True, device=0, project="water-debris-seg", name=name, exist_ok=True,
                fraction=a.fraction, plots=True)
    actual = Path(model.trainer.save_dir).resolve()
    res = {"source": a.source, "model": a.model, "job": job, "run_dir": str(actual), "data_yaml": str(data_yaml),
           "data_yaml_sha256": sha(data_yaml), "pretrained": str(w0), "pretrained_sha256": sha(w0),
           "selection": "best.pt = epoch with the best Ultralytics depth fitness (delta1, median-aligned) on the day TEST split (D4)",
           "train_args": {k: v for k, v in vars(model.trainer.args).items() if isinstance(v, (int, float, str, bool, type(None)))},
           "wandb_train_run": wb.get("url"), "curve": curve(actual / "results.csv"), "scores": {}}
    for ck in ("best", "last"):
        w = actual / "weights" / f"{ck}.pt"
        res[ck] = {"weights": str(w), "sha256": sha(w)}
        for split in ("test_day", "test_night"):
            pref = OUT / f"ft_{a.source}_{a.model}_{ck}_{split}_{job}"
            subprocess.run([sys.executable, str(REPO / "water-debris-seg/scripts/depth/eval_depth.py"), "--data", str(DATA / a.source),
                            "--split", split, "--weights", str(w), "--arm", "fine-tuned", "--K", K[a.source],
                            "--refit-split", "test_day", "--device", "0", "--out", str(pref)], check=True)
            res["scores"][f"{ck}_{split}"] = str(pref) + ".json"
    from ultralytics.utils.torch_utils import get_flops, get_num_params
    net = YOLO(str(actual / "weights" / "best.pt")).model
    res.update(params=int(get_num_params(net)), gflops=float(get_flops(net, a.imgsz)))
    dst = OUT / f"d01_{a.source}_{a.model}_{job}.json"
    dst.write_text(json.dumps(res, indent=2))
    print("done", name, "best epoch", res["curve"]["selected_epoch"], "->", dst, flush=True)


if __name__ == "__main__":
    main()
