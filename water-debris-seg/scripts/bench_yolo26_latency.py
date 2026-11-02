#!/usr/bin/env python3
"""Ablation 11: end-to-end latency of every trained YOLO26 / YOLO26-seg checkpoint, all on one GPU in one job.

The per-run latency in train_yolo26.py was measured inside each training job (shared nodes, PyTorch FP16), so runs
are timed on different GPUs next to other jobs, and at batch 1 PyTorch's fixed per-call overhead hides the size of
the model. This script times every model the same way, one after another in one exclusive job, with two backends:
  pytorch   best.pt, half=True                                    (as train_yolo26.py)
  tensorrt  best.pt exported by Ultralytics to a TensorRT FP16 engine (batch 1, imgsz 640), as Ultralytics' own
            YOLO26 latency charts (T4 TensorRT10 FP16); the engine is built in a scratch directory
End-to-end = Ultralytics Results.speed preprocess + inference + postprocess per image (image decoding excluded), batch 1,
imgsz 640, on the first --images test images of the run's dataset after --warmup images; median / mean / p90.
Models: the newest ab11_<dataset>_<model>_<job>.json per (dataset, model) in sources/. Writes
sources/ab11_latency_<job>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ABL = REPO / "eval_outputs/water_debris_seg/ablations/11_yolo26_family"
PAT = re.compile(r"ab11_(lars11|flow|cansurf)_(yolo26[nslmx](?:-seg)?)_(\d+)\.json$")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def time_model(weights: str, images: list[Path], warmup: int, half: bool) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights, task=None)
    tot, inf = [], []
    for i, p in enumerate(images):
        r = model.predict(str(p), imgsz=640, half=half, device=0, batch=1, verbose=False)[0]
        if i >= warmup:
            tot.append(r.speed["preprocess"] + r.speed["inference"] + r.speed["postprocess"])
            inf.append(r.speed["inference"])
    st = lambda a: {"median": float(np.median(a)), "mean": float(np.mean(a)), "p90": float(np.percentile(a, 90))}
    return {"end_to_end_ms": st(tot), "inference_ms": st(inf), "n_timed": len(tot)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=int, default=220)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--backends", nargs="+", default=["pytorch", "tensorrt"])
    ap.add_argument("--only", nargs="+", help="dataset/model keys to time (smoke tests), e.g. flow/yolo26n")
    args = ap.parse_args()

    import torch
    import ultralytics

    latest = {}
    for p in sorted((ABL / "sources").glob("ab11_*.json")):
        m = PAT.match(p.name)
        if m and ((m[1], m[2]) not in latest or int(m[3]) > int(PAT.match(latest[(m[1], m[2])].name)[3])):
            latest[(m[1], m[2])] = p
    job = os.environ.get("SLURM_JOB_ID", "local")
    dst = ABL / "sources" / f"ab11_latency_{job}.json"
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
           "setting": f"batch 1, imgsz 640, first {args.images} test images, {args.warmup} warm-up; "
                      "Ultralytics Results.speed preprocess + inference + postprocess", "models": {}}
    try:
        import tensorrt

        out["tensorrt"] = tensorrt.__version__
    except ImportError:
        args.backends = [b for b in args.backends if b != "tensorrt"]
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        for (ds, model), rj in sorted(latest.items()):
            if args.only and f"{ds}/{model}" not in args.only:
                continue
            r = json.loads(rj.read_text())
            best = REPO / r["best"]["weights"]
            images = sorted(Path(r["test_split"], "images").iterdir())[: args.images]
            entry = {"run_json": str(rj.relative_to(REPO)), "weights": str(best.relative_to(REPO)), "weights_sha256": sha(best),
                     "task": r["task"]}
            for b in args.backends:
                if b == "pytorch":
                    entry[b] = time_model(str(best), images, args.warmup, half=True)
                else:
                    w = Path(tmp) / f"{ds}_{model}.pt"
                    shutil.copy(best, w)
                    from ultralytics import YOLO

                    eng = YOLO(str(w)).export(format="engine", half=True, imgsz=640, batch=1, device=0, verbose=False)
                    entry[b] = {**time_model(eng, images, args.warmup, half=True), "engine_sha256": sha(Path(eng))}
            out["models"][f"{ds}/{model}"] = entry
            dst.write_text(json.dumps(out, indent=2))   # after every model: a timeout keeps what was timed
            print(ds, model, {b: round(entry[b]["end_to_end_ms"]["median"], 2) for b in args.backends}, flush=True)
    out["complete"] = True
    dst.write_text(json.dumps(out, indent=2))
    print("->", dst)


if __name__ == "__main__":
    main()
