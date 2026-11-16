#!/usr/bin/env python3
"""D01 latency (as ablation 11, bench_yolo26_latency.py): every D01 fine-tuned YOLO26-depth best.pt plus the five
released YOLO26-depth weights, timed one after another in one exclusive job on one GPU.

Backends: pytorch (half=True) and tensorrt (Ultralytics export, FP16 engine, batch 1, imgsz 640, built in a scratch dir).
End-to-end = Ultralytics Results.speed preprocess + inference + postprocess per image (image decoding excluded; the
depth postprocess includes resizing the depth map back to the native image), batch 1, imgsz 640, on the first --images
images of the source's test_day split after --warmup images; median / mean / p90.
Fine-tuned models: the newest d01_<source>_<model>_<job>.json per (source, model) in sources/finetune/ with
train_args.epochs == 50 and fraction == 1.0 (excludes smoke runs). Released models are timed on Pohang test_day images
(same architecture as the fine-tuned ones; a timing reference only).
Writes eval_outputs/water_debris_depth/ablations/D01_rgb/sources/latency/d01_latency_<job>.json.
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

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "eval_outputs/water_debris_depth/ablations/D01_rgb/sources"
DATA = REPO / "data/water_debris/depth_d01"
CKPT = REPO / "data/water_debris/checkpoints/yolo/depth"
PAT = re.compile(r"d01_(pohang|multiaqua)_(yolo26[nsmlx]-depth)_(\d+)\.json$")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def time_model(weights: str, images: list[Path], warmup: int, half: bool) -> dict:
    from ultralytics import YOLO
    model = YOLO(weights, task="depth")
    tot, inf = [], []
    for i, p in enumerate(images):
        r = model.predict(str(p), imgsz=640, half=half, device=0, batch=1, rect=False, verbose=False)[0]
        if i >= warmup:
            tot.append(r.speed["preprocess"] + r.speed["inference"] + r.speed["postprocess"]); inf.append(r.speed["inference"])
    st = lambda a: {"median": float(np.median(a)), "mean": float(np.mean(a)), "p90": float(np.percentile(a, 90))}
    return {"end_to_end_ms": st(tot), "inference_ms": st(inf), "n_timed": len(tot)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=220); ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--backends", nargs="+", default=["pytorch", "tensorrt"])
    ap.add_argument("--only", nargs="+", help="keys to time (smoke), e.g. pohang/yolo26n-depth released/yolo26n-depth")
    a = ap.parse_args()
    import torch
    import ultralytics
    latest = {}
    for p in sorted((SRC / "finetune").glob("d01_*.json")):
        m = PAT.match(p.name)
        if not m: continue
        r = json.loads(p.read_text()); ta = r.get("train_args", {})
        if ta.get("epochs") != 50 or float(ta.get("fraction", 1.0)) != 1.0: continue
        k = (m[1], m[2])
        if k not in latest or int(m[3]) > int(PAT.match(latest[k].name)[3]): latest[k] = p
    jobs = [(f"{s}/{mdl}", Path(json.loads(p.read_text())["best"]["weights"]), s, str(p.relative_to(REPO)))
            for (s, mdl), p in sorted(latest.items())]
    jobs += [(f"released/yolo26{c}-depth", CKPT / f"yolo26{c}-depth.pt", "pohang", None) for c in "nsmlx"]
    job = os.environ.get("SLURM_JOB_ID", "local")
    (SRC / "latency").mkdir(parents=True, exist_ok=True)
    dst = SRC / "latency" / f"d01_latency_{job}.json"
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "ultralytics": ultralytics.__version__,
           "setting": f"batch 1, imgsz 640 letterbox, first {a.images} test_day images of the source, {a.warmup} warm-up; "
                      "Ultralytics Results.speed preprocess + inference + postprocess", "models": {}}
    try:
        import tensorrt
        out["tensorrt"] = tensorrt.__version__
    except ImportError:
        a.backends = [b for b in a.backends if b != "tensorrt"]
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        for key, w, src, rj in jobs:
            if a.only and key not in a.only: continue
            images = sorted((DATA / src / "images" / "test_day").glob("*.jpg"))[: a.images]
            entry = {"run_json": rj, "weights": str(w), "weights_sha256": sha(w), "timing_images": f"{src}/test_day"}
            for b in a.backends:
                if b == "pytorch":
                    entry[b] = time_model(str(w), images, a.warmup, half=True)
                else:
                    wc = Path(tmp) / (key.replace("/", "_") + ".pt"); shutil.copy(w, wc)
                    from ultralytics import YOLO
                    eng = YOLO(str(wc)).export(format="engine", half=True, imgsz=640, batch=1, device=0, verbose=False)
                    entry[b] = {**time_model(eng, images, a.warmup, half=True), "engine_sha256": sha(Path(eng))}
            out["models"][key] = entry
            dst.write_text(json.dumps(out, indent=2))   # after every model: a timeout keeps what was timed
            print(key, {b: round(entry[b]["end_to_end_ms"]["median"], 2) for b in a.backends}, flush=True)
    out["complete"] = True
    dst.write_text(json.dumps(out, indent=2))
    print("->", dst)


if __name__ == "__main__":
    main()
