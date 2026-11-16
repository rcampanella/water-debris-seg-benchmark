"""D01 scorer (depth README D12, D13): score one YOLO26-depth model on one test split of one D01 dataset.

Prediction (D12): Ultralytics predict path (square 640 letterbox, rect False), depth resized back to the native image;
  zero-shot = released weights with their shipped calibration; fine-tuned = the run's weights, metric scale refitted
  (D12 point 3) with Ultralytics' scale-only formula b = mean(log gt - log pred) pooled over all ground-truth pixels
  <= max depth of the --refit-split (the day test split, D4); Ultralytics' shipped calibration is scored as well.
Relative table (D13): pinned MDEC evaluator (toshas/mdec_benchmark @ 4037628 + one-line patch, third_party/):
  MonoDepthEvaluator._align_least_squares (per image, 5-95 % trimmed fit) then clip [1e-3, max], metrics_benchmark
  (MAE m, RMSE m, AbsRel %), delta1 (metrics_eigen definition) and metrics_pointcloud (F-Score at 10 cm; nearest
  neighbours by exact PyTorch search instead of the compiled Chamfer kernel, validated in Q11).
Absolute table (D13): metrics_benchmark (KITTI devkit formulas, checked equal to the devkit in Q11) on the
  unaligned prediction, no capping: SILog (LogSI, log(m)*100), absErrorRel %, sqErrorRel %, iRMSE 1/km, MAE, RMSE.
Every metric overall (gt <= max depth) and per band 0-5 / 5-10 / 10-15 / 15-20 m (alignment always from the whole
image); per-image values in <out>.jsonl, means + bootstrap 95 % CI over images in <out>.json.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
import types

import cv2
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MDEC = os.path.join(REPO, "third_party", "mdec_benchmark")
BANDS = [(0, 5), (5, 10), (10, 15), (15, 20)]


def load_mdec():
    sys.path.insert(0, os.path.expanduser("~/envs/mdec-deps")); sys.path.insert(0, MDEC)
    six = types.ModuleType("torch._six"); six.string_classes = (str,); sys.modules["torch._six"] = six   # removed in torch 2

    class NN:   # exact nearest neighbours (squared distances), chunked; replaces the compiled CUDA Chamfer kernel
        def __call__(self, a, b):
            def mins(x, y):   # exact, chunked over both point sets (frames can hold >100k points; 40 GB GPU)
                out = []
                yd = y.double()
                for i in range(0, x.shape[1], 2048):
                    xi = x[:, i:i + 2048].double(); best = None
                    for j in range(0, yd.shape[1], 32768):
                        m = torch.cdist(xi, yd[:, j:j + 32768]).min(2).values
                        best = m if best is None else torch.minimum(best, m)
                    out.append(best ** 2)
                return torch.cat(out, 1).float()
            return mins(a, b), mins(b, a)
    cd = types.ModuleType("src.external_libs.chamfer_distance.chamfer_distance"); cd.ChamferDistance = NN
    cd.__all__ = ["ChamferDistance"]; sys.modules["src.external_libs.chamfer_distance.chamfer_distance"] = cd
    import src.core.evaluator as ev
    import src.core.metrics as mt
    mt.ChamferDistance = NN
    return ev, mt


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def predict_sparse(model, images, depth_dir, max_depth, device):
    """Yields (name, pred_at_gt_pixels, gt_values, (v, u), shape) for each image; pred is the model output as shipped."""
    def one_by_one():   # one path per call: a list source makes Ultralytics build a multi-worker loader (host / GPU OOM on 1,215 images)
        for im in images:
            yield model.predict(im, imgsz=640, verbose=False, device=device, rect=False)[0]
    for r in one_by_one():
        name = os.path.splitext(os.path.basename(r.path))[0]
        pred = r.depth.data.float().cpu().numpy()
        gt = cv2.imread(os.path.join(depth_dir, name + ".png"), cv2.IMREAD_UNCHANGED).astype(np.float64) / 256.0
        assert pred.shape == gt.shape, (name, pred.shape, gt.shape)
        v, u = np.nonzero((gt > 1e-3) & (gt <= max_depth))
        yield name, pred[v, u].astype(np.float64), gt[v, u], (v, u), gt.shape


def score_image(ev, mt, p, g, vu, shape, K, max_depth):
    out = {"n": int(len(g))}
    H, W = shape
    pred = np.zeros(shape, np.float32); tgt = np.zeros(shape, np.float32); mask = np.zeros(shape, bool)
    pred[vu] = p; tgt[vu] = g; mask[vu] = True
    al = ev.MonoDepthEvaluator._align_least_squares(pred.copy(), tgt, mask).clip(1e-3, max_depth)   # D13 relative
    pa = al[vu].astype(np.float64)
    def fams(pr_rel, pr_abs, gg, sub):
        if sub.sum() == 0: return None
        x, y = pr_rel[sub], gg[sub]
        rb = mt.metrics_benchmark(x, y); de = mt.metrics_eigen(x, y)
        rel = {"MAE": rb["MAE"], "RMSE": rb["RMSE"], "AbsRel": rb["AbsRel"], "delta1": de["$\\delta < 1.25$"]}
        m2 = np.zeros(shape, bool); m2[vu[0][sub], vu[1][sub]] = True
        pc = mt.metrics_pointcloud(al, tgt, m2, K)
        rel["F"] = pc["F-Score"]; rel["F20"] = pc["F-Score-20"]
        ab = mt.metrics_benchmark(pr_abs[sub], y)
        absm = {"SILog": ab["LogSI"], "absErrorRel": ab["AbsRel"], "sqErrorRel": ab["SqRel"], "iRMSE": ab["InvRMSE"],
                "MAE": ab["MAE"], "RMSE": ab["RMSE"]}
        return {"n": int(sub.sum()), "relative": rel, "absolute": absm}
    allm = np.ones(len(g), bool)
    out["all"] = fams(pa, p, g, allm)
    out["bands"] = {f"{a}-{b}": fams(pa, p, g, (g > a) & (g <= b)) for a, b in BANDS}
    return out


def aggregate(per_image, key_path, rng):
    vals = []
    for r in per_image:
        x = r
        for k in key_path:
            x = None if x is None else x.get(k)
        if x is not None: vals.append(x)
    if not vals: return {"n_images": 0}
    v = np.array(vals); bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(1000)]
    return {"mean": float(v.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n_images": len(v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root (images/<split>, depth/<split>)")
    ap.add_argument("--split", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--arm", choices=["zero-shot", "fine-tuned"], required=True)
    ap.add_argument("--K", required=True, help="fx,fy,cx,cy of the (undistorted) images")
    ap.add_argument("--refit-split", default="test_day", help="split used for the D12 metric-scale refit (fine-tuned)")
    ap.add_argument("--max-depth", type=float, default=20.0); ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True, help="output prefix (.json, .jsonl)")
    a = ap.parse_args()
    ev, mt = load_mdec()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    fx, fy, cx, cy = map(float, a.K.split(","))
    K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    def run(split):
        ims = sorted(glob.glob(f"{a.data}/images/{split}/*.jpg"))
        return ims, list(predict_sparse(model, ims, f"{a.data}/depth/{split}", a.max_depth, a.device))
    t0 = time.time()
    ims, preds = run(a.split)
    variants = {"shipped": 1.0}
    refit = None
    if a.arm == "fine-tuned":   # D12 point 3: refit on the day test split (= D4 split), pooled pixels
        rp = preds if a.split == a.refit_split else run(a.refit_split)[1]
        lg = np.concatenate([np.log(g) for _, p, g, _, _ in rp]); lp = np.concatenate([np.log(np.maximum(p, 1e-6)) for _, p, g, _, _ in rp])
        b = float(np.mean(lg - lp)); refit = {"b": b, "scale": float(np.exp(b)), "split": a.refit_split, "n_pixels": int(len(lg))}
        variants["refit"] = float(np.exp(b))
    rng = np.random.default_rng(0)
    summary = {"weights": a.weights, "weights_sha256": sha256(a.weights), "arm": a.arm, "data": a.data, "split": a.split,
               "n_images": len(ims), "max_depth": a.max_depth, "K": a.K, "refit": refit, "variants": {}}
    with open(a.out + ".jsonl", "w") as fj:
        for var, s in variants.items():
            per = []
            for name, p, g, vu, shape in preds:
                r = score_image(ev, mt, p * s, g, vu, shape, K, a.max_depth); r["name"] = name; r["variant"] = var
                per.append(r); fj.write(json.dumps(r) + "\n")
            agg = {"all": {}, "bands": {}}
            for fam, keys in (("relative", ["F", "F20", "MAE", "RMSE", "AbsRel", "delta1"]),
                              ("absolute", ["SILog", "absErrorRel", "sqErrorRel", "iRMSE", "MAE", "RMSE"])):
                agg["all"][fam] = {k: aggregate(per, ["all", fam, k], rng) for k in keys}
                for bn in [f"{x}-{y}" for x, y in BANDS]:
                    agg["bands"].setdefault(bn, {})[fam] = {k: aggregate(per, ["bands", bn, fam, k], rng) for k in keys}
            agg["gt_pixels"] = int(sum(r["n"] for r in per))
            summary["variants"][var] = agg
    summary["seconds"] = time.time() - t0
    json.dump(summary, open(a.out + ".json", "w"), indent=1)
    for var, agg in summary["variants"].items():
        r, b = agg["all"]["relative"], agg["all"]["absolute"]
        print(f"{os.path.basename(a.weights)} {a.split} [{var}] F {r['F']['mean']:.2f}  AbsRel(rel) {r['AbsRel']['mean']:.2f}%  "
              f"SILog {b['SILog']['mean']:.2f}  absErrRel {b['absErrorRel']['mean']:.2f}%  ({len(ims)} images)", flush=True)


if __name__ == "__main__":
    main()
