"""D01 step 4 prep: Ultralytics depth datasets from the standard split (README D6) and the unadapted ground truth.

Layout (Ultralytics depth task, as its depth-kitti.yaml): <root>/<source>/images/<split>/<name>.jpg and
<root>/<source>/depth/<split>/<name>.png (uint16, depth_scale 256 = KITTI convention, 0 = no ground truth).
  * frames: manifest_standard_<source>.json, only frames passing D10 (d10 == True);
  * splits: train, test_day (= Ultralytics 'val': checkpoint / calibration on the test split, D4), test_night;
  * Pohang images: left camera, undistorted with the same camera matrix (cv2 initUndistortRectifyMap(K, D, None, K)),
    the geometry the ground truth was projected into (builder --gt-frame undistorted); MULTIAQUA ZED images have
    zero distortion and are used as released;
  * native resolution, JPEG quality 95; ground-truth PNG copied unchanged (Ultralytics' loader does the resizing,
    as for its KITTI dataset; standard pipeline, README D12);
  * name = <sequence>_<frame>.
Also writes <root>/<source>.yaml and <root>/<source>_files.json (per file: split, source frame, SHA-256).
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import zipfile
from multiprocessing import get_context

import cv2
import numpy as np

SPLITS = {"train": "train", "test_day": "test_day", "test_night": "test_night"}
_G = {}


def _init(src, raw_root, mq_zip):
    _G["src"] = src; _G["z"] = {}; _G["raw_root"] = raw_root; _G["mq_zip"] = mq_zip; _G["maps"] = {}


def _zip(path):
    if path not in _G["z"]: _G["z"][path] = zipfile.ZipFile(path)
    return _G["z"][path]


def _undistort_map(seq):
    if seq not in _G["maps"]:
        cz = zipfile.ZipFile(f"{_G['raw_root']}/{seq}/calibration.zip")
        name = [n for n in cz.namelist() if n.endswith("intrinsics.json")][0]
        I = json.loads(cz.read(name))["stereo_left"]
        K = np.array([[I["focal_length"], 0, I["cc_x"]], [0, I["focal_length"], I["cc_y"]], [0, 0, 1.0]])
        D = np.array(I["distortion_coefficients"], float); wh = (I["image_width"], I["image_height"])
        _G["maps"][seq] = cv2.initUndistortRectifyMap(K, D, None, K, wh, cv2.CV_32FC1)
    return _G["maps"][seq]


def work(job):
    rec, out_root, gt_root = job
    src = _G["src"]; seq, fr, split = rec["sequence"], rec["frame"], rec["split"]
    name = f"{seq}_{fr}" if src == "pohang" else fr
    if src == "pohang":
        b = _zip(f"{_G['raw_root']}/{seq}/stereo.zip").read(f"stereo/left_images/{fr}.png")
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.remap(img, *_undistort_map(seq), cv2.INTER_LINEAR)
        gt = f"{gt_root}/{seq}/raw/depth/{fr}.png"
    else:
        b = _zip(_G["mq_zip"]).read(f"data/zed/{fr}.png")
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        gt = f"{gt_root}/multiaqua/raw/depth/{fr}.png"
    d = cv2.imread(gt, cv2.IMREAD_UNCHANGED)
    assert d is not None and d.dtype == np.uint16 and d.shape[:2] == img.shape[:2], (gt, None if d is None else d.shape, img.shape)
    ip = f"{out_root}/images/{SPLITS[split]}/{name}.jpg"; dp = f"{out_root}/depth/{SPLITS[split]}/{name}.png"
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95]); assert ok
    open(ip, "wb").write(enc.tobytes()); shutil.copyfile(gt, dp)
    return {"name": name, "split": SPLITS[split], "sequence": seq, "frame": fr, "image_sha256": hashlib.sha256(enc.tobytes()).hexdigest(),
            "depth_sha256": hashlib.sha256(open(dp, "rb").read()).hexdigest(), "gt_points": int((d > 0).sum()), "shape": list(img.shape[:2])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["pohang", "multiaqua"], required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--gt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--raw-root", default="/scratch-shared/rcampanella/water-debris-seg/depth_raw")
    ap.add_argument("--mq-zip", default="/scratch-shared/rcampanella/water-debris-seg/depth_raw/multiaqua/MULTIAQUA_full.zip")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    root = f"{a.out}/{a.source}"
    for sp in SPLITS.values():
        os.makedirs(f"{root}/images/{sp}", exist_ok=True); os.makedirs(f"{root}/depth/{sp}", exist_ok=True)
    recs = [r for r in json.load(open(a.manifest)) if r.get("d10")]
    print(f"[{a.source}] {len(recs)} frames passing D10", flush=True)
    with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.source, a.raw_root, a.mq_zip)) as pool:
        files = []
        for i, f in enumerate(pool.imap_unordered(work, [(r, root, a.gt) for r in recs], chunksize=8)):
            files.append(f)
            if (i + 1) % 1000 == 0: print(f"[{a.source}] {i + 1}/{len(recs)}", flush=True)
    files.sort(key=lambda x: (x["split"], x["name"]))
    json.dump(files, open(f"{a.out}/{a.source}_files.json", "w"))
    yml = (f"# D01 depth dataset ({a.source}), built by prepare_depth_dataset.py from {os.path.basename(a.manifest)}\n"
           f"# val = day test split (checkpoint / calibration on test, README D4); test_night scored by our scorer only\n"
           f"path: {root}\ntrain: images/train\nval: images/test_day\ntest: images/test_night\n"
           f"max_depth: 20\ndepth_scale: 256\nnc: 1\nnames:\n  0: depth\nchannels: 3\n")
    open(f"{a.out}/{a.source}.yaml", "w").write(yml)
    from collections import Counter
    print(dict(Counter(f["split"] for f in files)), flush=True)


if __name__ == "__main__":
    main()
