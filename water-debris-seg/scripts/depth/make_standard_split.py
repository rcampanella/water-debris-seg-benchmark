"""D01 step 3, main split (depth README D6, 2026-11-13): standard train / test split.
  Pohang (by sequence, Eigen-style by drive): train = pohang00, 02, 03 (every stereo frame, 10 Hz);
      day test = pohang04 (2 Hz keyframes, frame index % 5 == 0); night test = pohang01, 05 (2 Hz keyframes).
  MULTIAQUA (official split, paper Sec. IV-D1): official train -> train; official val -> day test;
      official test (lj4) -> night test.
Frames with no ground truth (no LiDAR file, undecodable image) are excluded; the D10 frame filter (>= 50 GT points
within 20 m) is applied where ground-truth stats exist (frames whose stats are not built yet are counted as pending).
Outputs: OUT/manifest_standard_<source>.json (frame, sequence, split, d10) and OUT/standard_split_summary.json.
Usage: python make_standard_split.py --out OUT --ts-dir <pohang0X_stereo_timestamp.txt dir> --mq-split-dir <dir with
       multiaqua_split_{train,val,test}.txt> --gt <ground-truth root with <source>/stats/>
"""
import argparse
import collections
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

POHANG = {"pohang00": "train", "pohang02": "train", "pohang03": "train", "pohang04": "test_day",
          "pohang01": "test_night", "pohang05": "test_night"}
MQ = {"train": "train", "val": "test_day", "test": "test_night"}


def load_stats(path):
    if not os.path.exists(path): return None
    with open(path) as fh: r = json.load(fh)
    v = (r.get("versions") or {}).get("raw")
    return {"has_gt": bool(v), "d10": bool(v and v["pass_min_points"]), "errors": r.get("errors", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--ts-dir", required=True)
    ap.add_argument("--mq-split-dir", required=True); ap.add_argument("--gt", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    recs = {"pohang": [], "multiaqua": []}
    for seq, split in POHANG.items():
        fi = np.loadtxt(f"{a.ts_dir}/{seq}_stereo_timestamp.txt")[:, 1].astype(int)
        for k in fi:
            if split != "train" and k % 5: continue          # test: 2 Hz keyframes
            recs["pohang"].append({"sequence": seq, "frame": f"{k:06d}", "split": split,
                                   "stats": f"{a.gt}/{seq}/stats/{k:06d}.json"})
    for off, split in MQ.items():
        for fr in open(f"{a.mq_split_dir}/multiaqua_split_{off}.txt").read().split():
            recs["multiaqua"].append({"sequence": "_".join(fr.split("_")[:2]), "frame": fr, "split": split,
                                      "official_split": off, "stats": f"{a.gt}/multiaqua/stats/{fr}.json"})
    summary = {"rule": __doc__.split("\n")[1:7], "sources": {}}
    for src, rs in recs.items():
        with ThreadPoolExecutor(32) as ex: st = list(ex.map(lambda r: load_stats(r["stats"]), rs, chunksize=64))
        cnt = collections.defaultdict(collections.Counter)
        out = []
        for r, s in zip(rs, st):
            key = r["split"]
            if s is None: cnt[key]["pending_gt"] += 1; r["d10"] = None
            elif not s["has_gt"]: cnt[key]["no_gt"] += 1; r["d10"] = False; r["errors"] = s["errors"]
            else: cnt[key]["frames"] += 1; cnt[key]["d10"] += s["d10"]; r["d10"] = s["d10"]
            cnt[key]["listed"] += 1
            out.append({k: v for k, v in r.items() if k != "stats"})
        by_seq = collections.defaultdict(collections.Counter)
        for r in out: by_seq[f"{r['sequence']}:{r['split']}"]["d10" if r["d10"] else ("pending" if r["d10"] is None else "fail_or_no_gt")] += 1
        summary["sources"][src] = {"counts": {k: dict(v) for k, v in cnt.items()},
                                   "by_sequence": {k: dict(v) for k, v in sorted(by_seq.items())}}
        json.dump(out, open(f"{a.out}/manifest_standard_{src}.json", "w"))
        for k, v in cnt.items(): print(src, k, dict(v))
    json.dump(summary, open(f"{a.out}/standard_split_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
