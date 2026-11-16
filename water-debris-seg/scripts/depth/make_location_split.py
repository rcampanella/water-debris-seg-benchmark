"""D01 step 3: GPS location-disjoint train / test split (depth README D6) for Pohang Canal and MULTIAQUA.

Route environments (Pohang loop; MULTIAQUA Ljubljanica river lj1/lj3/lj4, Lake Bled bl1, lj2 area):
  * a reference track (one day sequence, time-ordered, smoothed) is parametrised by arc length s;
  * every frame gets the s of its nearest reference point (and its lateral distance);
  * K test segments, each FRAC/K of the route length, centred at evenly spaced s with a seeded random phase;
  * day frames inside a test segment -> test; day frames whose camera is within GAP m of any test-segment
    reference point or any test frame -> gap (dropped); other day frames -> train;
  * night sequences -> test only, 'night_unseen' (inside a test segment) or 'night_seen' (elsewhere).
Site environments (MULTIAQUA adr1: sub-sequences at separate coastal sites, km apart): sub-sequences are grouped into
  sites (centroids > SITE_M apart); whole sites are held out, choosing the site set whose frame share is closest to
  FRAC (ties: seed). Sites are far apart, so no gap is needed (checked: minimum train-test distance is reported).
Frame sampling (README): Pohang train = every stereo frame (10 Hz), Pohang test / night = 2 Hz keyframes (index % 5);
  MULTIAQUA = every frame. Frames not sampled get split 'unsampled'.
Outputs (OUT dir): manifest_<source>.json (one record per frame), split_summary.json (counts, with and without the D10
  frame filter where ground-truth stats exist), split_map_<env>.png (route map: train / gap / test / night).
"""
import argparse
import collections
import glob
import json
import os
import zipfile

import numpy as np
from scipy.spatial import cKDTree

POHANG_NIGHT = {"pohang01", "pohang05"}
MQ_NIGHT_PREFIX = "lj4"


def lla_to_local(lat, lon, lat0, lon0):
    y = (np.atleast_1d(lat) - lat0) * 111320.0
    x = (np.atleast_1d(lon) - lon0) * 111320.0 * np.cos(np.radians(np.atleast_1d(lat)))   # local scale per point
    return np.stack([x, y], 1)


def smooth_track(xy, win):
    if win <= 1 or len(xy) < win: return xy
    k = np.ones(win) / win
    pad = win // 2
    xp = np.pad(xy, ((pad, pad), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, i], k, mode="valid") for i in range(2)], 1)[: len(xy)]


def resample(xy, step=1.0):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    s = np.arange(0, d[-1], step)
    return np.stack([np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])], 1), s


def route_split(env, frames, ref_xy, frac, k, gap, seed, loop):
    """frames: list of dicts with xy, night. Adds s, lat_dist, split."""
    ref, s_ref = resample(ref_xy)
    L = s_ref[-1]
    tree = cKDTree(ref)
    xy = np.array([f["xy"] for f in frames])
    dist, idx = tree.query(xy)
    s = s_ref[idx]
    rng = np.random.default_rng(seed)
    seg_len = frac * L / k
    slot = L / k
    if loop:    # loop: k evenly spaced centres, one random phase
        centres = (rng.uniform(0, slot) + np.arange(k) * slot) % L
    else:       # open route: one segment per slot, centre jittered inside its slot
        centres = (np.arange(k) + 0.5) * slot + rng.uniform(-0.5, 0.5, k) * (slot - seg_len)
    def in_test(sv):
        dd = np.abs(sv[:, None] - centres[None, :])
        if loop: dd = np.minimum(dd, L - dd)
        return (dd <= seg_len / 2).any(1)
    test_ref = ref[in_test(s_ref)]
    t_mask = in_test(s)
    night = np.array([f["night"] for f in frames])
    test_pts = np.vstack([test_ref, xy[t_mask & ~night]]) if (t_mask & ~night).any() else test_ref
    near_test = cKDTree(test_pts).query(xy)[0] <= gap
    for f, si, di, tm, nt, nm in zip(frames, s, dist, t_mask, near_test, night):
        f["s"] = float(si); f["lat_dist"] = float(di)
        if nm: f["split"] = "night_unseen" if tm else "night_seen"
        elif tm: f["split"] = "test"
        elif nt: f["split"] = "gap"
        else: f["split"] = "train"
    return {"env": env, "type": "route", "loop": loop, "route_length_m": float(L), "k_segments": k,
            "segment_length_m": float(seg_len), "segment_centres_m": [float(c) for c in sorted(centres)],
            "gap_m": gap, "seed": seed, "ref": ref, "test_ref": test_ref}


def grid_split(env, frames, frac, cell_m, gap, seed, weight=None, n_regions=1):
    """Spatial blocks: square cells of cell_m; whole cells drawn in seeded random order until >= frac of the day
    frames are test; day frames within gap m of a test frame -> gap; night frames tagged by their cell."""
    xy = np.array([f["xy"] for f in frames]); night = np.array([f["night"] for f in frames])
    cell = [tuple(c) for c in np.floor(xy / cell_m).astype(int)]
    # weight: frame key -> 0/1 (passes the D10 frame filter); cells are drawn until frac of the *usable* day frames is test
    w = [1.0 if weight is None else float(weight.get(f["wkey"], 0)) for f in frames]
    day_cells = collections.Counter()
    for c, n, wi in zip(cell, night, w):
        if not n: day_cells[c] += wi
    day_cells = collections.Counter({c: v for c, v in day_cells.items() if v > 0} or day_cells)
    n_day = sum(day_cells.values())
    rng = np.random.default_rng(seed)
    # regions: k-means (seeded) on the centres of cells with day frames, so the test set covers every part of the area
    cells_sorted = sorted(day_cells)
    cxy = (np.array(cells_sorted, float) + 0.5) * cell_m
    k = max(1, min(n_regions, len(cells_sorted)))
    cen = cxy[rng.choice(len(cxy), k, replace=False)]
    for _ in range(50):
        lab = np.argmin(((cxy[:, None] - cen[None]) ** 2).sum(-1), 1)
        cen = np.array([cxy[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(k)])
    region = {c: int(l) for c, l in zip(cells_sorted, lab)}
    test_cells, got = set(), 0; region_report = []
    for j in range(k):   # per region: draw cells until frac of the region's (usable) day frames is test
        rc = [c for c in cells_sorted if region[c] == j]
        tot = sum(day_cells[c] for c in rc); g = 0
        target = frac * tot
        for i in rng.permutation(len(rc)):     # skip cells that overshoot the target by > 25% (dense canal cells)
            if g >= target: break
            wc = day_cells[rc[i]]
            if g + wc <= 1.25 * target:
                test_cells.add(rc[i]); g += wc
        region_report.append({"region": j, "cells": len(rc), "usable_frames": tot, "test_usable_frames": g,
                              "test_share": (g / tot) if tot else None})
        got += g
    t_mask = np.array([c in test_cells for c in cell])
    test_pts = xy[t_mask & ~night]
    near = cKDTree(test_pts).query(xy)[0] <= gap if len(test_pts) else np.zeros(len(xy), bool)
    for f, c, tm, nt, nm in zip(frames, cell, t_mask, near, night):
        f["cell"] = [int(c[0]), int(c[1])]; f["s"] = None; f["lat_dist"] = None
        if nm: f["split"] = "night_unseen" if tm else "night_seen"
        elif tm: f["split"] = "test"
        elif nt: f["split"] = "gap"
        else: f["split"] = "train"
    return {"env": env, "type": "grid", "cell_m": cell_m, "gap_m": gap, "seed": seed, "n_day_cells": len(day_cells),
            "test_cells": sorted([list(map(int, c)) for c in test_cells]), "weighted_by": "D10 pass" if weight is not None else "frames",
            "test_share_before_gap": got / max(n_day, 1), "n_regions": k, "regions": region_report}


def site_split(env, frames, frac, site_m, seed, weight=None):
    seqs = sorted({f["sequence"] for f in frames})
    cen = {q: np.mean([f["xy"] for f in frames if f["sequence"] == q], 0) for q in seqs}
    # single-linkage grouping of sub-sequences into sites
    site = {q: i for i, q in enumerate(seqs)}
    for a in seqs:
        for b in seqs:
            if np.hypot(*(cen[a] - cen[b])) < site_m:
                sa, sb = site[a], site[b]
                for q in seqs:
                    if site[q] == sb: site[q] = sa
    sites = sorted(set(site.values()))
    wf = lambda f: 1.0 if weight is None else float(weight.get(f["wkey"], 0))
    n_site = {si: sum(wf(f) for f in frames if site[f["sequence"]] == si and not f["night"]) for si in sites}
    N = sum(n_site.values()) or 1
    best = None
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sites))
    from itertools import combinations
    for r in range(1, len(sites)):
        for comb in combinations([sites[i] for i in order], r):
            share = sum(n_site[c] for c in comb) / N
            if best is None or abs(share - frac) < abs(best[1] - frac): best = (comb, share)
    test_sites = set(best[0])
    for f in frames:
        f["site"] = int(site[f["sequence"]]); f["s"] = None; f["lat_dist"] = None
        f["split"] = ("night_unseen" if f["night"] else "test") if site[f["sequence"]] in test_sites else ("night_seen" if f["night"] else "train")
    tr = np.array([f["xy"] for f in frames if f["split"] == "train"]); te = np.array([f["xy"] for f in frames if f["split"] == "test"])
    min_d = float(cKDTree(te).query(tr)[0].min()) if len(tr) and len(te) else None
    return {"env": env, "type": "sites", "site_m": site_m, "sites": {str(si): sorted(q for q in seqs if site[q] == si) for si in sites},
            "frames_per_site": {str(si): n_site[si] for si in sites}, "test_sites": sorted(int(x) for x in test_sites),
            "test_frame_share": best[1], "min_train_test_distance_m": min_d, "seed": seed}


def pohang_frames(raw_root, nav_only, ts_dir):
    out = []
    for i in range(6):
        seq = f"pohang0{i}"
        nz = f"{raw_root}/{seq}/navigation.zip"
        if not os.path.exists(nz): nz = f"{nav_only}/{seq}/navigation.zip"
        b = np.loadtxt(zipfile.ZipFile(nz).open("navigation/baseline.txt"))
        tb, north, east = b[:, 0], b[:, 5], b[:, 6]
        ts = np.loadtxt(f"{ts_dir}/{seq}_stereo_timestamp.txt")
        tc, fi = ts[:, 0], ts[:, 1].astype(int)
        e = np.interp(tc, tb, east); n = np.interp(tc, tb, north)
        outside = (tc < tb[0]) | (tc > tb[-1])
        for t, k, x, y, o in zip(tc, fi, e, n, outside):
            out.append({"source": "pohang", "sequence": seq, "frame": f"{k:06d}", "t": float(t), "xy": [float(x), float(y)],
                        "night": seq in POHANG_NIGHT, "keyframe": bool(k % 5 == 0), "pose_extrapolated": bool(o)})
    return out


def mq_frames(zip_path):
    z = zipfile.ZipFile(zip_path); out = []
    for n in z.namelist():
        if n.startswith("data/GPSIMU/") and n.endswith(".json"):
            g = json.loads(z.read(n)); fr = n.split("/")[-1][:-5]
            out.append({"source": "multiaqua", "sequence": "_".join(fr.split("_")[:2]), "frame": fr,
                        "t": g["timestamp1"] + g["timestamp2"] * 1e-6, "lat": g["latitude"], "lon": g["longitude"],
                        "night": fr.startswith(MQ_NIGHT_PREFIX), "keyframe": True})
    zed = {n.split("/")[-1][:-4] for n in z.namelist() if n.startswith("data/zed/") and n.endswith(".png")}
    missing_gps = sorted(zed - {f["frame"] for f in out})
    return out, missing_gps


def ref_track(frames, seq, win):
    f = sorted([x for x in frames if x["sequence"] == seq], key=lambda r: r["t"])
    return smooth_track(np.array([x["xy"] for x in f]), win)


def gt_pass(stats_dirs):
    """frame key -> pass_min_points (raw), where ground-truth stats exist."""
    from concurrent.futures import ThreadPoolExecutor
    files = [f for d in stats_dirs for f in glob.glob(d + "/*.json")]
    def load(p):
        with open(p) as fh: r = json.load(fh)
        v = (r.get("versions") or {}).get("raw")
        seq = r.get("sequence") or ""
        return (seq if seq.startswith("pohang") else "multiaqua", r["frame"]), (bool(v["pass_min_points"]) if v else False)
    with ThreadPoolExecutor(32) as ex: return dict(ex.map(load, files, chunksize=64))


def plot(env_info, frames, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    col = {"train": "#1f77b4", "gap": "#bbbbbb", "test": "#d62728", "night_unseen": "#9467bd", "night_seen": "#c5b0d5", "unsampled": None}
    fig, ax = plt.subplots(figsize=(9, 9))
    if env_info.get("ref") is not None: ax.plot(*env_info["ref"].T, "-", color="#999999", lw=0.5, label="reference track")
    for sp in ("train", "gap", "night_seen", "night_unseen", "test"):
        p = np.array([f["xy"] for f in frames if f["split"] == sp])
        if len(p): ax.plot(p[:, 0], p[:, 1], ".", ms=2 if sp in ("train", "gap") else 3, color=col[sp], label=f"{sp} ({len(p)})")
    ax.set_aspect("equal"); ax.legend(markerscale=4, fontsize=8); ax.set_title(title); ax.set_xlabel("east (m)"); ax.set_ylabel("north (m)")
    plt.savefig(path, dpi=90, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pohang-raw", default="/scratch-shared/rcampanella/water-debris-seg/depth_raw")
    ap.add_argument("--pohang-nav-only", default="/scratch-shared/rcampanella/water-debris-seg/depth_raw/nav_only")
    ap.add_argument("--pohang-ts", required=True, help="dir with pohang0X_stereo_timestamp.txt")
    ap.add_argument("--mq-zip", default="/scratch-shared/rcampanella/water-debris-seg/depth_raw/multiaqua/MULTIAQUA_full.zip")
    ap.add_argument("--gt-stats", nargs="*", default=[], help="ground-truth stats dirs (for D10 counts)")
    ap.add_argument("--frac", type=float, default=0.20)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gap", type=float, default=50.0)
    ap.add_argument("--site-m", type=float, default=500.0)
    ap.add_argument("--cell-m", type=float, default=150.0, help="grid split cell size (m)")
    ap.add_argument("--weight", choices=["d10", "frames"], default="d10", help="draw test cells by usable (D10) frames")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    envs = []
    gp = gt_pass(a.gt_stats) if a.gt_stats else {}
    W = gp if (gp and a.weight == "d10") else None
    # ---- Pohang: one loop, local metres = UTM east / north
    P = pohang_frames(a.pohang_raw, a.pohang_nav_only, a.pohang_ts)
    for f in P: f["wkey"] = (f["sequence"], f["frame"])
    envs.append(("pohang_loop", P, route_split("pohang_loop", P, ref_track(P, "pohang00", 5), a.frac, a.k, a.gap, a.seed, loop=True)))
    # ---- MULTIAQUA
    M, missing_gps = mq_frames(a.mq_zip)
    lat0, lon0 = np.mean([f["lat"] for f in M]), np.mean([f["lon"] for f in M])
    for f in M:
        f["xy"] = lla_to_local(f["lat"], f["lon"], lat0, lon0)[0].tolist(); f["wkey"] = ("multiaqua", f["frame"])
    def sel(pref): return [f for f in M if any(f["sequence"].startswith(p) for p in pref)]
    river = sel(("lj1", "lj3", "lj4")); bled = sel(("bl1",)); lj2 = sel(("lj2",)); adr = sel(("adr1",))
    envs.append(("mq_ljubljanica", river, grid_split("mq_ljubljanica", river, a.frac, a.cell_m, a.gap, a.seed, W, n_regions=5)))
    for f in bled: f["split"] = "test"; f["s"] = None; f["lat_dist"] = None
    envs.append(("mq_bled", bled, {"env": "mq_bled", "type": "whole environment held out (too few usable frames to split)"}))
    envs.append(("mq_lj2", lj2, grid_split("mq_lj2", lj2, a.frac, a.cell_m, a.gap, a.seed, W, n_regions=3)))
    envs.append(("mq_adr_sites", adr, site_split("mq_adr_sites", adr, a.frac, a.site_m, a.seed, W)))
    # ---- frame sampling (README): Pohang test / night at 2 Hz keyframes
    for f in P:
        if f["split"] in ("test", "night_unseen", "night_seen") and not f["keyframe"]: f["split_before_sampling"] = f["split"]; f["split"] = "unsampled"
    # ---- outputs
    summary = {"params": {k: v for k, v in vars(a).items()}, "multiaqua_frames_without_gps": missing_gps, "envs": {}}
    for name, frames, info in envs:
        cnt = collections.Counter(f["split"] for f in frames)
        by_seq = collections.defaultdict(collections.Counter)
        for f in frames: by_seq[f["sequence"]][f["split"]] += 1
        d10 = collections.Counter(); have = 0
        for f in frames:
            key = f["wkey"]
            if key in gp:
                have += 1
                if gp[key]: d10[f["split"]] += 1
        info_out = {k: v for k, v in info.items() if k not in ("ref", "test_ref")}
        summary["envs"][name] = {**info_out, "counts": dict(cnt), "counts_by_sequence": {k: dict(v) for k, v in sorted(by_seq.items())},
                                 "frames_with_gt_stats": have, "counts_passing_D10": dict(d10)}
        plot(info, frames, f"{a.out}/split_map_{name}.png", f"{name}: location split (seed {a.seed})")
    for src, frames in (("pohang", P), ("multiaqua", M)):
        recs = [{k: v for k, v in f.items() if k not in ("lat", "lon")} for f in frames]
        json.dump(recs, open(f"{a.out}/manifest_{src}.json", "w"))
    json.dump(summary, open(f"{a.out}/split_summary.json", "w"), indent=1)
    for name, s in summary["envs"].items():
        print(name, s["counts"], "| D10:", s["counts_passing_D10"], f"(gt stats for {s['frames_with_gt_stats']})")


if __name__ == "__main__":
    main()
