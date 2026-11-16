"""G1 clock check: summarise a camera-time-offset sweep of build_depth_gt.py (Uhrig version, stats only).

For each run directory (one offset): pooled tau-outlier rate among SGM-valid, non-water LiDAR points <= 20 m, the
median LiDAR - SGM disparity residual per disparity bin, and a least-squares fit of the per-bin medians to
    r(d) = a + b * d^2
A camera-time offset shifts the scene along the motion (forward ~3 m/s), giving a residual that grows ~ d^2
(delta_d = d^2 * delta_z / (f B)); a calibration offset (e.g. rectified principal point) gives a constant a.
Usage: python analyze_clock_sweep.py OUT_JSON RUN_DIR [RUN_DIR ...]   (RUN_DIR contains <seq>/params.json, stats/)
"""
import glob
import json
import os
import sys

import numpy as np


def summarise(run):
    p = json.load(open(glob.glob(os.path.join(run, "*", "params.json"))[0]))
    bins = {}
    n_valid = n_out = 0
    rate_by_scan = {}
    for f in glob.glob(os.path.join(run, "*", "stats", "*.json")):
        u = json.load(open(f))["versions"].get("uhrig")
        if not u:
            continue
        for b in u["stereo_residual_by_disparity_bin"]:
            if not b["n"]:
                continue
            k = (b["d_lo"], b["d_hi"])
            e = bins.setdefault(k, {"n": 0, "sum": 0.0, "outliers": 0, "medians": []})
            e["n"] += b["n"]; e["sum"] += b["sum"]; e["outliers"] += b["outliers"]; e["medians"].append(b["median"])
        for j, v in u["by_scan_offset"].items():
            r = rate_by_scan.setdefault(j, [0, 0]); r[0] += v["tau_outliers"]; r[1] += v["sgm_valid_le_max_depth_nonwater"]
            n_out += v["tau_outliers"]; n_valid += v["sgm_valid_le_max_depth_nonwater"]
    rows = []
    for (lo, hi), e in sorted(bins.items()):
        rows.append({"d_lo": lo, "d_hi": hi, "n": e["n"], "mean_residual_px": e["sum"] / e["n"],
                     "median_of_frame_medians_px": float(np.median(e["medians"])), "tau_rate": e["outliers"] / e["n"]})
    # fit a + b d^2 on bin means (weights = n), bins with >= 200 points
    use = [r for r in rows if r["n"] >= 200]
    fit = None
    if len(use) >= 3:
        d = np.array([(r["d_lo"] + r["d_hi"]) / 2 for r in use]); y = np.array([r["mean_residual_px"] for r in use])
        w = np.sqrt([r["n"] for r in use])
        A = np.c_[np.ones_like(d), d ** 2] * w[:, None]
        a, b = np.linalg.lstsq(A, y * w, rcond=None)[0]
        fit = {"a_px": float(a), "b_px_per_px2": float(b), "bins_used": len(use)}
    return {"run": run, "cam_time_offset_s": p["cam_time_offset_s"], "n_scans": p["n_scans"],
            "tau_rate_sgm_valid_le20": n_out / max(n_valid, 1), "n_sgm_valid_le20": n_valid,
            "tau_rate_by_scan_offset": {j: v[0] / max(v[1], 1) for j, v in sorted(rate_by_scan.items(), key=lambda x: int(x[0]))},
            "bins": rows, "fit_a_plus_b_d2": fit}


if __name__ == "__main__":
    out = [summarise(r) for r in sys.argv[2:]]
    out.sort(key=lambda x: (x["n_scans"], x["cam_time_offset_s"]))
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for s in out:
        f = s["fit_a_plus_b_d2"] or {}
        near = [r for r in s["bins"] if r["d_lo"] >= 120]
        nn = sum(r["n"] for r in near)
        near_rate = sum(r["tau_rate"] * r["n"] for r in near) / nn if nn else float("nan")
        print(f"n{s['n_scans']:>2} off {s['cam_time_offset_s']:+.2f}s  tau {s['tau_rate_sgm_valid_le20']:.3f} "
              f"(n={s['n_sgm_valid_le20']})  near(d>=120px) tau {near_rate:.3f} (n={nn})  "
              f"fit a={f.get('a_px', float('nan')):+.2f}px b={f.get('b_px_per_px2', float('nan')):+.2e}")
