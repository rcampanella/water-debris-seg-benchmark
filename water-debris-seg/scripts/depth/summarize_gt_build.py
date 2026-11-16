"""Coverage summary of a full ground-truth build (unadapted): per sequence, frames passing the D10 rule, kept points
per band (0-5/5-10/10-15/15-20 m) and how many of them lie on the (tag-only) water mask.
Usage: python summarize_gt_build.py OUT_JSON STATS_DIR [STATS_DIR ...] [--version raw]"""
import collections, glob, json, sys
import numpy as np

def main():
    args = sys.argv[1:]; ver = "raw"
    if "--version" in args: i = args.index("--version"); ver = args[i + 1]; del args[i:i + 2]
    out, dirs = args[0], args[1:]
    S = collections.defaultdict(lambda: {"frames": 0, "pass": 0, "no_gt": 0, "fin": np.zeros(5), "onw": np.zeros(5)})
    errs = collections.Counter()
    from concurrent.futures import ThreadPoolExecutor
    files = [f for d in dirs for f in glob.glob(d + "/*.json")]
    def load(f):
        with open(f) as fh: return json.load(fh)
    with ThreadPoolExecutor(32) as ex: recs = list(ex.map(load, files, chunksize=64))   # GPFS small files: parallel reads
    for r in recs:
            s = S[r.get("sequence") or r["frame"].split("_")[0]]; s["frames"] += 1
            for e in r.get("errors", []): errs[e] += 1
            v = (r.get("versions") or {}).get(ver)
            if not v: s["no_gt"] += 1; continue
            s["pass"] += v["pass_min_points"]
            s["fin"] += v["stats_per_band"]["final_points"]; s["onw"] += v["stats_per_band"]["final_points_on_water_mask"]
    res = {"version": ver, "errors": dict(errs), "sequences": {}}
    for sq in sorted(S):
        s = S[sq]; f = s["fin"][:4]; w = s["onw"][:4]
        res["sequences"][sq] = {"frames": s["frames"], "frames_without_gt": s["no_gt"], "frames_pass_min_points": s["pass"],
                                "points_per_band_le20": f.astype(int).tolist(), "points_on_water_mask_per_band": w.astype(int).tolist(),
                                "share_on_water": round(float(w.sum() / max(f.sum(), 1)), 4)}
        print(f"{sq:9s} frames {s['frames']:6d} pass {s['pass']:6d} ({s['pass']/max(s['frames'],1):.0%})  pts/band {f.astype(int).tolist()}  on water {w.sum()/max(f.sum(),1):.1%}")
    json.dump(res, open(out, "w"), indent=1)
    print("errors", dict(errs))

if __name__ == "__main__":
    main()
