#!/usr/bin/env python3
"""D01 bookkeeping audit: split manifests, per-image -> summary means, pooled table values, checkpoint SHA-256.

    python water-debris-seg/scripts/depth/audit_d01_consistency.py eval_outputs/water_debris_depth/ablations/D01_rgb/sources

Checks aggregation and provenance only; metric formulas are covered by the Q11 scorer check."""
import json, glob, collections, os, re, hashlib, sys
os.chdir(sys.argv[1])
out = {}
# 1. split manifests
for src in ("pohang", "multiaqua"):
    m = json.load(open(f"gt_unadapted_v1/manifest_standard_{src}.json"))
    c = collections.Counter((r["split"], r.get("d10")) for r in m)
    seqs = collections.defaultdict(set)
    for r in m:
        if r.get("d10"): seqs[r["split"]].add(r.get("sequence", r.get("region", "?")))
    keys = collections.defaultdict(set)
    for r in m:
        if r.get("d10"): keys[r["split"]].add((r.get("sequence"), r.get("frame")))
    overlap = {f"{a}&{b}": len(keys[a] & keys[b]) for a in keys for b in keys if a < b}
    print(src, "d10 counts", {s: c[(s, True)] for s in ("train", "test_day", "test_night")},
          "sequences", {k: sorted(v)[:8] for k, v in seqs.items()}, "frame overlap", overlap)
# 2. jsonl means vs json means
metrics = [("relative", "F"), ("relative", "AbsRel"), ("relative", "delta1"), ("absolute", "SILog"), ("absolute", "absErrorRel")]
worst = 0; nfiles = 0; nimg_mismatch = []
pooled = collections.defaultdict(lambda: collections.defaultdict(list))
for jf in sorted(glob.glob("zeroshot/*.json") + glob.glob("finetune/ft_*.json")):
    if "29276450" in jf or "29276394" in jf or "29276494" in jf: continue  # smoke / superseded runs
    j = json.load(open(jf)); lf = jf + "l"
    if not os.path.exists(lf): print("NO JSONL", jf); continue
    rows = [json.loads(x) for x in open(lf)]
    variants = j.get("variants") or {"_": {"all": j.get("all")}}
    for vname, v in variants.items():
        rs = [r for r in rows if r.get("variant", "_") == vname] if vname != "_" else rows
        if not rs: rs = rows
        for grp, met in metrics:
            vals = [r["all"][grp][met] for r in rs if r["all"] and r["all"][grp].get(met) is not None]
            ref = v["all"][grp][met]["mean"]
            mine = sum(vals) / len(vals)
            worst = max(worst, abs(mine - ref) / max(abs(ref), 1e-9))
            if v["all"][grp][met]["n_images"] != len(vals): nimg_mismatch.append((jf, vname, met))
            key = re.sub(r"_test_(day|night)", "", os.path.basename(jf)); key = re.sub(r"_\d+\.json", "", key)
            pooled[(key, vname, met)][re.search(r"test_(day|night)", jf).group(1)] = vals
        nfiles += 1
print(f"json files checked: {nfiles}; worst relative diff jsonl-mean vs json-mean: {worst:.2e}; n_images mismatches: {len(nimg_mismatch)}")
ZSV = list((json.load(open("zeroshot/zs_pohang_yolo26n_test_day_29276563.json")).get("variants") or {"_":0}).keys())[0]
# 3. pooled table spot check vs d01_summary / d01_results
res = open("../d01_results.md").read()
def pooled_mean(key, var, met):
    d = pooled[(key, var, met)]; v = d.get("day", []) + d.get("night", []); return sum(v) / len(v), len(v)
for key, var, met in [("zs_pohang_yolo26n", ZSV, "F"), ("ft_pohang_yolo26x-depth_best", "refit", "absErrorRel"),
                      ("zs_multiaqua_yolo26m", ZSV, "SILog"), ("ft_multiaqua_yolo26l-depth_best", "refit", "F")]:
    mv, n = pooled_mean(key, var, met)
    print(f"pooled {key} {var} {met}: recomputed {mv:.2f} (n={n})  appears in d01_results: {f'{mv:.2f}' in res}")
# 4. checkpoint hashes
bad = 0; checked = 0
for jf in glob.glob("finetune/ft_*.json"):
    j = json.load(open(jf)); w = j["weights"]
    if not os.path.exists(w): print("MISSING ckpt", w); bad += 1; continue
    h = hashlib.sha256(open(w, "rb").read()).hexdigest(); checked += 1
    if h != j["weights_sha256"]: print("HASH MISMATCH", jf); bad += 1
print(f"checkpoints re-hashed: {checked}, problems: {bad}")
