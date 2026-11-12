import json, glob, sys, numpy as np, collections
S=sys.argv[1]; out={}
def stats(le, bands):
    le=np.array(le); b=np.array(bands) if len(bands) else np.zeros((0,4))
    return {"n":int(len(le)),"frac_ge1":round(float((le>=1).mean()),3),"frac_ge50":round(float((le>=50).mean()),3),
            "frac_ge500":round(float((le>=500).mean()),3),"median":float(np.median(le)),"p90":float(np.percentile(le,90)),
            "band_frac_frames_ge1":[round(float((b[:,k]>=1).mean()),3) for k in range(4)],
            "band_points_share":[round(float(b[:,k].sum()/max(b.sum(),1)),3) for k in range(4)]}
for f in sorted(glob.glob(f"{S}/cov_pohang0?.json")):
    d=json.load(open(f)); r=d["rows"]; le=[x["le20"] for x in r]
    s=stats(le,[x["bands"] for x in r]); s["n_scans_total"]=d["n_scans"]; s["step"]=d["step"]
    z=np.array(le)==0; runs=[]; c=0
    for v in z:
        if v: c+=1
        else:
            if c: runs.append(c); c=0
    if c: runs.append(c)
    s["longest_zero_run_s"]=max(runs)*d["step"]/10 if runs else 0
    out[d["seq"]]=s
p=f"{S}/cov_multiaqua.json"
try:
    d=json.load(open(p)); rows=d["rows"]
except FileNotFoundError:
    rows=[json.loads(l) for l in open(p+".part")]; d={"n_lidar":None}
sp={}
for k in ("train","val","test"):
    for l in open(f"{S}/../../../f55947ca-4b54-4ca8-8755-6f2d1e9f6a42/scratchpad/mq_probe/x/split/{k}.txt") if False else []: pass
g=collections.defaultdict(list)
for x in rows: g[x["frame"].split("_")[0]].append(x)
for k,v in sorted(g.items()): out["multiaqua_"+k]=stats([x["le20"] for x in v],[x["bands"] for x in v])
out["multiaqua_all"]=stats([x["le20"] for x in rows],[x["bands"] for x in rows]); out["multiaqua_all"]["n_lidar_files"]=d.get("n_lidar")
json.dump(out,open(f"{S}/q6_coverage_summary.json","w"),indent=1)
for k,v in out.items(): print(k, {a:v[a] for a in ("n","frac_ge1","frac_ge50","frac_ge500","median","band_frac_frames_ge1")}, v.get("longest_zero_run_s",""))
