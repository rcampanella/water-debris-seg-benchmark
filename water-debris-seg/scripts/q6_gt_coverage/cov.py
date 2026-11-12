import sys, json, numpy as np; sys.path.insert(0,sys.argv[1]); from rz import open_zip; from proj import *
from concurrent.futures import ThreadPoolExecutor
seq,step,out=sys.argv[2],int(sys.argv[3]),sys.argv[4]
B="https://pohang-canal-dataset.s3.us-west-2.amazonaws.com/"
R,t,K,D,wh=load_cal(f"{sys.argv[1]}/cal{seq[-2:]}")
names=sorted(n for n in open_zip(B+f"{seq}/lidar/lidar_front.zip").namelist() if n.endswith('.bin'))
pick=names[::step]
def work(chunk):
    z=open_zip(B+f"{seq}/lidar/lidar_front.zip"); res=[]
    for n in chunk:
        uv,d=project(read_scan(z.read(n)),R,t,K,D,wh)
        res.append({"t":int(n.rsplit('/',1)[1][:-4]),"fov":int(len(d)),"le20":int((d<=20).sum()),
                    "bands":[int(((d>a)&(d<=a+5)).sum()) for a in (0,5,10,15)],"dmin":float(d.min()) if len(d) else None})
    return res
chunks=[pick[i::4] for i in range(4)]
with ThreadPoolExecutor(4) as ex: rows=sorted(sum(ex.map(work,chunks),[]),key=lambda r:r["t"])
json.dump({"seq":seq,"n_scans":len(names),"step":step,"rows":rows},open(out,"w"))
le=np.array([r["le20"] for r in rows]); print(seq,len(names),len(rows),"frac>=1",(le>=1).mean().round(3),">=100",(le>=100).mean().round(3),">=1000",(le>=1000).mean().round(3),"median",np.median(le))
