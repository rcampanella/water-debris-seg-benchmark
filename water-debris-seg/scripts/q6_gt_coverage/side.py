import sys, json, numpy as np; sys.path.insert(0,sys.argv[1]); from rz import open_zip; from proj import *
seq=sys.argv[2]; B="https://pohang-canal-dataset.s3.us-west-2.amazonaws.com/"
front=json.load(open(f"{sys.argv[1]}/cov_{seq}.json"))["rows"]
res={}
for lid in ("lidar_port","lidar_starboard"):
    R,t,K,D,wh=load_cal(f"{sys.argv[1]}/cal{seq[-2:]}",lidar=lid)
    z=open_zip(B+f"{seq}/lidar/{lid}.zip"); names=sorted(n for n in z.namelist() if n.endswith('.bin'))
    lt=np.array([int(n.rsplit('/',1)[1][:-4]) for n in names]); rows=[]
    for r in front[::8]:
        i=int(np.argmin(abs(lt-r["t"]))); b=z.read(names[i])
        if len(b)%28: print(lid,'size',len(b)); break
        uv,d=project(read_scan(b),R,t,K,D,wh)
        rows.append({"t":r["t"],"front_le20":r["le20"],"side_fov":int(len(d)),"side_le20":int((d<=20).sum()),"side_bands":[int(((d>a)&(d<=a+5)).sum()) for a in (0,5,10,15)]})
    res[lid]=rows
    a=np.array([[x["front_le20"],x["side_le20"]] for x in rows]); print(lid,len(rows),'frames side>=1',(a[:,1]>=1).mean().round(3),'front0 & side>=1',((a[:,0]==0)&(a[:,1]>=1)).mean().round(3),'side bands sum',np.sum([x["side_bands"] for x in rows],0))
json.dump(res,open(f"{sys.argv[1]}/side_{seq}.json","w"))
