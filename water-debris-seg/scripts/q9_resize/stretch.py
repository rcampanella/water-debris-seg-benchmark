import sys, io, json, numpy as np, cv2; sys.path.insert(0,sys.argv[1]); from rz import open_zip; from mqproj import *
from ultralytics import YOLO
S6,Q,W,out=sys.argv[1:5]; R,T,C,M,D,wh=load_mq(Q); m=YOLO(W)
frames=[r["frame"] for r in json.load(open(f"{S6}/mq_watermask_check.json"))]
z=open_zip("https://data.lmi.link/datasets/MULTIAQUA_full.zip",1<<20); res=[]
def met(p,g):
    ar=np.mean(np.abs(p-g)/g); s=np.median(g)/np.median(p); arm=np.mean(np.abs(p*s-g)/g)
    A=np.c_[p,np.ones_like(p)]; ab=np.linalg.lstsq(A,g,rcond=None)[0]; arl=np.mean(np.abs(A@ab-g)/g)
    d1=np.mean(np.maximum(p/g,g/p)<1.25); return dict(absrel=ar,absrel_median=arm,absrel_lsq=arl,delta1=d1,scale_med=s)
for fr in frames:
    img=cv2.imdecode(np.frombuffer(z.read(f"data/zed/{fr}.png"),np.uint8),1); H,Wd=img.shape[:2]
    gt=cv2.imdecode(np.frombuffer(z.read(f"annotations/{fr}.png"),np.uint8),-1)
    P=np.load(io.BytesIO(z.read(f"data/lidar/{fr}.npy"))); uv,d=mq_project(P,R,T,C,M,D,wh)
    u=np.clip(uv[:,0].astype(int),0,Wd-1); v=np.clip(uv[:,1].astype(int),0,H-1)
    keep=(d<=20)&~np.isin(gt[v,u],[3,5,6,7]); u,v,d=u[keep],v[keep],d[keep]
    if len(d)<50: continue
    dl=m.predict(img,imgsz=640,verbose=False,device='cpu')[0].depth.data.cpu().numpy()        # letterbox (official predict)
    st=cv2.resize(img,(640,640),interpolation=cv2.INTER_LINEAR)
    ds=m.predict(st,imgsz=640,verbose=False,device='cpu')[0].depth.data.cpu().numpy()          # stretched input
    ds=cv2.resize(ds,(Wd,H),interpolation=cv2.INTER_LINEAR)
    sp=np.zeros((H,Wd),np.float32); sp[v,u]=d
    nn=cv2.resize(sp,(640,640),interpolation=cv2.INTER_NEAREST)                            # Ultralytics val-style GT
    r={"frame":fr,"n_pts":int(len(d)),"n_pts_after_val_resize":int((nn>0).sum()),"pred_shape_letterbox":list(dl.shape),
       "letterbox":met(dl[v,u],d),"stretch":met(ds[v,u],d)}
    r={k:({a:round(float(b),4) for a,b in x.items()} if isinstance(x,dict) else x) for k,x in r.items()}; res.append(r); print(r,flush=True)
json.dump(res,open(out,"w"),indent=1)
for k in ("letterbox","stretch"):
    print(k,{a:round(float(np.mean([r[k][a] for r in res])),4) for a in res[0][k]})
print("frames",len(res),"GT points kept by Ultralytics val resize",sum(r["n_pts_after_val_resize"] for r in res),"of",sum(r["n_pts"] for r in res))
