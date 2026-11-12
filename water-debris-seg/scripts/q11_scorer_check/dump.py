import sys, io, json, numpy as np, cv2; sys.path.insert(0,sys.argv[1]); from rz import open_zip; from mqproj import *
from ultralytics import YOLO
S6,Q,W,OUT=sys.argv[1:5]; R,T,C,M,D,wh=load_mq(Q); m=YOLO(W)
frames=[r["frame"] for r in json.load(open(f"{S6}/mq_watermask_check.json"))]
z=open_zip("https://data.lmi.link/datasets/MULTIAQUA_full.zip",1<<20)
for fr in frames:
    img=cv2.imdecode(np.frombuffer(z.read(f"data/zed/{fr}.png"),np.uint8),1); H,Wd=img.shape[:2]
    gt=cv2.imdecode(np.frombuffer(z.read(f"annotations/{fr}.png"),np.uint8),-1)
    P=np.load(io.BytesIO(z.read(f"data/lidar/{fr}.npy"))); uv,d=mq_project(P,R,T,C,M,D,wh)
    u=np.clip(uv[:,0].astype(int),0,Wd-1); v=np.clip(uv[:,1].astype(int),0,H-1)
    keep=(d<=20)&~np.isin(gt[v,u],[3,5,6,7]); u,v,d=u[keep],v[keep],d[keep]
    if len(d)<50: continue
    o=np.argsort(-d); u,v,d=u[o],v[o],d[o]          # several points per pixel: nearest written last (kept)
    tgt=np.zeros((H,Wd),np.float32); tgt[v,u]=d
    pr=m.predict(img,imgsz=640,verbose=False,device='cpu')[0].depth.data.cpu().numpy().astype(np.float32)
    np.savez_compressed(f"{OUT}/{fr}.npz",pred=pr,target=tgt,K=np.r_[np.c_[M,np.zeros(3)],[[0,0,0,1]]].astype(np.float32))
    print(fr,(tgt>0).sum(),flush=True)
