import sys, io, json, random, numpy as np, cv2; sys.path.insert(0,sys.argv[1]); from rz import open_zip; from mqproj import *
from ultralytics import YOLO
S,Q,W=sys.argv[1:4]; R,T,C,M,D,wh=load_mq(Q); m=YOLO(W)
rows=json.load(open(f"{S}/cov_multiaqua.json"))["rows"]; random.seed(0)
pick=[]
for sq in ("adr1","bl1","lj1","lj2","lj3"):
    c=[r["frame"] for r in rows if r["frame"].startswith(sq+"_") and r["le20"]>=50]; pick+=random.sample(c,min(6,len(c)))
z=open_zip("https://data.lmi.link/datasets/MULTIAQUA_full.zip",1<<20); names=set(z.namelist()); out=[]
for fr in pick:
    if f"annotations/{fr}.png" not in names: continue
    img=cv2.imdecode(np.frombuffer(z.read(f"data/zed/{fr}.png"),np.uint8),1)
    gt=cv2.imdecode(np.frombuffer(z.read(f"annotations/{fr}.png"),np.uint8),-1); gw=np.isin(gt,[3,5]); ign=np.isin(gt,[6,7])
    r=m.predict(img,imgsz=640,conf=0.25,retina_masks=True,verbose=False,device='cpu')[0]; pw=np.zeros(gt.shape,bool)
    if r.masks is not None:
        for c,mk in zip(r.boxes.cls.int().tolist(),r.masks.data.cpu().numpy()):
            if m.names[c]=='Water': pw|=mk>0.5
    v=~ign; iou=float((pw&gw&v).sum()/max(((pw|gw)&v).sum(),1))
    P=np.load(io.BytesIO(z.read(f"data/lidar/{fr}.npy"))); uv,d=mq_project(P,R,T,C,M,D,wh)
    u=np.clip(uv[:,0].astype(int),0,wh[0]-1); vv=np.clip(uv[:,1].astype(int),0,wh[1]-1); le=d<=20
    g=gw[vv,u]&le; ig=ign[vv,u]&le
    dil={}
    for k in (0,5,10,20):
        pk=cv2.dilate(pw.astype(np.uint8),np.ones((2*k+1,2*k+1),np.uint8))>0 if k else pw
        p=pk[vv,u]&le; dil[k]=[int((g&~p).sum()),int((~g&p&~ig).sum())]
    p=pw[vv,u]&le
    rec={"dilation_missed_removed":dil,"frame":fr,"water_iou":round(iou,3),"pts_le20":int(le.sum()),"pts_on_gt_water":int(g.sum()),"gt_water_kept_by_pred":int((g&~p).sum()),
         "pts_on_gt_nonwater_removed_by_pred":int((~g&p&~ig).sum()),"pts_on_ignore_or_own_boat":int(ig.sum())}
    out.append(rec); print(rec,flush=True)
json.dump(out,open(f"{S}/mq_watermask_check.json","w"),indent=1)
for k in ("0","5","10","20"):
    ms=sum(r["dilation_missed_removed"][int(k)][0] for r in out); rm=sum(r["dilation_missed_removed"][int(k)][1] for r in out)
    rm_x=sum(r["dilation_missed_removed"][int(k)][1] for r in out if r["frame"]!="adr1_11_008400")
    print("dilate",k,"px: missed water pts",ms,"of",sum(r["pts_on_gt_water"] for r in out),"| non-water removed",rm,"(excl. pier frame",rm_x,")")
a=lambda k: sum(r[k] for r in out)
print("frames",len(out),"mean water IoU",round(np.mean([r['water_iou'] for r in out]),3),"pts<=20",a('pts_le20'),"on GT water",a('pts_on_gt_water'),"kept by pred (missed)",a('gt_water_kept_by_pred'),"nonwater removed",a('pts_on_gt_nonwater_removed_by_pred'),"ignore/own boat",a('pts_on_ignore_or_own_boat'))
