import sys, json, numpy as np, cv2; sys.path.insert(0,sys.argv[1]); from rz import open_zip; from proj import *
from ultralytics import YOLO
seq=sys.argv[2]; W=sys.argv[3]; idxs=list(map(int,sys.argv[5:])); B="https://pohang-canal-dataset.s3.us-west-2.amazonaws.com/"
m=YOLO(W); names_cls=m.names; print(names_cls)
ts=np.array([float(l.split()[0]) for l in open(sys.argv[4]) if l.strip()])
E=json.load(open(f"{sys.argv[1]}/cal{seq[-2:]}/extrinsics.json")); Rl=quat2R(E['lidar_front']['quaternion']); tl=np.array(E['lidar_front']['translation'])
R,t,K,D,wh=load_cal(f"{sys.argv[1]}/cal{seq[-2:]}")
zl=open_zip(B+f"{seq}/lidar/lidar_front.zip"); ln=sorted(n for n in zl.namelist() if n.endswith('.bin')); lt=np.array([int(n.rsplit('/',1)[1][:-4]) for n in ln])
zs=open_zip(B+f"{seq}/stereo.zip"); out=[]
for idx in idxs:
    ci=int(np.argmin(abs(ts*1e9-lt[idx]))); img=cv2.imdecode(np.frombuffer(zs.read(f"stereo/left_images/{ci:06d}.png"),np.uint8),1)
    r=m.predict(img,imgsz=640,conf=0.25,retina_masks=True,verbose=False,device='cpu')[0]
    water=np.zeros(img.shape[:2],bool)
    if r.masks is not None:
        for c,mk in zip(r.boxes.cls.int().tolist(), r.masks.data.cpu().numpy()):
            if names_cls[c].lower()=='water': water|=mk>0.5
    P=read_scan(zl.read(ln[idx])); h=-(P@Rl.T+tl)[:,2]
    Pc=P@R.T+t; ok=Pc[:,2]>0.5; P2=Pc[ok]; hh=h[ok]
    uv,_=cv2.projectPoints(P2.astype(np.float64),np.zeros(3),np.zeros(3),K,D); uv=uv[:,0]
    inb=(uv[:,0]>=0)&(uv[:,0]<wh[0])&(uv[:,1]>=0)&(uv[:,1]<wh[1]); uv=uv[inb]; d=P2[inb,2]; hh=hh[inb]
    u,v=uv[:,0].astype(int),uv[:,1].astype(int); onw=water[v,u]; le=d<=20
    rec={"seq":seq,"lidar_idx":idx,"cam_frame":ci,"water_frac_img":round(float(water.mean()),3),"n_fov":int(len(d)),"n_le20":int(le.sum()),
         "on_water_all":int(onw.sum()),"on_water_le20":int((onw&le).sum()),
         "height_on_water_pct":np.percentile(hh[onw],[5,50,95]).round(2).tolist() if onw.any() else None,
         "height_off_water_pct":np.percentile(hh[~onw],[5,50,95]).round(2).tolist() if (~onw).any() else None}
    print(rec); out.append(rec)
    vis=img.copy(); vis[water]=(0.5*vis[water]+0.5*np.array([255,128,0])).astype(np.uint8)
    for (a,b),w,dd in zip(uv.astype(int),onw,d): cv2.circle(vis,(a,b),4,(0,0,255) if w else (0,255,0),-1)
    cv2.imwrite(f"{sys.argv[1]}/wm_{seq}_{idx}.jpg",vis[::2,::2])
json.dump(out,open(f"{sys.argv[1]}/wm_{seq}.json","w"),indent=1)
