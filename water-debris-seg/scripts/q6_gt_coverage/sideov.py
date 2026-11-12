import sys, numpy as np, cv2; sys.path.insert(0,sys.argv[1]); from rz import open_zip; from proj import *
seq,tq=sys.argv[2],int(sys.argv[3]); B="https://pohang-canal-dataset.s3.us-west-2.amazonaws.com/"
ts=np.array([float(l.split()[0]) for l in open(sys.argv[4]) if l.strip()]); ci=int(np.argmin(abs(ts*1e9-tq)))
img=cv2.imdecode(np.frombuffer(open_zip(B+f"{seq}/stereo.zip").read(f"stereo/left_images/{ci:06d}.png"),np.uint8),1)
for lid,col in (("lidar_front",None),("lidar_port",(255,0,255)),("lidar_starboard",(255,255,0))):
    R,t,K,D,wh=load_cal(f"{sys.argv[1]}/cal{seq[-2:]}",lidar=lid); z=open_zip(B+f"{seq}/lidar/{lid}.zip")
    names=sorted(n for n in z.namelist() if n.endswith('.bin')); lt=np.array([int(n.rsplit('/',1)[1][:-4]) for n in names])
    uv,d=project(read_scan(z.read(names[int(np.argmin(abs(lt-tq)))])),R,t,K,D,wh); k=d<=20
    print(lid,'<=20m',k.sum(),'0-5m',(d<=5).sum())
    cm=cv2.applyColorMap(np.clip(d/20*255,0,255).astype(np.uint8)[:,None],cv2.COLORMAP_TURBO)[:,0]
    for (u,v),c,dd in zip(uv[k].astype(int),cm[k],d[k]): cv2.circle(img,(u,v),4,tuple(int(x) for x in c) if col is None else col,-1)
cv2.imwrite(sys.argv[5],img[::2,::2])
