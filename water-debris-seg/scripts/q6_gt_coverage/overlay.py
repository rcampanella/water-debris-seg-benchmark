import sys, numpy as np, cv2; sys.path.insert(0,sys.argv[1]); from rz import open_zip; from proj import *
seq,idx=sys.argv[2],int(sys.argv[3]); B="https://pohang-canal-dataset.s3.us-west-2.amazonaws.com/"
zl=open_zip(B+f"{seq}/lidar/lidar_front.zip"); names=sorted(n for n in zl.namelist() if n.endswith('.bin'))
lt=np.array([int(n.rsplit('/',1)[1][:-4]) for n in names])
ts=np.array([float(l.split()[0]) for l in open(sys.argv[4]) if l.strip()])
ci=int(np.argmin(abs(ts*1e9-lt[idx]))); print('lidar',lt[idx],'cam frame',ci,'dt ms',(ts[ci]*1e9-lt[idx])/1e6)
zs=open_zip(B+f"{seq}/stereo.zip"); img=cv2.imdecode(np.frombuffer(zs.read(f"stereo/left_images/{ci:06d}.png"),np.uint8),1)
R,t,K,D,wh=load_cal(sys.argv[5]); uv,z=project(read_scan(zl.read(names[idx])),R,t,K,D,wh)
print('depth pct',np.percentile(z,[5,50,95]).round(1));print('in FOV',len(z),'<=20m',(z<=20).sum(), 'bands',[int(((z>a)&(z<=a+5)).sum()) for a in (0,5,10,15)])
cm=cv2.applyColorMap(np.clip(z/80*255,0,255).astype(np.uint8)[:,None],cv2.COLORMAP_TURBO)[:,0]
for (u,v),c,d in zip(uv.astype(int),cm,z): cv2.circle(img,(u,v),4 if d<=20 else 2,tuple(int(x) for x in c),-1)
cv2.imwrite(sys.argv[6],img[::2,::2])
