import sys, numpy as np, cv2; sys.path.insert(0,sys.argv[1]); from mqproj import *
R,T,C,M,D,wh=load_mq(sys.argv[2]); P=np.load(sys.argv[3]); img=cv2.imread(sys.argv[4])
print(P.shape, P[:2]); uv,z=mq_project(P,R,T,C,M,D,wh)
print('depth pct',np.percentile(z,[5,50,95]).round(1),'in FOV',len(z),'<=20',(z<=20).sum(),[int(((z>a)&(z<=a+5)).sum()) for a in (0,5,10,15)])
cm=cv2.applyColorMap(np.clip(z/80*255,0,255).astype(np.uint8)[:,None],cv2.COLORMAP_TURBO)[:,0]
for (u,v),c,d in zip(uv.astype(int),cm,z): cv2.circle(img,(u,v),5 if d<=20 else 3,tuple(int(x) for x in c),-1)
cv2.imwrite(sys.argv[5],img[::2,::2])
