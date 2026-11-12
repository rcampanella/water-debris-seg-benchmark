import sys, json, numpy as np; sys.path.insert(0,sys.argv[1]); from rz import open_zip; from proj import *
seq=sys.argv[2]; B="https://pohang-canal-dataset.s3.us-west-2.amazonaws.com/"
E=json.load(open(f"{sys.argv[1]}/cal{seq[-2:]}/extrinsics.json")); Rl=quat2R(E['lidar_front']['quaternion']); tl=np.array(E['lidar_front']['translation'])
R,t,K,D,wh=load_cal(f"{sys.argv[1]}/cal{seq[-2:]}")
z=open_zip(B+f"{seq}/lidar/lidar_front.zip"); names=sorted(n for n in z.namelist() if n.endswith('.bin'))
for idx in map(int,sys.argv[3:]):
    P=read_scan(z.read(names[idx])); Pb=P@Rl.T+tl          # body frame, FRD (z down)
    Pc=P@R.T+t; inf=Pc[:,2]>0.5
    h=-Pb[:,2]                                               # height above body origin (up +)
    rng=np.linalg.norm(Pb[:,:2],axis=1)
    print(idx,'all pts',len(P),'height pct (1,5,25,50)',np.percentile(h,[1,5,25,50]).round(2))
    hist,edges=np.histogram(h,bins=np.arange(-6,4,0.25)); 
    print('  hist height (m):',' '.join(f"{e:.2f}:{c}" for e,c in zip(edges[:-1],hist) if c>50))
    for lo,hi in ((-1.0,0.0),(0.0,0.75),(0.75,4)):
        m=(h>=lo)&(h<hi); r=rng[m]
        print(f'  h[{lo},{hi}) n={m.sum()} range pct(5,50,95)',np.percentile(r,[5,50,95]).round(1) if m.sum() else '-', 'in cam FOV',int((m&inf).sum()))
