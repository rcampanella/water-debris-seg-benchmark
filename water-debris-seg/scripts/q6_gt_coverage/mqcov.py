import sys, io, json, numpy as np; sys.path.insert(0,sys.argv[1]); from rz import open_zip; from mqproj import *
from concurrent.futures import ThreadPoolExecutor
U="https://data.lmi.link/datasets/MULTIAQUA_full.zip"; R,T,C,M,D,wh=load_mq(sys.argv[2])
names=sorted(n for n in open_zip(U,1<<20).namelist() if n.startswith('data/lidar/') and n.endswith('.npy'))
def work(ch):
    z=open_zip(U,1<<19); out=[]
    for n in ch:
        P=np.load(io.BytesIO(z.read(n))); uv,d=mq_project(P,R,T,C,M,D,wh); fr=n.rsplit('/',1)[1][:-4]
        try: g=json.loads(z.read(f'data/GPSIMU/{fr}.json')); ll=[g['latitude'],g['longitude']]
        except KeyError: ll=None
        out.append({"frame":fr,"fov":int(len(d)),"le20":int((d<=20).sum()),"bands":[int(((d>a)&(d<=a+5)).sum()) for a in (0,5,10,15)],"latlon":ll})
    return out
import os, time
part=sys.argv[3]+'.part'; done=set()
if os.path.exists(part): done={json.loads(l)['frame'] for l in open(part)}
todo=[n for n in names if n.rsplit('/',1)[1][:-4] not in done]
with open(part,'a') as f:
    for i in range(0,len(todo),50):
        for r in work(todo[i:i+50]): f.write(json.dumps(r)+'\n')
        f.flush(); print('done',i+50,flush=True); time.sleep(2)
rows=sorted({r['frame']:r for r in map(json.loads,open(part))}.values(),key=lambda r:r['frame'])
json.dump({"n_lidar":len(names),"rows":rows},open(sys.argv[3],"w")); print(len(rows))
