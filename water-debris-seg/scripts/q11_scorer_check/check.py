import sys, glob, json, numpy as np, cv2, torch, warnings
S,O=sys.argv[1:3]; sys.path.insert(0,f'{S}/mdec'); warnings.filterwarnings('ignore')
import types; _six=types.ModuleType('torch._six'); _six.string_classes=(str,); sys.modules['torch._six']=_six  # compat shim: removed in torch 2.0
class CD:   # Chamfer: pure torch (compiled extension not built on the login node); same outputs (squared NN distances)
    def __call__(self,a,b):
        d=torch.cdist(a.double(),b.double())**2; return d.min(2).values.float(), d.min(1).values.float()
_cdm=types.ModuleType('src.external_libs.chamfer_distance.chamfer_distance'); _cdm.ChamferDistance=CD; _cdm.__all__=['ChamferDistance']; sys.modules['src.external_libs.chamfer_distance.chamfer_distance']=_cdm  # pre-stub: compiled CUDA ext not built on login node
import src.core.metrics as mt, src.core.evaluator as ev
mt.ChamferDistance=CD
names=sorted(f.rsplit('/',1)[1][:-4] for f in glob.glob(f'{S}/gt/*.png'))
rd=lambda p: cv2.imread(p,-1).astype(np.float64)/256.0
# ---------- 1) absolute metrics: devkit vs our numpy vs MDEC metrics_benchmark (same quantised PNG values) ----------
def ref_abs(p,g):
    e=np.abs(g-p); dl=np.log(g)-np.log(p)
    return [e.mean(),np.sqrt((e**2).mean()),np.abs(1/g-1/p).mean(),np.sqrt(((1/g-1/p)**2).mean()),np.abs(dl).mean(),
            np.sqrt((dl**2).mean()),np.sqrt((dl**2).mean()-dl.mean()**2),(e/g).mean(),(e**2/g**2).mean()]
keys=['mae','rmse','inverse mae','inverse rmse','log mae','log rmse','scale invariant log','abs relative','squared relative']
to_mdec={'mae':('MAE',1),'rmse':('RMSE',1),'inverse mae':('InvMAE',1e-3),'inverse rmse':('InvRMSE',1e-3),'log mae':('LogMAE',1e-2),
         'log rmse':('LogRMSE',1e-2),'scale invariant log':('LogSI',1e-2),'abs relative':('AbsRel',1e-2),'squared relative':('SqRel',1e-2)}
R=[];M=[]
for n in names:
    g=rd(f'{S}/gt/{n}.png'); p=rd(f'{S}/pred/{n}.png'); m=g>0
    R.append(ref_abs(p[m],g[m])); mb=mt.metrics_benchmark(p[m],g[m]); M.append([mb[to_mdec[k][0]]*to_mdec[k][1] for k in keys])
R=np.array(R);M=np.array(M)
dev={l.split(':')[0].replace('mean ','').strip():float(l.split(':')[1]) for l in open(f'{S}/pred/stats_depth.txt') if l.startswith('mean')}
per=[]
for i,n in enumerate(names[:20]):   # devkit writes per-image errors for the first 20 files (alphasort = sorted)
    v=np.array([float(x) for x in open(f'{S}/pred/errors_out/{n}.txt').read().split()]); per.append(v)
per=np.array(per)
abs_out={'devkit_mean':dev,'ours_mean':dict(zip(keys,R.mean(0))),'mdec_mean_raw_units':dict(zip(keys,M.mean(0))),
 'max_rel_diff_mean_devkit_vs_ours':float(np.max(np.abs(R.mean(0)-np.array([dev[k] for k in keys]))/np.abs(np.array([dev[k] for k in keys])))),
 'max_rel_diff_mean_devkit_vs_mdec':float(np.max(np.abs(M.mean(0)-np.array([dev[k] for k in keys]))/np.abs(np.array([dev[k] for k in keys])))),
 'max_rel_diff_per_image_devkit_vs_ours_first20':float(np.max(np.abs(R[:20]-per)/np.abs(per)))}
# ---------- 2) relative metrics: MDEC evaluator (patched) vs our numpy, float predictions ----------
E=ev.MonoDepthEvaluator(mode='mono',metrics=('benchmark','pointcloud'),min=1e-3,max=20)
def ref_rel(pred,t,mask,K):
    p=pred.astype(np.float64); lo,hi=np.percentile(p[mask],[5,95]); fm=mask&(p>lo)&(p<hi)
    a,b=np.linalg.lstsq(np.c_[p[fm],np.ones(fm.sum())],t[fm].astype(np.float64),rcond=None)[0]; p=np.clip(p*a+b,1e-3,20)
    x=p[mask]; y=t[mask].astype(np.float64); e=np.abs(x-y)
    vv,uu=np.nonzero(mask); rays=(np.linalg.inv(K[:3,:3].astype(np.float64))@np.c_[uu,vv,np.ones_like(uu)].T).T
    d=torch.cdist(torch.tensor(rays*x[:,None]),torch.tensor(rays*y[:,None])); pr=(d.min(1).values<0.1).double().mean(); rc=(d.min(0).values<0.1).double().mean()
    F=float(100*2*pr*rc/(pr+rc)) if (pr+rc)>0 else 0.0
    return {'MAE':e.mean(),'RMSE':np.sqrt((e**2).mean()),'AbsRel':100*(e/y).mean(),'F-Score':F}
rel=[]
for n in names:
    z=np.load(f'{O}/data/{n}.npz'); pred,t,K=z['pred'],z['target'],z['K']; mask=E._get_mask(t)
    m=E._eval_single(pred.copy(),t,mask,K,None,None,['benchmark','pointcloud']); r=ref_rel(pred,t,mask,K)
    rel.append({'frame':n,'mdec':{k:m[k] for k in r},'ours':r})
mx={k:max(abs(x['mdec'][k]-x['ours'][k]) for x in rel) for k in rel[0]['ours']}
rel_out={'frames':len(rel),'mean_mdec':{k:float(np.mean([x['mdec'][k] for x in rel])) for k in mx},
         'mean_ours':{k:float(np.mean([x['ours'][k] for x in rel])) for k in mx},'max_abs_diff_per_image':mx}
# synthetic: affine prediction must be perfect after alignment
rng=np.random.default_rng(0); t=np.zeros((120,200),np.float32); mk=rng.random(t.shape)<0.05; t[mk]=rng.uniform(1,19.9,mk.sum())
Ks=np.array([[100,0,100,0],[0,100,60,0],[0,0,1,0],[0,0,0,1]],np.float32); p=np.where(mk,(t-1)/2,0.5).astype(np.float32)
sm=E._eval_single(p,t,E._get_mask(t),Ks,None,None,['benchmark','pointcloud'])
out={'absolute':abs_out,'relative':rel_out,'synthetic_affine':{k:sm[k] for k in ('MAE','AbsRel','F-Score')}}
json.dump(out,open(f'{S}/final_check.json','w'),indent=1,default=float)
print(json.dumps({'abs devkit vs ours (mean, max rel diff)':abs_out['max_rel_diff_mean_devkit_vs_ours'],
 'abs devkit vs ours (per image, max rel diff)':abs_out['max_rel_diff_per_image_devkit_vs_ours_first20'],
 'abs devkit vs MDEC (mean, max rel diff)':abs_out['max_rel_diff_mean_devkit_vs_mdec'],
 'rel MDEC vs ours (max abs diff per image)':mx,'rel means MDEC':rel_out['mean_mdec'],'synthetic':out['synthetic_affine']},indent=1,default=float))
