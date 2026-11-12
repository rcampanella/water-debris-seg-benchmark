import numpy as np, cv2
def ld(fn):
    fs=cv2.FileStorage(fn,cv2.FILE_STORAGE_READ); return {k:fs.getNode(k).mat() for k in fs.root().keys() if fs.getNode(k).isMap()}
def load_mq(Q):
    L=ld(f'{Q}/lidar_zed.yaml'); fs=cv2.FileStorage(f'{Q}/zed.yaml',cv2.FILE_STORAGE_READ)
    M=fs.getNode('cameraMatrix').mat(); D=fs.getNode('distCoeffs').mat(); return L['R'],L['T'].ravel(),L['C'].astype(float),M,D,(2208,1242)
def mq_project(P,R,T,C,M,D,wh):
    Pc=(R@(C@P[:,:3].T)).T+T; ok=Pc[:,2]>0.5; Pc=Pc[ok]
    if len(Pc)==0: return np.zeros((0,2)),np.zeros(0)
    uv,_=cv2.projectPoints(Pc.astype(np.float64),np.zeros(3),np.zeros(3),M,D); uv=uv[:,0]; w,h=wh
    inb=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h); return uv[inb],Pc[inb,2]
