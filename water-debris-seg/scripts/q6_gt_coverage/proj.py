import json, numpy as np, cv2
def quat2R(q):
    x,y,z,w=q; n=np.sqrt(x*x+y*y+z*z+w*w); x,y,z,w=x/n,y/n,z/n,w/n
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def load_cal(d, lidar='lidar_front', cam='stereo_left'):
    E=json.load(open(f'{d}/extrinsics.json')); I=json.load(open(f'{d}/intrinsics.json'))[cam]
    Rl,tl=quat2R(E[lidar]['quaternion']),np.array(E[lidar]['translation'])
    Rc,tc=quat2R(E[cam]['quaternion']),np.array(E[cam]['translation'])
    R=Rc.T@Rl; t=Rc.T@(tl-tc)                      # lidar -> camera
    K=np.array([[I['focal_length'],0,I['cc_x']],[0,I['focal_length'],I['cc_y']],[0,0,1]])
    return R,t,K,np.array(I['distortion_coefficients']),(I['image_width'],I['image_height'])
def read_scan(b):
    r=np.frombuffer(b,np.uint8).reshape(-1,28); f=r[:,:16].copy().view(np.float32)
    keep=np.abs(f[:,:3]).sum(1)>0; return f[keep,:3]
def project(P,R,t,K,D,wh):
    Pc=P@R.T+t; ok=Pc[:,2]>0.5; Pc=Pc[ok]
    if len(Pc)==0: return np.zeros((0,2)),np.zeros(0)
    uv,_=cv2.projectPoints(Pc.astype(np.float64),np.zeros(3),np.zeros(3),K,D); uv=uv[:,0]
    w,h=wh; inb=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    return uv[inb],Pc[inb,2]
