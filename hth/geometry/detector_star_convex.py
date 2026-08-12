from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from .model import Candidate
METHOD="star_convex"
BASELINE_PARAMETERS={"ray_count":180,"minimum_radius_fraction":0.10,"maximum_radius_fraction":0.72,"minimum_support_fraction":0.55,"smoothing_window":5,"bbox_padding_fraction":0.0}
def _parameters(o):
 v=dict(BASELINE_PARAMETERS); o=o or {}; u=sorted(set(o)-set(v));
 if u: raise ValueError(f"Unknown Star-Convex parameters: {', '.join(u)}")
 v.update(o); v["ray_count"]=int(v["ray_count"]); v["smoothing_window"]=int(v["smoothing_window"])
 for k in set(v)-{"ray_count","smoothing_window"}: v[k]=float(v[k])
 return v
def _boundary(mask,v):
 m=(mask>0).astype(np.uint8); h,w=m.shape; ys,xs=np.nonzero(m); c=np.array([xs.mean(),ys.mean()]) if len(xs) else np.array([w/2,h/2]); rmax=min(h,w)*v["maximum_radius_fraction"]; rmin=min(h,w)*v["minimum_radius_fraction"]; pts=[]
 for a in np.linspace(0,2*np.pi,v["ray_count"],endpoint=False):
  rs=np.linspace(rmin,rmax,max(32,int(rmax))); xy=c+np.column_stack((np.cos(a)*rs,np.sin(a)*rs)); xi=np.clip(xy[:,0].astype(int),0,w-1); yi=np.clip(xy[:,1].astype(int),0,h-1); hit=np.where(m[yi,xi]>0)[0]
  if len(hit): pts.append(xy[hit[-1]])
 return c,np.asarray(pts,np.float32)
def detect(*,image_bgr,mask,parameters=None):
 del image_bgr; v=_parameters(parameters); c,pts=_boundary(mask,v); h,w=mask.shape; support=len(pts)/max(v["ray_count"],1)
 if support<v["minimum_support_fraction"] or len(pts)<4: return Candidate(METHOD,None,None,0,0,{"parameters":v,"reason":"insufficient_star_support","ray_support_fraction":support},status="no_candidate")
 win=max(1,v["smoothing_window"]); pts2=pts.copy()
 if win>1 and len(pts)>=win: pts2=np.asarray([pts[(i-win//2):(i+win//2+1)].mean(axis=0) if i>=win//2 and i+win//2<len(pts) else pts[i] for i in range(len(pts))],np.float32)
 hull=cv2.convexHull(pts2); corners=cv2.boxPoints(cv2.minAreaRect(hull)); x,y,bw,bh=cv2.boundingRect(corners.astype(np.float32)); pad=int(round(min(h,w)*v["bbox_padding_fraction"])); bbox=[max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad)]
 return Candidate(METHOD,bbox,corners.astype(float).tolist(),support,support,{"parameters":v,"center":c.tolist(),"ray_support_fraction":support,"evidence":"star_convex_mask_boundary"})
def debug_images(*,image_bgr,mask,parameters=None,candidate_corners=None,verbose=False):
 del verbose; v=_parameters(parameters); c,pts=_boundary(mask,v); ov=image_bgr.copy(); ci=tuple(np.rint(c).astype(int)); cv2.circle(ov,ci,5,(255,0,255),-1)
 for p in pts: cv2.line(ov,ci,tuple(np.rint(p).astype(int)),(0,255,255),1)
 if candidate_corners is not None: cv2.polylines(ov,[np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1,1,2)],True,(0,0,255),3)
 return {"star-rays.png":ov,"star-mask.png":np.where(mask>0,255,0).astype(np.uint8)}
__all__=["BASELINE_PARAMETERS","METHOD","debug_images","detect"]
