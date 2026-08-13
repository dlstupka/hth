from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from .model import Candidate

METHOD="signed_polar_boundary_vote"
BASELINE_PARAMETERS={"ray_count":180,"inner_radius_fraction":0.12,"outer_radius_fraction":0.70,"gradient_percentile":82.0,"minimum_support_fraction":0.35,"bbox_padding_fraction":0.0,"polarity":"bright_inside"}

def _parameters(o):
    v=dict(BASELINE_PARAMETERS); o=o or {}; u=sorted(set(o)-set(v))
    if u: raise ValueError(f"Unknown Signed Polar Boundary Voting parameters: {', '.join(u)}")
    v.update(o); v["ray_count"]=int(v["ray_count"])
    for k in ("inner_radius_fraction","outer_radius_fraction","gradient_percentile","minimum_support_fraction","bbox_padding_fraction"): v[k]=float(v[k])
    if v["ray_count"]<16: raise ValueError("ray_count must be >= 16")
    if v["polarity"] not in {"bright_inside","dark_inside","absolute"}: raise ValueError("polarity must be bright_inside, dark_inside, or absolute")
    return v

def _evidence(image,v):
    g=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    gf=cv2.GaussianBlur(g,(0,0),1.0).astype(np.float32)
    gx=cv2.Sobel(gf,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gf,cv2.CV_32F,0,1,ksize=3)
    h,w=g.shape; c=np.array([w/2,h/2],np.float32); rmax=min(h,w)*v["outer_radius_fraction"]; rmin=min(h,w)*v["inner_radius_fraction"]; pts=[]
    evidence=np.zeros_like(gf)
    for a in np.linspace(0,2*np.pi,v["ray_count"],endpoint=False):
        ca,sa=np.cos(a),np.sin(a); rs=np.linspace(rmin,rmax,max(24,int(rmax-rmin))); xy=c+np.column_stack((ca*rs,sa*rs)); xi=np.clip(xy[:,0].astype(int),0,w-1); yi=np.clip(xy[:,1].astype(int),0,h-1)
        radial=gx[yi,xi]*ca+gy[yi,xi]*sa
        vals=-radial if v["polarity"]=="bright_inside" else radial if v["polarity"]=="dark_inside" else np.abs(radial)
        vals=np.maximum(vals,0); threshold=np.percentile(vals,v["gradient_percentile"]); idx=np.where((vals>=threshold)&(vals>0))[0]
        evidence[yi,xi]=np.maximum(evidence[yi,xi],vals)
        if len(idx): pts.append(xy[idx[-1]])
    return evidence,np.asarray(pts,np.float32)

def detect(*,image_bgr,mask,parameters=None):
    del mask; v=_parameters(parameters); ev,pts=_evidence(image_bgr,v); h,w=ev.shape
    if len(pts)<v["ray_count"]*v["minimum_support_fraction"]: return Candidate(METHOD,None,None,0,0,{"parameters":v,"reason":"insufficient_signed_ray_support","supported_rays":len(pts)},status="no_candidate")
    rect=cv2.minAreaRect(pts.reshape(-1,1,2)); corners=cv2.boxPoints(rect); x,y,bw,bh=cv2.boundingRect(corners.astype(np.float32)); pad=int(round(min(h,w)*v["bbox_padding_fraction"])); bbox=[max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad)]; support=len(pts)/v["ray_count"]
    return Candidate(METHOD,bbox,corners.astype(float).tolist(),support,support,{"parameters":v,"supported_rays":len(pts),"ray_support_fraction":support,"evidence":"signed_polar_gradient_boundary_votes"})

def debug_images(*,image_bgr,mask,parameters=None,candidate_corners=None,verbose=False):
    del mask,verbose; v=_parameters(parameters); ev,pts=_evidence(image_bgr,v); norm=cv2.normalize(ev,None,0,255,cv2.NORM_MINMAX).astype(np.uint8); ov=image_bgr.copy()
    for p in pts: cv2.circle(ov,tuple(np.rint(p).astype(int)),2,(0,255,255),-1)
    if candidate_corners is not None: cv2.polylines(ov,[np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1,1,2)],True,(0,0,255),3)
    return {"signed-polar-gradient.png":norm,"signed-polar-boundary-votes.png":ov}
__all__=["BASELINE_PARAMETERS","METHOD","debug_images","detect"]
