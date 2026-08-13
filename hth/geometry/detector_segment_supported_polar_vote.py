from __future__ import annotations
import cv2
import numpy as np
from .model import Candidate
from . import detector_polar_boundary_vote

METHOD="segment_supported_polar_vote"
BASELINE_PARAMETERS={"ray_count":180,"inner_radius_fraction":0.12,"outer_radius_fraction":0.70,"gradient_percentile":82.0,"minimum_support_fraction":0.35,"segment_distance_fraction":0.018,"minimum_segment_length_fraction":0.10,"minimum_segment_support_fraction":0.30,"bbox_padding_fraction":0.0}

def _parameters(o):
    v=dict(BASELINE_PARAMETERS); o=o or {}; u=sorted(set(o)-set(v))
    if u: raise ValueError(f"Unknown Segment-Supported Polar Voting parameters: {', '.join(u)}")
    v.update(o); v["ray_count"]=int(v["ray_count"])
    for k in set(v)-{"ray_count"}: v[k]=float(v[k])
    if v["ray_count"]<16: raise ValueError("ray_count must be >= 16")
    return v

def _point_segment_distance(p,a,b):
    ab=b-a; den=float(np.dot(ab,ab))
    if den<=1e-9: return float(np.linalg.norm(p-a))
    t=np.clip(float(np.dot(p-a,ab))/den,0.0,1.0); return float(np.linalg.norm(p-(a+t*ab)))

def _evidence(image,v):
    pv={k:v[k] for k in ("ray_count","inner_radius_fraction","outer_radius_fraction","gradient_percentile","minimum_support_fraction","bbox_padding_fraction")}
    mag,pts=detector_polar_boundary_vote._evidence(image,pv)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD); found=lsd.detect(gray)[0]; h,w=gray.shape; diag=float(np.hypot(h,w)); min_len=diag*v["minimum_segment_length_fraction"]
    segments=[]
    if found is not None:
        for line in found[:,0,:]:
            a=np.array(line[:2],np.float32); b=np.array(line[2:],np.float32)
            if np.linalg.norm(b-a)>=min_len: segments.append((a,b))
    max_dist=diag*v["segment_distance_fraction"]; supported=[]
    for p in pts:
        if any(_point_segment_distance(p,a,b)<=max_dist for a,b in segments): supported.append(p)
    return mag,pts,np.asarray(supported,np.float32),segments

def detect(*,image_bgr,mask,parameters=None):
    del mask; v=_parameters(parameters); mag,raw,pts,segments=_evidence(image_bgr,v); h,w=mag.shape
    needed=max(v["ray_count"]*v["minimum_support_fraction"]*v["minimum_segment_support_fraction"],4)
    if len(pts)<needed: return Candidate(METHOD,None,None,0,0,{"parameters":v,"reason":"insufficient_segment_supported_votes","raw_votes":len(raw),"supported_votes":len(pts),"segments":len(segments)},status="no_candidate")
    rect=cv2.minAreaRect(pts.reshape(-1,1,2)); corners=cv2.boxPoints(rect); x,y,bw,bh=cv2.boundingRect(corners.astype(np.float32)); pad=int(round(min(h,w)*v["bbox_padding_fraction"])); bbox=[max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad)]; support=len(pts)/max(1,len(raw))
    return Candidate(METHOD,bbox,corners.astype(float).tolist(),support,support,{"parameters":v,"raw_votes":len(raw),"supported_votes":len(pts),"segment_count":len(segments),"segment_support_fraction":support,"evidence":"polar_votes_supported_by_lsd_segments"})

def debug_images(*,image_bgr,mask,parameters=None,candidate_corners=None,verbose=False):
    del mask,verbose; v=_parameters(parameters); mag,raw,pts,segments=_evidence(image_bgr,v); norm=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX).astype(np.uint8); ov=image_bgr.copy()
    for a,b in segments: cv2.line(ov,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),(255,160,0),1)
    for p in raw: cv2.circle(ov,tuple(np.rint(p).astype(int)),1,(100,100,255),-1)
    for p in pts: cv2.circle(ov,tuple(np.rint(p).astype(int)),3,(0,255,255),-1)
    if candidate_corners is not None: cv2.polylines(ov,[np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1,1,2)],True,(0,0,255),3)
    return {"segment-polar-gradient.png":norm,"segment-supported-polar-votes.png":ov}
__all__=["BASELINE_PARAMETERS","METHOD","debug_images","detect"]
