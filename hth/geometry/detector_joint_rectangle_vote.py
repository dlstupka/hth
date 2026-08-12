from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from .model import Candidate

METHOD="joint_rectangle_vote"
BASELINE_PARAMETERS={
    "canny_low":50.0,
    "canny_high":150.0,
    "hough_threshold":80,
    "axis_tolerance_degrees":12.0,
    "minimum_side_support":0.18,
    "minimum_area_fraction":0.16,
    "bbox_padding_fraction":0.0,
}

def _parameters(overrides):
    v=dict(BASELINE_PARAMETERS); overrides=overrides or {}
    unknown=sorted(set(overrides)-set(v))
    if unknown: raise ValueError(f"Unknown Joint Rectangle Voting parameters: {', '.join(unknown)}")
    v.update(overrides); v["hough_threshold"]=int(v["hough_threshold"])
    for k in set(v)-{"hough_threshold"}: v[k]=float(v[k])
    return v

def _lines(image,v):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    edges=cv2.Canny(gray,int(v["canny_low"]),int(v["canny_high"]))
    lines=cv2.HoughLines(edges,1,np.pi/180,v["hough_threshold"])
    return edges, [] if lines is None else [tuple(x[0]) for x in lines]

def _candidate(image,v):
    edges,lines=_lines(image,v); h,w=edges.shape; tol=np.deg2rad(v["axis_tolerance_degrees"])
    vertical=[]; horizontal=[]
    for rho,theta in lines:
        a=min(abs(theta),abs(np.pi-theta))
        if a<=tol: vertical.append((rho,theta))
        if abs(theta-np.pi/2)<=tol: horizontal.append((rho,theta))
    if len(vertical)<2 or len(horizontal)<2: return edges,None,lines
    xs=sorted(r for r,t in vertical)
    ys=sorted(r for r,t in horizontal)
    left,right=xs[0],xs[-1]; top,bottom=ys[0],ys[-1]
    x1,x2=int(round(left)),int(round(right)); y1,y2=int(round(top)),int(round(bottom))
    if x2<=x1 or y2<=y1: return edges,None,lines
    area=(x2-x1)*(y2-y1)/(h*w)
    # Support each side in a thin edge band.
    band=max(2,int(round(min(h,w)*0.006)))
    supports=[
        np.count_nonzero(edges[max(0,y1-band):min(h,y1+band+1),max(0,x1):min(w,x2+1)]) / max(x2-x1,1),
        np.count_nonzero(edges[max(0,y2-band):min(h,y2+band+1),max(0,x1):min(w,x2+1)]) / max(x2-x1,1),
        np.count_nonzero(edges[max(0,y1):min(h,y2+1),max(0,x1-band):min(w,x1+band+1)]) / max(y2-y1,1),
        np.count_nonzero(edges[max(0,y1):min(h,y2+1),max(0,x2-band):min(w,x2+band+1)]) / max(y2-y1,1),
    ]
    return edges,{"bbox":[x1,y1,x2,y2],"area_fraction":area,"supports":supports},lines

def detect(*,image_bgr,mask,parameters=None):
    del mask
    v=_parameters(parameters); edges,c,lines=_candidate(image_bgr,v); h,w=edges.shape
    if c is None: return Candidate(METHOD,None,None,0,0,{"parameters":v,"reason":"insufficient_joint_line_families","line_count":len(lines)},status="no_candidate")
    if c["area_fraction"]<v["minimum_area_fraction"] or min(c["supports"])<v["minimum_side_support"]:
        return Candidate(METHOD,None,None,0,0,{"parameters":v,"reason":"weak_joint_rectangle","area_fraction":c["area_fraction"],"side_support":c["supports"]},status="no_candidate")
    x1,y1,x2,y2=c["bbox"]; pad=int(round(min(h,w)*v["bbox_padding_fraction"])); x1=max(0,x1-pad);y1=max(0,y1-pad);x2=min(w,x2+pad);y2=min(h,y2+pad)
    corners=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    score=min(1.0,0.5*min(c["supports"])+0.5*min(1.0,c["area_fraction"]/max(v["minimum_area_fraction"],1e-6)))
    return Candidate(METHOD,[x1,y1,x2,y2],corners,score,score,{"parameters":v,"area_fraction":c["area_fraction"],"side_support":c["supports"],"line_count":len(lines),"evidence":"joint_four_side_hough_vote"})

def debug_images(*,image_bgr,mask,parameters=None,candidate_corners=None,verbose=False):
    del mask,verbose
    v=_parameters(parameters); edges,c,lines=_candidate(image_bgr,v); ov=image_bgr.copy()
    for rho,theta in lines[:100]:
        a,b=np.cos(theta),np.sin(theta); x0,y0=a*rho,b*rho
        p1=(int(x0+2000*(-b)),int(y0+2000*a)); p2=(int(x0-2000*(-b)),int(y0-2000*a))
        cv2.line(ov,p1,p2,(0,255,255),1)
    if candidate_corners is not None:
        pts=np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1,1,2); cv2.polylines(ov,[pts],True,(0,0,255),3)
    return {"joint-rectangle-edges.png":edges,"joint-rectangle-votes.png":ov}

__all__=["BASELINE_PARAMETERS","METHOD","debug_images","detect"]
