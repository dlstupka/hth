from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from .model import Candidate

METHOD="whitespace_frame"
BASELINE_PARAMETERS={
    "background_threshold":245,
    "minimum_border_background_fraction":0.55,
    "minimum_page_area_fraction":0.18,
    "maximum_page_area_fraction":0.98,
    "close_kernel_fraction":0.01,
    "bbox_padding_fraction":0.0,
}

def _parameters(overrides):
    v=dict(BASELINE_PARAMETERS); overrides=overrides or {}
    unknown=sorted(set(overrides)-set(v))
    if unknown: raise ValueError(f"Unknown Whitespace Frame parameters: {', '.join(unknown)}")
    v.update(overrides); v["background_threshold"]=int(v["background_threshold"])
    for k in set(v)-{"background_threshold"}: v[k]=float(v[k])
    return v

def _proposal(image,values):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    h,w=gray.shape
    threshold=int(values["background_threshold"])
    bright_background=np.where(gray>=threshold,255,0).astype(np.uint8)
    dark_threshold=255-threshold
    dark_background=np.where(gray<=dark_threshold,255,0).astype(np.uint8)

    def border_fraction(candidate):
        border=np.concatenate([candidate[0,:],candidate[-1,:],candidate[:,0],candidate[:,-1]])
        return float(np.count_nonzero(border))/len(border)

    bright_border_fraction=border_fraction(bright_background)
    dark_border_fraction=border_fraction(dark_background)
    if dark_border_fraction>bright_border_fraction:
        background=dark_background
        polarity="dark"
        selected_border_fraction=dark_border_fraction
    else:
        background=bright_background
        polarity="light"
        selected_border_fraction=bright_border_fraction
    diagnostics={
        "background_polarity":polarity,
        "background_threshold":threshold,
        "dark_background_threshold":dark_threshold,
        "border_background_fraction":selected_border_fraction,
        "bright_border_background_fraction":bright_border_fraction,
        "dark_border_background_fraction":dark_border_fraction,
    }
    inv=cv2.bitwise_not(background)
    k=max(3,int(round(min(h,w)*values["close_kernel_fraction"]))|1)
    inv=cv2.morphologyEx(inv,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(k,k)))
    contours,_=cv2.findContours(inv,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return background,None,diagnostics
    contour=max(contours,key=cv2.contourArea)
    return background,contour,diagnostics

def detect(*,image_bgr,mask,parameters=None):
    del mask
    v=_parameters(parameters); bg,c,diagnostics=_proposal(image_bgr,v); h,w=bg.shape; area=float(h*w)
    bf=diagnostics["border_background_fraction"]
    if bf<v["minimum_border_background_fraction"]:
        return Candidate(METHOD,None,None,0,0,{"parameters":v,"reason":"border_not_background",**diagnostics},status="no_candidate")
    if c is None:
        return Candidate(METHOD,None,None,0,0,{"parameters":v,"reason":"no_enclosed_page_region",**diagnostics},status="no_candidate")
    rect=cv2.minAreaRect(c); corners=cv2.boxPoints(rect); x,y,bw,bh=cv2.boundingRect(corners.astype(np.float32)); frac=bw*bh/area
    if not v["minimum_page_area_fraction"]<=frac<=v["maximum_page_area_fraction"]:
        return Candidate(METHOD,None,None,0,0,{"parameters":v,"reason":"page_area_out_of_range","page_area_fraction":frac,**diagnostics},status="no_candidate")
    pad=int(round(min(h,w)*v["bbox_padding_fraction"])); bbox=[max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad)]
    score=min(1.0,0.5*bf+0.5*min(1.0,frac/max(v["minimum_page_area_fraction"],1e-6)))
    return Candidate(METHOD,bbox,corners.astype(float).tolist(),score,score,{"parameters":v,"page_area_fraction":frac,"evidence":"surrounding_negative_space_frame",**diagnostics})

def debug_images(*,image_bgr,mask,parameters=None,candidate_corners=None,verbose=False):
    del mask,verbose
    v=_parameters(parameters); bg,c,diagnostics=_proposal(image_bgr,v); ov=image_bgr.copy()
    if c is not None: cv2.drawContours(ov,[c],-1,(0,255,255),2)
    if candidate_corners is not None:
        pts=np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1,1,2); cv2.polylines(ov,[pts],True,(0,0,255),3)
    cv2.putText(ov,f"background={diagnostics['background_polarity']} border={diagnostics['border_background_fraction']:.3f}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,cv2.LINE_AA)
    return {"whitespace-background.png":bg,"whitespace-frame.png":ov}

__all__=["BASELINE_PARAMETERS","METHOD","debug_images","detect"]
