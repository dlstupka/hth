from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from .model import Candidate
METHOD="distance_transform_rect"
BASELINE_PARAMETERS={"distance_threshold_fraction":0.18,"minimum_core_area_fraction":0.006,"proposal_expansion_fraction":0.12,"minimum_mask_coverage":0.12,"minimum_bbox_area_fraction":0.14,"bbox_padding_fraction":0.0}
def _parameters(o):
 v=dict(BASELINE_PARAMETERS); o=o or {}; u=sorted(set(o)-set(v));
 if u: raise ValueError(f"Unknown Distance-Transform Rectangle Proposal parameters: {', '.join(u)}")
 v.update(o)
 for k in v: v[k]=float(v[k])
 return v
def _proposal(mask,v):
 m=np.where(mask>0,255,0).astype(np.uint8); d=cv2.distanceTransform(m,cv2.DIST_L2,5); mx=float(d.max()); core=np.where(d>=mx*v["distance_threshold_fraction"],255,0).astype(np.uint8) if mx>0 else np.zeros_like(m); pts=cv2.findNonZero(core)
 if pts is None: return d,core,None
 x,y,w,h=cv2.boundingRect(pts); ex=int(round(w*v["proposal_expansion_fraction"])); ey=int(round(h*v["proposal_expansion_fraction"])); H,W=m.shape; return d,core,(max(0,x-ex),max(0,y-ey),min(W,x+w+ex),min(H,y+h+ey))
def detect(*,image_bgr,mask,parameters=None):
 del image_bgr; v=_parameters(parameters); d,core,p=_proposal(mask,v); H,W=mask.shape; area=float(H*W); corefrac=np.count_nonzero(core)/area
 if p is None or corefrac<v["minimum_core_area_fraction"]: return Candidate(METHOD,None,None,0,0,{"parameters":v,"reason":"insufficient_distance_core","core_area_fraction":corefrac},status="no_candidate")
 x1,y1,x2,y2=p; pad=int(round(min(H,W)*v["bbox_padding_fraction"])); x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(W,x2+pad); y2=min(H,y2+pad); af=(x2-x1)*(y2-y1)/area; coverage=np.count_nonzero(mask[y1:y2,x1:x2])/max((x2-x1)*(y2-y1),1)
 if af<v["minimum_bbox_area_fraction"] or coverage<v["minimum_mask_coverage"]: return Candidate(METHOD,None,None,0,0,{"parameters":v,"reason":"weak_rectangle_proposal","bbox_area_fraction":af,"mask_coverage":coverage},status="no_candidate")
 corners=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]; score=min(1.0,0.55*coverage+0.45*min(1.0,af/max(v["minimum_bbox_area_fraction"],1e-6))); return Candidate(METHOD,[x1,y1,x2,y2],corners,score,score,{"parameters":v,"core_area_fraction":corefrac,"bbox_area_fraction":af,"mask_coverage":coverage,"evidence":"distance_core_rectangle_proposal"})
def debug_images(*,image_bgr,mask,parameters=None,candidate_corners=None,verbose=False):
 del verbose; v=_parameters(parameters); d,core,p=_proposal(mask,v); mx=float(d.max()); norm=np.rint(d/max(mx,1e-9)*255).astype(np.uint8); ov=image_bgr.copy()
 if p: cv2.rectangle(ov,(p[0],p[1]),(p[2],p[3]),(0,255,255),2)
 if candidate_corners is not None: cv2.polylines(ov,[np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1,1,2)],True,(0,0,255),3)
 return {"distance-rect-transform.png":norm,"distance-rect-core.png":core,"distance-rect-proposal.png":ov}
__all__=["BASELINE_PARAMETERS","METHOD","debug_images","detect"]
