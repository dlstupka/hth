from __future__ import annotations
import json, os, threading
from pathlib import Path
import cv2, numpy as np
from .model import Candidate
METHOD="learned_page_mask"
PROTOTXT_ENV="HTH_LEARNED_PAGE_MASK_PROTOTXT"; WEIGHTS_ENV="HTH_LEARNED_PAGE_MASK_WEIGHTS"; PROVENANCE_ENV="HTH_LEARNED_PAGE_MASK_PROVENANCE"
BASELINE_PARAMETERS={"mask_threshold":0.50,"minimum_mask_area_fraction":0.15,"close_kernel_fraction":0.006,"polygon_epsilon_fraction":0.012,"bbox_padding_fraction":0.0}
_THREAD_LOCAL=threading.local()
def _parameters(o):
 v=dict(BASELINE_PARAMETERS); o=o or {}; u=sorted(set(o)-set(v))
 if u: raise ValueError(f"Unknown Learned Page-Mask parameters: {', '.join(u)}")
 v.update(o)
 for k in v: v[k]=float(v[k])
 return v
def _asset(name):
 raw=os.environ.get(name,"").strip()
 if not raw: raise RuntimeError(f"{METHOD} lifecycle did not set {name}")
 p=Path(raw)
 if not p.is_file(): raise RuntimeError(f"{METHOD} asset does not exist: {p}")
 return p
def _assets():
 p=_asset(PROTOTXT_ENV); w=_asset(WEIGHTS_ENV); prov=json.loads(_asset(PROVENANCE_ENV).read_text(encoding="utf-8"))
 return p,w,prov
def _network(proto,weights):
 key=(str(proto.resolve()),str(weights.resolve()))
 if getattr(_THREAD_LOCAL,"key",None)!=key:
  _THREAD_LOCAL.key=key
  _THREAD_LOCAL.net=cv2.dnn.readNet(str(weights),str(proto),"Caffe")
 return _THREAD_LOCAL.net
def _probability(image):
 proto,weights,prov=_assets(); resized=cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA).astype(np.float32)
 normalized=0.0039*(resized-127.0); blob=np.transpose(normalized,(2,0,1))[None,:,:,:]
 net=_network(proto,weights); net.setInput(blob); raw=np.asarray(net.forward("out"),dtype=np.float32)
 if raw.ndim==4 and raw.shape[:2]==(1,1): raw=raw[0,0]
 elif raw.ndim==3 and raw.shape[0]==1: raw=raw[0]
 elif raw.ndim!=2: raise RuntimeError(f"{METHOD} PageNet output has unexpected shape {tuple(raw.shape)}")
 h,w=image.shape[:2]; return np.clip(cv2.resize(raw,(w,h),interpolation=cv2.INTER_LINEAR),0,1),prov
def _proposal(image,v):
 prob,prov=_probability(image); binary=np.where(prob>=v["mask_threshold"],255,0).astype(np.uint8); h,w=binary.shape
 if v["close_kernel_fraction"]>0:
  k=max(3,int(round(min(h,w)*v["close_kernel_fraction"]))|1); binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
 contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 return prob,binary,max(contours,key=cv2.contourArea) if contours else None,prov
def detect(*,image_bgr,mask,parameters=None):
 del mask; v=_parameters(parameters); prob,binary,contour,prov=_proposal(image_bgr,v); h,w=binary.shape; area=float(h*w)
 common={"parameters":v,"model_id":prov.get("model_id","pagenet-ohio"),"model_family":"PageNet","model_weights_sha256":prov.get("weights_sha256"),"model_license":prov.get("license"),"model_source":prov.get("upstream_repository"),"inference_backend":"opencv-dnn-caffe"}
 if contour is None: return Candidate(METHOD,None,None,0,0,{**common,"reason":"no_learned_page_region"},status="no_candidate")
 af=float(cv2.contourArea(contour))/area
 if af<v["minimum_mask_area_fraction"]: return Candidate(METHOD,None,None,0,0,{**common,"reason":"learned_mask_too_small","mask_area_fraction":af},status="no_candidate")
 peri=max(cv2.arcLength(contour,True),1.0); approx=cv2.approxPolyDP(contour,v["polygon_epsilon_fraction"]*peri,True)
 corners=approx.reshape(4,2).astype(np.float32) if len(approx)==4 and cv2.isContourConvex(approx) else cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
 x,y,bw,bh=cv2.boundingRect(corners); pad=int(round(min(h,w)*v["bbox_padding_fraction"])); x1,y1=max(0,x-pad),max(0,y-pad); x2,y2=min(w,x+bw+pad),min(h,y+bh+pad)
 selected=prob[binary>0]; mp=float(selected.mean()) if selected.size else 0.0; score=min(1.0,0.65*mp+0.35*min(1.0,af/0.5))
 return Candidate(METHOD,[x1,y1,x2,y2],corners.astype(float).tolist(),score,score,{**common,"mask_area_fraction":af,"mean_page_probability":mp,"evidence":"pagenet_learned_page_segmentation"})
def debug_images(*,image_bgr,mask,parameters=None,candidate_corners=None,verbose=False):
 del mask,verbose; v=_parameters(parameters); prob,binary,contour,prov=_proposal(image_bgr,v); overlay=image_bgr.copy()
 if contour is not None: cv2.drawContours(overlay,[contour],-1,(0,255,255),2)
 if candidate_corners is not None: cv2.polylines(overlay,[np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1,1,2)],True,(0,0,255),3)
 return {"learned-page-probability.png":np.rint(prob*255).astype(np.uint8),"learned-page-mask.png":binary,"learned-page-boundary.png":overlay}
__all__=["BASELINE_PARAMETERS","METHOD","PROTOTXT_ENV","WEIGHTS_ENV","PROVENANCE_ENV","debug_images","detect"]
