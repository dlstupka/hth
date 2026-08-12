from __future__ import annotations
import hashlib, os, threading
from pathlib import Path
import cv2
import numpy as np
from .model import Candidate

METHOD="learned_page_mask"
MODEL_ENV="HTH_LEARNED_PAGE_MASK_MODEL"
MODEL_INPUT_SIZE=(512,512)
MODEL_CONTRACT="onnx-rgb-512-single-channel-mask-v1"
BASELINE_PARAMETERS={
    "mask_threshold":0.50,
    "minimum_mask_area_fraction":0.15,
    "close_kernel_fraction":0.006,
    "polygon_epsilon_fraction":0.012,
    "bbox_padding_fraction":0.0,
}
_THREAD_LOCAL=threading.local()
_HASH_CACHE={}
_HASH_LOCK=threading.Lock()

def _parameters(overrides):
    values=dict(BASELINE_PARAMETERS); overrides=overrides or {}
    unknown=sorted(set(overrides)-set(values))
    if unknown: raise ValueError(f"Unknown Learned Page-Mask parameters: {', '.join(unknown)}")
    values.update(overrides)
    for key in values: values[key]=float(values[key])
    if not 0.0 < values["mask_threshold"] < 1.0: raise ValueError("mask_threshold must be between 0 and 1")
    return values

def _model_path():
    raw=os.environ.get(MODEL_ENV,"").strip()
    if not raw: raise RuntimeError(f"{METHOD} requires an ONNX model; set {MODEL_ENV} to the model path")
    path=Path(raw)
    if not path.is_file(): raise RuntimeError(f"{METHOD} model does not exist: {path}")
    if path.suffix.lower()!=".onnx": raise RuntimeError(f"{METHOD} requires an .onnx model: {path}")
    return path

def _model_sha256(path):
    stat=path.stat(); key=(str(path.resolve()),int(stat.st_mtime_ns),int(stat.st_size))
    with _HASH_LOCK: cached=_HASH_CACHE.get(key)
    if cached: return cached
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with _HASH_LOCK:
        _HASH_CACHE.clear(); _HASH_CACHE[key]=digest
    return digest

def _network(path):
    key=str(path.resolve())
    if getattr(_THREAD_LOCAL,"model_key",None)!=key:
        _THREAD_LOCAL.model_key=key
        _THREAD_LOCAL.network=cv2.dnn.readNetFromONNX(str(path))
    return _THREAD_LOCAL.network

def _probability_map(image_bgr,path):
    blob=cv2.dnn.blobFromImage(image_bgr,scalefactor=1.0/255.0,size=MODEL_INPUT_SIZE,mean=(0,0,0),swapRB=True,crop=False)
    net=_network(path); net.setInput(blob); raw=np.asarray(net.forward(),dtype=np.float32)
    if raw.ndim==4 and raw.shape[:2]==(1,1): raw=raw[0,0]
    elif raw.ndim==3 and raw.shape[0]==1: raw=raw[0]
    elif raw.ndim!=2: raise RuntimeError(f"{METHOD} model output must be one foreground channel; got shape {tuple(raw.shape)}")
    finite=raw[np.isfinite(raw)]
    if finite.size==0: raise RuntimeError(f"{METHOD} model produced no finite output")
    if float(finite.min())<0.0 or float(finite.max())>1.0:
        raw=np.clip(raw,-40,40); raw=1.0/(1.0+np.exp(-raw))
    h,w=image_bgr.shape[:2]
    return cv2.resize(raw,(w,h),interpolation=cv2.INTER_LINEAR)

def _proposal(image_bgr,values):
    path=_model_path(); probability=_probability_map(image_bgr,path)
    binary=np.where(probability>=values["mask_threshold"],255,0).astype(np.uint8)
    h,w=binary.shape
    if values["close_kernel_fraction"]>0:
        k=max(3,int(round(min(h,w)*values["close_kernel_fraction"]))|1)
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return path,probability,binary,max(contours,key=cv2.contourArea) if contours else None

def detect(*,image_bgr,mask,parameters=None):
    del mask
    values=_parameters(parameters); path,probability,binary,contour=_proposal(image_bgr,values)
    h,w=binary.shape; area=float(h*w)
    common={"parameters":values,"model_sha256":_model_sha256(path),"model_contract":MODEL_CONTRACT,"inference_backend":"opencv-dnn"}
    if contour is None: return Candidate(METHOD,None,None,0,0,{**common,"reason":"no_learned_page_region"},status="no_candidate")
    mask_area_fraction=float(cv2.contourArea(contour))/area
    if mask_area_fraction<values["minimum_mask_area_fraction"]:
        return Candidate(METHOD,None,None,0,0,{**common,"reason":"learned_mask_too_small","mask_area_fraction":mask_area_fraction},status="no_candidate")
    perimeter=max(cv2.arcLength(contour,True),1.0)
    approx=cv2.approxPolyDP(contour,values["polygon_epsilon_fraction"]*perimeter,True)
    corners=approx.reshape(4,2).astype(np.float32) if len(approx)==4 and cv2.isContourConvex(approx) else cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    x,y,bw,bh=cv2.boundingRect(corners); pad=int(round(min(h,w)*values["bbox_padding_fraction"]))
    x1,y1=max(0,x-pad),max(0,y-pad); x2,y2=min(w,x+bw+pad),min(h,y+bh+pad)
    selected=probability[binary>0]; mean_probability=float(selected.mean()) if selected.size else 0.0
    score=min(1.0,0.65*mean_probability+0.35*min(1.0,mask_area_fraction/0.5))
    return Candidate(METHOD,[x1,y1,x2,y2],corners.astype(float).tolist(),score,score,{**common,"mask_area_fraction":mask_area_fraction,"mean_page_probability":mean_probability,"evidence":"learned_page_segmentation_mask"})

def debug_images(*,image_bgr,mask,parameters=None,candidate_corners=None,verbose=False):
    del mask,verbose
    values=_parameters(parameters); path,probability,binary,contour=_proposal(image_bgr,values)
    prob=np.rint(np.clip(probability,0,1)*255).astype(np.uint8); overlay=image_bgr.copy()
    if contour is not None: cv2.drawContours(overlay,[contour],-1,(0,255,255),2)
    if candidate_corners is not None:
        pts=np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1,1,2); cv2.polylines(overlay,[pts],True,(0,0,255),3)
    cv2.putText(overlay,f"model {_model_sha256(path)[:12]}",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
    return {"learned-page-probability.png":prob,"learned-page-mask.png":binary,"learned-page-boundary.png":overlay}

__all__=["BASELINE_PARAMETERS","METHOD","MODEL_CONTRACT","MODEL_ENV","MODEL_INPUT_SIZE","debug_images","detect"]
