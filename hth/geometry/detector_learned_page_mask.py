from __future__ import annotations
import json, os, threading
from pathlib import Path
import cv2, numpy as np
from .model import Candidate

METHOD="learned_page_mask"
PROTOTXT_ENV="HTH_LEARNED_PAGE_MASK_PROTOTXT"
WEIGHTS_ENV="HTH_LEARNED_PAGE_MASK_WEIGHTS"
PROVENANCE_ENV="HTH_LEARNED_PAGE_MASK_PROVENANCE"
OUTPUT_LAYER_ENV="HTH_LEARNED_PAGE_MASK_OUTPUT_LAYER"
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
    if getattr(_THREAD_LOCAL,"key",None)!=key or not hasattr(_THREAD_LOCAL,"net"):
        # Publish the cache key only after construction succeeds.  A failed
        # backend load must not leave thread-local state looking initialized.
        net=cv2.dnn.readNet(str(weights),str(proto),"Caffe")
        _THREAD_LOCAL.net=net
        _THREAD_LOCAL.key=key
    return _THREAD_LOCAL.net

def _probability_256(image):
    """Run PageNet using the preprocessing and 256x256 output contract of upstream test_pretrained.py."""
    proto,weights,prov=_assets()
    resized=cv2.resize(image,(256,256),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    normalized=0.0039*(resized-127.0)
    blob=np.transpose(normalized,(2,0,1))[None,:,:,:]
    net=_network(proto,weights); net.setInput(blob)
    output_layer=os.environ.get(OUTPUT_LAYER_ENV,"").strip()
    if not output_layer:
        raise RuntimeError(f"{METHOD} lifecycle did not set {OUTPUT_LAYER_ENV}")
    raw=np.asarray(net.forward(output_layer),dtype=np.float32)
    if raw.ndim==4 and raw.shape[:2]==(1,1): raw=raw[0,0]
    elif raw.ndim==3 and raw.shape[0]==1: raw=raw[0]
    elif raw.ndim!=2: raise RuntimeError(f"{METHOD} PageNet output has unexpected shape {tuple(raw.shape)}")
    if raw.shape != (256,256):
        raw=cv2.resize(raw,(256,256),interpolation=cv2.INTER_LINEAR)
    return np.clip(raw,0,1),prov

def _fill_holes(binary):
    if not np.any(binary): return binary
    flood=binary.copy(); h,w=binary.shape
    mask=np.zeros((h+2,w+2),np.uint8)
    # Find a background seed instead of assuming (0,0) is background.
    border=np.concatenate((binary[0,:],binary[-1,:],binary[:,0],binary[:,-1]))
    if np.all(border): return binary
    seed=None
    for x in range(w):
        if binary[0,x]==0: seed=(x,0); break
        if binary[h-1,x]==0: seed=(x,h-1); break
    if seed is None:
        for y in range(h):
            if binary[y,0]==0: seed=(0,y); break
            if binary[y,w-1]==0: seed=(w-1,y); break
    cv2.floodFill(flood,mask,seed,255)
    return binary | cv2.bitwise_not(flood)

def _postprocess(prob,v):
    # Threshold at native PageNet output resolution, matching upstream behavior,
    # before any display-size interpolation can blur the decision boundary.
    binary=np.where(prob>=v["mask_threshold"],255,0).astype(np.uint8)
    if v["close_kernel_fraction"]>0:
        k=max(3,int(round(256*v["close_kernel_fraction"]))|1)
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary,4,cv2.CV_32S)
    if count<=1:
        return binary,None
    label=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    dominant=np.where(labels==label,255,0).astype(np.uint8)
    dominant=_fill_holes(dominant)
    contours,_=cv2.findContours(dominant,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return dominant,max(contours,key=cv2.contourArea) if contours else None

def _proposal(image,v):
    prob,prov=_probability_256(image)
    binary,contour=_postprocess(prob,v)
    return prob,binary,contour,prov

def _scale_points(points,width,height):
    scale=np.array([width/256.0,height/256.0],dtype=np.float32)
    return np.asarray(points,dtype=np.float32)*scale

def detect(*,image_bgr,mask,parameters=None):
    del mask
    v=_parameters(parameters); prob,binary,contour,prov=_proposal(image_bgr,v); h,w=image_bgr.shape[:2]
    selected=prob[binary>0]
    diagnostics={
        "parameters":v,"model_id":prov.get("model_id","pagenet-ohio"),"model_family":"PageNet",
        "model_weights_sha256":prov.get("weights_sha256"),"model_license":prov.get("license"),
        "model_source":prov.get("upstream_repository"),"inference_backend":"opencv-dnn-caffe",
        "probability_min":float(prob.min()),"probability_max":float(prob.max()),"probability_mean":float(prob.mean()),
        "thresholded_fraction":float(np.count_nonzero(binary))/float(binary.size),
    }
    if contour is None:
        return Candidate(METHOD,None,None,0,0,{**diagnostics,"reason":"no_learned_page_region"},status="no_candidate")
    area_fraction=float(cv2.contourArea(contour))/float(256*256)
    diagnostics["mask_area_fraction"]=area_fraction
    if area_fraction<v["minimum_mask_area_fraction"]:
        return Candidate(METHOD,None,None,0,0,{**diagnostics,"reason":"learned_mask_too_small"},status="no_candidate")
    peri=max(cv2.arcLength(contour,True),1.0)
    approx=cv2.approxPolyDP(contour,v["polygon_epsilon_fraction"]*peri,True)
    corners256=approx.reshape(4,2).astype(np.float32) if len(approx)==4 and cv2.isContourConvex(approx) else cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    corners=_scale_points(corners256,w,h)
    x,y,bw,bh=cv2.boundingRect(corners)
    pad=int(round(min(h,w)*v["bbox_padding_fraction"]))
    x1,y1=max(0,x-pad),max(0,y-pad); x2,y2=min(w,x+bw+pad),min(h,y+bh+pad)
    mean_probability=float(selected.mean()) if selected.size else 0.0
    score=min(1.0,0.65*mean_probability+0.35*min(1.0,area_fraction/0.5))
    diagnostics.update({"mean_page_probability":mean_probability,"evidence":"pagenet_learned_page_segmentation","postprocess_resolution":"256x256"})
    return Candidate(METHOD,[x1,y1,x2,y2],corners.astype(float).tolist(),score,score,diagnostics)

def debug_images(*,image_bgr,mask,parameters=None,candidate_corners=None,verbose=False):
    del mask,verbose
    v=_parameters(parameters); prob,binary,contour,prov=_proposal(image_bgr,v); h,w=image_bgr.shape[:2]
    prob_full=cv2.resize(prob,(w,h),interpolation=cv2.INTER_LINEAR)
    mask_full=cv2.resize(binary,(w,h),interpolation=cv2.INTER_NEAREST)
    overlay=image_bgr.copy()
    if contour is not None:
        scaled=_scale_points(contour.reshape(-1,2),w,h)
        cv2.polylines(overlay,[np.rint(scaled).astype(np.int32).reshape(-1,1,2)],True,(0,255,255),2)
    if candidate_corners is not None:
        cv2.polylines(overlay,[np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1,1,2)],True,(0,0,255),3)
    return {"learned-page-probability.png":np.rint(prob_full*255).astype(np.uint8),"learned-page-mask.png":mask_full,"learned-page-boundary.png":overlay}

__all__=["BASELINE_PARAMETERS","METHOD","PROTOTXT_ENV","WEIGHTS_ENV","PROVENANCE_ENV","OUTPUT_LAYER_ENV","debug_images","detect"]
