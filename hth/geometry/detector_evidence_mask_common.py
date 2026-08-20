from __future__ import annotations
import cv2, numpy as np
from .model import Candidate

BASELINE_PARAMETERS={"probability_threshold":0.5,"minimum_page_area_fraction":0.08,"close_kernel_fraction":0.006,"page_padding_fraction":0.02}

def parameters(overrides, *, label):
    values=dict(BASELINE_PARAMETERS); overrides=overrides or {}
    unknown=sorted(set(overrides)-set(values))
    if unknown: raise ValueError(f"Unknown {label} parameters: {', '.join(unknown)}")
    values.update(overrides)
    for key in values: values[key]=float(values[key])
    return values

def dominant(probability, values):
    prob=np.asarray(probability,dtype=np.float32)
    binary=np.where(prob>=values['probability_threshold'],255,0).astype(np.uint8)
    if values['close_kernel_fraction']>0:
        k=max(3,int(round(min(binary.shape)*values['close_kernel_fraction']))|1)
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary,8,cv2.CV_32S)
    if count<=1: return binary,None
    label=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    chosen=np.where(labels==label,255,0).astype(np.uint8)
    contours,_=cv2.findContours(chosen,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return chosen,max(contours,key=cv2.contourArea) if contours else None

def candidate(method,image_bgr,probability,values,diagnostics):
    mask,contour=dominant(probability,values); h,w=image_bgr.shape[:2]
    base={**diagnostics,"parameters":values,"probability_min":float(np.min(probability)),"probability_max":float(np.max(probability)),"probability_mean":float(np.mean(probability))}
    if contour is None: return Candidate(method,None,None,0,0,{**base,"reason":"no_learned_page_region"},status="no_candidate")
    ph,pw=probability.shape[:2]; area=float(cv2.contourArea(contour))/max(1.0,float(ph*pw)); base['mask_area_fraction']=area
    if area<values['minimum_page_area_fraction']: return Candidate(method,None,None,0,0,{**base,"reason":"learned_mask_too_small"},status="no_candidate")
    corners=cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    corners[:,0]*=w/float(pw); corners[:,1]*=h/float(ph)
    frac=values['page_padding_fraction']
    if frac>0:
        center=corners.mean(axis=0); corners=center+(corners-center)*(1+2*frac); corners[:,0]=np.clip(corners[:,0],0,w-1); corners[:,1]=np.clip(corners[:,1],0,h-1)
    x,y,bw,bh=cv2.boundingRect(corners); selected=np.asarray(probability)[mask>0]
    score=float(selected.mean()) if selected.size else 0.0
    return Candidate(method,[int(x),int(y),int(x+bw),int(y+bh)],corners.astype(float).tolist(),score,score,base)

def debug_images(image_bgr,probability,values,candidate_corners=None,prefix='learned'):
    mask,contour=dominant(probability,values); h,w=image_bgr.shape[:2]
    prob_full=cv2.resize(np.asarray(probability,dtype=np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
    mask_full=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST); overlay=image_bgr.copy()
    if contour is not None:
        pts=contour.reshape(-1,2).astype(np.float32); pts[:,0]*=w/float(probability.shape[1]); pts[:,1]*=h/float(probability.shape[0]); cv2.polylines(overlay,[np.rint(pts).astype(np.int32).reshape(-1,1,2)],True,(0,255,255),2)
    if candidate_corners is not None: cv2.polylines(overlay,[np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1,1,2)],True,(0,0,255),3)
    return {f'{prefix}-probability.png':np.rint(np.clip(prob_full,0,1)*255).astype(np.uint8),f'{prefix}-mask.png':mask_full,f'{prefix}-boundary.png':overlay}
