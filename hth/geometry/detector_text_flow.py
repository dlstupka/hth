from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from .model import Candidate

METHOD = "text_flow"
BASELINE_PARAMETERS = {
    "minimum_component_area_fraction": 0.00002,
    "maximum_component_area_fraction": 0.01,
    "line_join_fraction": 0.030,
    "minimum_line_count": 3,
    "minimum_text_coverage_fraction": 0.08,
    "bbox_padding_fraction": 0.02,
}

def _parameters(overrides):
    values = dict(BASELINE_PARAMETERS)
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise ValueError(f"Unknown Text Flow parameters: {', '.join(unknown)}")
    values.update(overrides)
    values["minimum_line_count"] = int(values["minimum_line_count"])
    for key in set(values) - {"minimum_line_count"}:
        values[key] = float(values[key])
    return values

def _text_components(mask, values):
    m = np.where(mask > 0, 255, 0).astype(np.uint8)
    h, w = m.shape
    area = float(h*w)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    boxes = []
    for i in range(1, count):
        a = int(stats[i, cv2.CC_STAT_AREA])
        frac = a / area
        if values["minimum_component_area_fraction"] <= frac <= values["maximum_component_area_fraction"]:
            x,y,bw,bh = (int(stats[i,j]) for j in (cv2.CC_STAT_LEFT,cv2.CC_STAT_TOP,cv2.CC_STAT_WIDTH,cv2.CC_STAT_HEIGHT))
            boxes.append((x,y,bw,bh))
    return m, boxes

def _line_mask(mask, boxes, values):
    h,w = mask.shape
    out = np.zeros_like(mask)
    for x,y,bw,bh in boxes:
        cv2.rectangle(out, (x,y), (x+bw,y+bh), 255, -1)
    join = max(3, int(round(w * values["line_join_fraction"])))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (join, 3))
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

def detect(*, image_bgr, mask, parameters=None):
    del image_bgr
    values = _parameters(parameters)
    m, boxes = _text_components(mask, values)
    h,w = m.shape
    line_mask = _line_mask(m, boxes, values)
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = [c for c in contours if cv2.boundingRect(c)[2] >= 0.10*w]
    if len(lines) < values["minimum_line_count"]:
        return Candidate(METHOD,None,None,0,0,{"parameters":values,"reason":"insufficient_text_lines","line_count":len(lines)},status="no_candidate")
    pts = np.concatenate(lines, axis=0)
    rect = cv2.minAreaRect(pts)
    corners = cv2.boxPoints(rect)
    x,y,bw,bh = cv2.boundingRect(corners.astype(np.float32))
    coverage = float(np.count_nonzero(line_mask[y:y+bh, x:x+bw])) / max(bw*bh,1)
    if coverage < values["minimum_text_coverage_fraction"]:
        return Candidate(METHOD,None,None,0,0,{"parameters":values,"reason":"insufficient_text_coverage","line_count":len(lines),"text_coverage":coverage},status="no_candidate")
    pad = int(round(min(h,w)*values["bbox_padding_fraction"]))
    bbox=[max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad)]
    score=min(1.0,0.6*min(1.0,len(lines)/12.0)+0.4*min(1.0,coverage/max(values["minimum_text_coverage_fraction"],1e-6)))
    return Candidate(METHOD,bbox,corners.astype(float).tolist(),score,score,{"parameters":values,"line_count":len(lines),"text_coverage":coverage,"component_count":len(boxes),"evidence":"text_line_envelope"})

def debug_images(*, image_bgr, mask, parameters=None, candidate_corners=None, verbose=False):
    del verbose
    values=_parameters(parameters)
    m,boxes=_text_components(mask,values)
    lm=_line_mask(m,boxes,values)
    comp=image_bgr.copy()
    for x,y,w,h in boxes:
        cv2.rectangle(comp,(x,y),(x+w,y+h),(0,255,255),1)
    if candidate_corners is not None:
        pts=np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1,1,2)
        cv2.polylines(comp,[pts],True,(0,0,255),3)
    return {"text-components.png":comp,"text-lines.png":lm}

__all__=["BASELINE_PARAMETERS","METHOD","debug_images","detect"]
