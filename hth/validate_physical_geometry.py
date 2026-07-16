#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from statistics import mean, median
from typing import Any

PASS, NEAR_PASS, FAIL, SKIP = "pass", "near_pass", "fail", "skip"

def args():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reference","--golden",dest="reference",type=Path,required=True)
    ap.add_argument("--analysis",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--fail-on",choices=("never","fail","near-pass"),default="fail")
    return ap.parse_args()

def valid_box(v):
    return isinstance(v,list) and len(v)==4 and all(isinstance(x,(int,float)) and math.isfinite(float(x)) for x in v) and v[2]>v[0] and v[3]>v[1]

def norm(v): return [int(round(float(x))) for x in v]
def width(b): return max(0,b[2]-b[0])
def height(b): return max(0,b[3]-b[1])
def area(b): return width(b)*height(b)

def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1])
    ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    union=area(a)+area(b)-inter
    return 0.0 if union<=0 else inter/union

def actual_box(r:dict[str,Any]):
    for keys in (("page_left_px","page_top_px","page_right_px","page_bottom_px"),
                 ("content_left_px","content_top_px","content_right_px","content_bottom_px")):
        if all(k in r for k in keys):
            v=[r[k] for k in keys]
            if valid_box(v): return norm(v)
    v=r.get("physical_document_bbox")
    return norm(v) if valid_box(v) else None

def classify(expected,actual,strict_iou,strict_px,near_iou,near_ratio):
    if actual is None:
        return FAIL,["no_valid_prediction"],{"iou":None,"maximum_edge_error_px":None,"maximum_edge_error_ratio":None,"edge_deltas_px":None,"edge_error_ratios":None}
    score=iou(expected,actual)
    names=("left","top","right","bottom")
    vals=[actual[i]-expected[i] for i in range(4)]
    deltas=dict(zip(names,vals))
    ratios={"left":abs(vals[0])/max(1,width(expected)),
            "right":abs(vals[2])/max(1,width(expected)),
            "top":abs(vals[1])/max(1,height(expected)),
            "bottom":abs(vals[3])/max(1,height(expected))}
    max_px=max(abs(x) for x in vals)
    max_ratio=max(ratios.values())
    d={"iou":round(score,6),"maximum_edge_error_px":max_px,
       "maximum_edge_error_ratio":round(max_ratio,6),
       "edge_deltas_px":deltas,
       "edge_error_ratios":{k:round(v,6) for k,v in ratios.items()},
       "width_delta_px":width(actual)-width(expected),
       "height_delta_px":height(actual)-height(expected),
       "area_delta_px":area(actual)-area(expected)}
    if score>=strict_iou and max_px<=strict_px: return PASS,[],d
    if score>=near_iou and max_ratio<=near_ratio:
        why=[]
        if score<strict_iou: why.append(f"iou_below_strict_threshold:{score:.4f}")
        if max_px>strict_px: why.append(f"edge_error_above_strict_threshold:{max_px}")
        return NEAR_PASS,why,d
    why=[]
    if score<near_iou: why.append(f"iou_below_near_threshold:{score:.4f}")
    if max_ratio>near_ratio: why.append(f"normalized_edge_error_exceeded:{max_ratio:.4f}")
    return FAIL,why or ["geometry_outside_acceptance"],d

def summary(items):
    comp=[x for x in items if x.get("iou") is not None]
    ious=[x["iou"] for x in comp]
    px=[x["maximum_edge_error_px"] for x in comp]
    ratio=[x["maximum_edge_error_ratio"] for x in comp]
    signed={}
    absolute={}
    for e in ("left","top","right","bottom"):
        vals=[x["edge_deltas_px"][e] for x in comp if x.get("edge_deltas_px")]
        signed[e]=round(mean(vals),3) if vals else None
        absolute[e]=round(mean(abs(v) for v in vals),3) if vals else None
    return {"comparable_page_count":len(comp),
            "mean_iou":round(mean(ious),6) if ious else None,
            "median_iou":round(median(ious),6) if ious else None,
            "mean_maximum_edge_error_px":round(mean(px),3) if px else None,
            "median_maximum_edge_error_px":round(median(px),3) if px else None,
            "mean_maximum_edge_error_ratio":round(mean(ratio),6) if ratio else None,
            "mean_signed_edge_delta_px":signed,
            "mean_absolute_edge_delta_px":absolute}

def main():
    a=args()
    ref=json.loads(a.reference.read_text(encoding="utf-8"))
    ana=json.loads(a.analysis.read_text(encoding="utf-8"))
    records={int(r["global_ordinal"]):r for r in ana.get("records",[]) if "global_ordinal" in r}
    ac=ref.get("acceptance",{})
    strict_iou=float(ac.get("minimum_iou",.95))
    strict_px=int(ac.get("maximum_edge_error_px",20))
    near_iou=float(ac.get("near_pass_minimum_iou",.90))
    near_ratio=float(ac.get("near_pass_maximum_edge_error_ratio",.06))
    results=[]
    for er in ref.get("pages",[]):
        n=int(er["global_ordinal"]); eb=er.get("physical_document_bbox")
        if not valid_box(eb):
            results.append({"global_ordinal":n,"status":SKIP,"reasons":["reference_bbox_not_populated"]}); continue
        expected=norm(eb); ar=records.get(n)
        if ar is None:
            results.append({"global_ordinal":n,"status":FAIL,"expected_bbox":expected,"actual_bbox":None,"reasons":["missing_analysis_record"],"iou":None,"maximum_edge_error_px":None,"maximum_edge_error_ratio":None,"edge_deltas_px":None,"edge_error_ratios":None}); continue
        actual=actual_box(ar)
        status,reasons,diag=classify(expected,actual,strict_iou,strict_px,near_iou,near_ratio)
        results.append({"global_ordinal":n,"label":er.get("label",""),
                        "reference_layout_type":er.get("layout_type"),
                        "analysis_quality_status":ar.get("quality_status"),
                        "analysis_quality_score":ar.get("quality_score"),
                        "status":status,"expected_bbox":expected,"actual_bbox":actual,
                        **diag,"reasons":reasons})
    report={"schema_version":"0.2","collection_id":ref.get("collection_id",""),
            "validator":"physical_geometry",
            "acceptance":{"pass":{"minimum_iou":strict_iou,"maximum_edge_error_px":strict_px},
                          "near_pass":{"minimum_iou":near_iou,"maximum_edge_error_ratio":near_ratio}},
            "reference_page_count":len(ref.get("pages",[])),
            "pass_count":sum(x["status"]==PASS for x in results),
            "near_pass_count":sum(x["status"]==NEAR_PASS for x in results),
            "fail_count":sum(x["status"]==FAIL for x in results),
            "skip_count":sum(x["status"]==SKIP for x in results),
            "summary":summary(results),"results":results}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(report,indent=2,ensure_ascii=False)
    a.output.write_text(text+"\n",encoding="utf-8")
    print(text)
    hard=report["fail_count"]>0; near=report["near_pass_count"]>0
    if a.fail_on=="never": return 0
    if a.fail_on=="near-pass": return 1 if hard or near else 0
    return 1 if hard else 0

if __name__=="__main__":
    raise SystemExit(main())
