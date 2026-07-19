"""Canonical raw and derived regression reports."""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Any
from .parameter_space import canonical_parameters


def ranking_key(result: dict[str, Any]) -> tuple[float, float, int, float]:
    s=result["summary"]; edge=s["mean_edge_error_px"]
    return (-float(s["mean_iou"]), -float(s["minimum_iou"]), int(s["failure_count"]), float(edge) if edge is not None else float("inf"))


def write_raw_results(path: Path, ranked: list[dict[str, Any]]) -> None:
    fields=["run_id","parameter_set_id","profile","rank","global_ordinal","label","layout_type","status","iou","left_error_px","top_error_px","right_error_px","bottom_error_px","edge_error_mean_px","edge_error_maximum_px","elapsed_ms","approved_bbox_json","predicted_bbox_json","parameters_json","error_type","error_message"]
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for result in ranked:
            for page in result["pages"]:
                errors=page.get("edge_errors",{})
                err=page.get("error",{})
                w.writerow({
                    "run_id":result.get("run_id",""),"parameter_set_id":result["parameter_set_id"],"profile":result.get("profile") or "","rank":result.get("rank",""),
                    "global_ordinal":page["global_ordinal"],"label":page["label"],"layout_type":page["layout_type"],"status":page["status"],"iou":page["iou"],
                    "left_error_px":errors.get("left"),"top_error_px":errors.get("top"),"right_error_px":errors.get("right"),"bottom_error_px":errors.get("bottom"),
                    "edge_error_mean_px":page.get("edge_error_mean_px"),"edge_error_maximum_px":page.get("edge_error_maximum_px"),"elapsed_ms":page.get("elapsed_ms"),
                    "approved_bbox_json":json.dumps(page.get("approved_bbox")),"predicted_bbox_json":json.dumps(page.get("predicted_bbox")),
                    "parameters_json":canonical_parameters(result["parameters"]),"error_type":err.get("type"),"error_message":err.get("message")})


def write_rankings(path: Path, ranked: list[dict[str, Any]]) -> None:
    fields=["rank","parameter_set_id","profile","mean_iou","minimum_iou","mean_edge_error_px","failure_count","elapsed_ms_total","parameters_json"]
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for r in ranked:
            s=r["summary"]; w.writerow({"rank":r["rank"],"parameter_set_id":r["parameter_set_id"],"profile":r.get("profile") or "","mean_iou":s["mean_iou"],"minimum_iou":s["minimum_iou"],"mean_edge_error_px":s["mean_edge_error_px"],"failure_count":s["failure_count"],"elapsed_ms_total":s["elapsed_ms_total"],"parameters_json":canonical_parameters(r["parameters"])})
