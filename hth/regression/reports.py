"""Canonical raw and derived regression reports."""
from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Any
from .parameter_space import canonical_parameters


def normalize_result_record(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize optional result fields at the execution/reporting boundary.

    Successful page evaluations may carry ``error: null`` after shard CSV
    reconstruction.  Downstream report writers expect mapping/list fields, so
    normalize them once before serialization while preserving real errors.
    """
    result["error"] = result.get("error") or {}
    result["warnings"] = result.get("warnings") or []
    result["metadata"] = result.get("metadata") or {}
    for page in result.get("pages") or []:
        page["error"] = page.get("error") or {}
        page["warnings"] = page.get("warnings") or []
        page["metadata"] = page.get("metadata") or {}
    return result


def ranking_key(result: dict[str, Any]) -> tuple[float, float, int, float]:
    s=result["summary"]; edge=s["mean_edge_error_px"]
    return (-float(s["mean_iou"]), -float(s["minimum_iou"]), int(s["failure_count"]), float(edge) if edge is not None else float("inf"))


def write_raw_results(path: Path, ranked: list[dict[str, Any]]) -> None:
    fields=["run_id","parameter_set_equivalence_family_id","parameter_set_equivalence_family_sha256","parameter_set_equivalence_family_size","parameter_set_id","parameter_identity_sha256","parameter_schema_version","parameter_grid_sha256","parameter_grid_ordinal","profile","rank","search_rank","requested_search_member","search_space_member","reference_roles_json","historic_reference_json","completion_index","completion_elapsed_seconds","search_fraction","global_ordinal","label","layout_type","status","iou","left_error_px","top_error_px","right_error_px","bottom_error_px","edge_error_mean_px","edge_error_maximum_px","elapsed_ms","approved_bbox_json","predicted_bbox_json","parameters_json","error_type","error_message"]
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for result in ranked:
            normalize_result_record(result)
            for page in result["pages"]:
                errors=page.get("edge_errors") or {}
                err=page.get("error") or {}
                w.writerow({
                    "run_id":result.get("run_id",""),"parameter_set_id":result["parameter_set_id"],"parameter_set_equivalence_family_id":result.get("parameter_set_equivalence_family_id",""),"parameter_set_equivalence_family_sha256":result.get("parameter_set_equivalence_family_sha256",""),"parameter_set_equivalence_family_size":result.get("parameter_set_equivalence_family_size",""),"parameter_identity_sha256":result.get("parameter_identity_sha256",""),"parameter_schema_version":result.get("parameter_schema_version",""),"parameter_grid_sha256":result.get("parameter_grid_sha256",""),"parameter_grid_ordinal":result.get("parameter_grid_ordinal",""),"profile":result.get("profile") or "","rank":result.get("rank",""),"search_rank":result.get("search_rank",""),"requested_search_member":1 if result.get("requested_search_member") else 0,"search_space_member":1 if result.get("search_space_member") else 0,"reference_roles_json":json.dumps(result.get("reference_roles") or []),"historic_reference_json":json.dumps(result.get("historic_reference") or {}),
                    "completion_index":(result.get("search_observation") or {}).get("completion_index", (result.get("search_observation") or {}).get("parameter_set_number", "")),
                    "completion_elapsed_seconds":(result.get("search_observation") or {}).get("elapsed_seconds", ""),
                    "search_fraction":(result.get("search_observation") or {}).get("search_fraction", ""),
                    "global_ordinal":page["global_ordinal"],"label":page["label"],"layout_type":page["layout_type"],"status":page["status"],"iou":page["iou"],
                    "left_error_px":errors.get("left"),"top_error_px":errors.get("top"),"right_error_px":errors.get("right"),"bottom_error_px":errors.get("bottom"),
                    "edge_error_mean_px":page.get("edge_error_mean_px"),"edge_error_maximum_px":page.get("edge_error_maximum_px"),"elapsed_ms":page.get("elapsed_ms"),
                    "approved_bbox_json":json.dumps(page.get("approved_bbox")),"predicted_bbox_json":json.dumps(page.get("predicted_bbox")),
                    "parameters_json":canonical_parameters(result["parameters"]),"error_type":err.get("type"),"error_message":err.get("message")})


def write_rankings(path: Path, ranked: list[dict[str, Any]]) -> None:
    fields=["rank","search_rank","requested_search_member","search_space_member","reference_roles_json","parameter_set_equivalence_family_id","parameter_set_equivalence_family_sha256","parameter_set_equivalence_family_size","parameter_set_id","parameter_identity_sha256","parameter_schema_version","parameter_grid_ordinal","profile","mean_iou","minimum_iou","mean_edge_error_px","mean_iou_success","failure_count","elapsed_ms_total","parameters_json"]
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for r in ranked:
            s=r["summary"]; w.writerow({"rank":r["rank"],"search_rank":r.get("search_rank",""),"requested_search_member":1 if r.get("requested_search_member") else 0,"search_space_member":1 if r.get("search_space_member") else 0,"reference_roles_json":json.dumps(r.get("reference_roles") or []),"parameter_set_id":r["parameter_set_id"],"parameter_set_equivalence_family_id":r.get("parameter_set_equivalence_family_id",""),"parameter_set_equivalence_family_sha256":r.get("parameter_set_equivalence_family_sha256",""),"parameter_set_equivalence_family_size":r.get("parameter_set_equivalence_family_size",""),"parameter_identity_sha256":r.get("parameter_identity_sha256",""),"parameter_schema_version":r.get("parameter_schema_version",""),"parameter_grid_ordinal":r.get("parameter_grid_ordinal",""),"profile":r.get("profile") or "","mean_iou":s["mean_iou"],"minimum_iou":s["minimum_iou"],"mean_edge_error_px":s["mean_edge_error_px"],"mean_iou_success":s.get("mean_iou_success",s["mean_iou"]),"failure_count":s["failure_count"],"elapsed_ms_total":s["elapsed_ms_total"],"parameters_json":canonical_parameters(r["parameters"])})
