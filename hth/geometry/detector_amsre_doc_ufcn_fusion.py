from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import detector_adaptive_multi_scale_radial_edge, detector_doc_ufcn_page_mask
from .model import Candidate

METHOD = "amsre_doc_ufcn_fusion"

CHILD_CALIBRATIONS: dict[str, dict[str, Any]] = {
    "adaptive_multi_scale_radial_edge": {
        "parameter_set_id": "21ea516c3c5a",
        "parameters": {
            "coarse_angle_step_degrees": 2.0454545454545454,
            "refined_angle_step_degrees": 0.35,
            "weak_side_support_fraction": 0.65,
            "side_assignment_tolerance_fraction": 0.0075,
            "maximum_refined_sides": 3,
        },
    },
    "doc_ufcn_page_mask": {
        "parameter_set_id": "595002645fcc",
        "parameters": {
            "minimum_confidence": 0.5,
            "minimum_component_area_fraction": 0.0005,
            "minimum_page_area_fraction": 0.2,
            "page_padding_fraction": 0.01,
        },
    },
}

BASELINE_PARAMETERS: dict[str, Any] = {
    "amsre_rescue_score_ceiling": 0.85,
    "doc_ufcn_minimum_confidence": 0.75,
    "minimum_corner_disagreement_fraction": 0.01,
    "maximum_amsre_refined_support_fraction": 1.0,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, float]:
    values = dict(BASELINE_PARAMETERS)
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise ValueError(f"Unknown Fusion Gen3 parameters: {', '.join(unknown)}")
    values.update(overrides)
    values = {name: float(value) for name, value in values.items()}
    for name in ("amsre_rescue_score_ceiling", "doc_ufcn_minimum_confidence"):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if values["minimum_corner_disagreement_fraction"] < 0.0:
        raise ValueError("minimum_corner_disagreement_fraction must be non-negative")
    if not 0.0 <= values["maximum_amsre_refined_support_fraction"] <= 1.0:
        raise ValueError("maximum_amsre_refined_support_fraction must be between 0 and 1")
    return values


def _child_parameters(module, method: str) -> dict[str, Any]:
    values = dict(module.BASELINE_PARAMETERS)
    values.update(CHILD_CALIBRATIONS[method]["parameters"])
    return values


def _order(points: list[list[float]] | np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered[:, 0] + ordered[:, 1])), axis=0)


def _corner_disagreement_fraction(first: Candidate, second: Candidate, image_bgr: np.ndarray) -> float:
    if first.corners is None or second.corners is None:
        return 1.0
    a = _order(first.corners)
    b = _order(second.corners)
    diagonal = max(1.0, float(np.hypot(image_bgr.shape[1], image_bgr.shape[0])))
    return float(np.mean(np.linalg.norm(a - b, axis=1)) / diagonal)



def _bbox_geometry(first: Candidate, second: Candidate, image_bgr: np.ndarray) -> dict[str, float | None]:
    if first.bbox is None or second.bbox is None:
        return {
            "doc_to_amsre_area_ratio": None,
            "doc_to_amsre_width_ratio": None,
            "doc_to_amsre_height_ratio": None,
            "center_displacement_fraction": None,
            "bbox_iou": None,
        }
    ax1, ay1, ax2, ay2 = (float(value) for value in first.bbox)
    bx1, by1, bx2, by2 = (float(value) for value in second.bbox)
    aw, ah = max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)
    bw, bh = max(0.0, bx2 - bx1), max(0.0, by2 - by1)
    area_a, area_b = aw * ah, bw * bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area_a + area_b - intersection
    diagonal = max(1.0, float(np.hypot(image_bgr.shape[1], image_bgr.shape[0])))
    center_a = np.asarray(((ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0), dtype=np.float64)
    center_b = np.asarray(((bx1 + bx2) / 2.0, (by1 + by2) / 2.0), dtype=np.float64)
    return {
        "doc_to_amsre_area_ratio": float(area_b / area_a) if area_a > 0.0 else None,
        "doc_to_amsre_width_ratio": float(bw / aw) if aw > 0.0 else None,
        "doc_to_amsre_height_ratio": float(bh / ah) if ah > 0.0 else None,
        "center_displacement_fraction": float(np.linalg.norm(center_a - center_b) / diagonal),
        "bbox_iou": float(intersection / union) if union > 0.0 else None,
    }


def _amsre_refined_support_fraction(candidate: Candidate) -> float:
    diagnostics = candidate.diagnostics or {}
    total = diagnostics.get("total_supported_rays")
    refined = diagnostics.get("refined_supported_rays")
    refinement_triggered = diagnostics.get("refinement_triggered")
    if refinement_triggered is False and refined is None:
        refined = 0
    try:
        total_value = float(total)
        refined_value = float(refined)
    except (TypeError, ValueError):
        # Missing provenance should never make rescue easier. 1.0 preserves
        # the pre-refinement arbitration behavior while the explicit search
        # can tighten the gate only when AMSRE exposes measured support.
        return 1.0
    if total_value <= 0.0:
        return 1.0
    return float(np.clip(refined_value / total_value, 0.0, 1.0))

def _summary(candidate: Candidate, method: str) -> dict[str, Any]:
    return {
        "method": method,
        "parameter_set_id": CHILD_CALIBRATIONS[method]["parameter_set_id"],
        "status": candidate.status,
        "bbox": candidate.bbox,
        "corners": candidate.corners,
        "confidence": float(candidate.confidence),
        "score": float(candidate.score),
        "diagnostics": candidate.diagnostics,
    }


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    values = _parameters(parameters)
    amsre = detector_adaptive_multi_scale_radial_edge.detect(
        image_bgr=image_bgr,
        mask=mask,
        parameters=_child_parameters(detector_adaptive_multi_scale_radial_edge, "adaptive_multi_scale_radial_edge"),
    )
    doc = detector_doc_ufcn_page_mask.detect(
        image_bgr=image_bgr,
        mask=mask,
        parameters=_child_parameters(detector_doc_ufcn_page_mask, "doc_ufcn_page_mask"),
    )

    diagnostics: dict[str, Any] = {
        "parameters": values,
        "children": {
            "adaptive_multi_scale_radial_edge": _summary(amsre, "adaptive_multi_scale_radial_edge"),
            "doc_ufcn_page_mask": _summary(doc, "doc_ufcn_page_mask"),
        },
        "evidence": "calibrated_amsre_primary_with_doc_ufcn_confidence_gated_rescue",
    }

    if amsre.status != "ok" or amsre.corners is None:
        if doc.status == "ok" and doc.corners is not None:
            diagnostics.update({"decision": "doc-ufcn-rescue-amsre-unavailable", "selected_child": "doc_ufcn_page_mask"})
            return Candidate(METHOD, doc.bbox, doc.corners, doc.confidence, doc.score, diagnostics)
        diagnostics.update({"decision": "no-child-candidate", "selected_child": None})
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    if doc.status != "ok" or doc.corners is None:
        diagnostics.update({"decision": "amsre-doc-ufcn-unavailable", "selected_child": "adaptive_multi_scale_radial_edge"})
        return Candidate(METHOD, amsre.bbox, amsre.corners, amsre.confidence, amsre.score, diagnostics)

    disagreement = _corner_disagreement_fraction(amsre, doc, image_bgr)
    refined_support_fraction = _amsre_refined_support_fraction(amsre)
    doc_selected_confidence = float(doc.diagnostics.get("selected_confidence") or doc.confidence)
    geometry = _bbox_geometry(amsre, doc, image_bgr)
    rescue_gates = {
        "amsre_score_below_ceiling": float(amsre.score) <= values["amsre_rescue_score_ceiling"],
        "doc_ufcn_confidence_sufficient": doc_selected_confidence >= values["doc_ufcn_minimum_confidence"],
        "corner_disagreement_sufficient": disagreement >= values["minimum_corner_disagreement_fraction"],
        "amsre_refined_support_below_ceiling": refined_support_fraction <= values["maximum_amsre_refined_support_fraction"],
    }
    rescue = all(rescue_gates.values())
    diagnostics.update({
        "corner_disagreement_fraction": disagreement,
        "amsre_score": float(amsre.score),
        "amsre_refined_support_fraction": refined_support_fraction,
        "amsre_refinement_triggered": (amsre.diagnostics or {}).get("refinement_triggered"),
        "amsre_refined_supported_rays": (amsre.diagnostics or {}).get("refined_supported_rays"),
        "amsre_total_supported_rays": (amsre.diagnostics or {}).get("total_supported_rays"),
        "doc_ufcn_selected_confidence": doc_selected_confidence,
        "candidate_geometry": geometry,
        "rescue_gates": rescue_gates,
        "decision": "doc-ufcn-refined-support-gated-rescue" if rescue else "amsre-primary",
        "selected_child": "doc_ufcn_page_mask" if rescue else "adaptive_multi_scale_radial_edge",
    })
    selected = doc if rescue else amsre
    return Candidate(METHOD, selected.bbox, selected.corners, selected.confidence, selected.score, diagnostics)


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray, parameters=None, candidate_corners=None, verbose=False):
    del verbose
    candidate = detect(image_bgr=image_bgr, mask=mask, parameters=parameters)
    overlay = image_bgr.copy()
    children = candidate.diagnostics.get("children") or {}
    for method, color in (("adaptive_multi_scale_radial_edge", (0, 255, 255)), ("doc_ufcn_page_mask", (255, 255, 0))):
        corners = (children.get(method) or {}).get("corners")
        if corners is not None:
            cv2.polylines(overlay, [np.rint(np.asarray(corners)).astype(np.int32).reshape(-1, 1, 2)], True, color, 2, cv2.LINE_AA)
    selected = candidate_corners if candidate_corners is not None else candidate.corners
    if selected is not None:
        cv2.polylines(overlay, [np.rint(np.asarray(selected)).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3, cv2.LINE_AA)
    return {"fusion-gen3-child-arbitration.png": overlay}


__all__ = ["BASELINE_PARAMETERS", "CHILD_CALIBRATIONS", "METHOD", "debug_images", "detect"]
