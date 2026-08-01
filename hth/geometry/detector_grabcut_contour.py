from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from . import detector_contour_quad, detector_grabcut
from .model import Candidate

METHOD = "grabcut_contour"

BASELINE_PARAMETERS: dict[str, Any] = {
    "grabcut_border_fraction": 0.02,
    "grabcut_erosion_kernel_fraction": 0.015,
    "grabcut_erosion_iterations": 1,
    "grabcut_iterations": 3,
    "grabcut_close_kernel_fraction": 0.02,
    "grabcut_close_iterations": 1,
    "grabcut_minimum_bbox_area_fraction": 0.10,
    "grabcut_minimum_contour_area_fraction": 0.04,
    "grabcut_polygon_epsilon_fraction": 0.018,
    "contour_minimum_area_fraction": 0.12,
    "contour_epsilon_max_fraction": 0.04,
    "contour_minimum_rectangularity": 0.55,
    "minimum_agreement_iou": 0.15,
    "grabcut_weight": 0.55,
    "contour_weight": 0.20,
    "agreement_weight": 0.25,
    "require_contour": False,
}


def _parameters(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown GrabCut + Contour parameters: {', '.join(unknown)}")
        values.update(overrides)

    integer_names = {
        "grabcut_erosion_iterations",
        "grabcut_iterations",
        "grabcut_close_iterations",
    }
    boolean_names = {"require_contour"}
    for name in values:
        if name in integer_names:
            values[name] = int(values[name])
        elif name in boolean_names:
            values[name] = bool(values[name])
        else:
            values[name] = float(values[name])

    for name in (
        "grabcut_border_fraction",
        "grabcut_minimum_bbox_area_fraction",
        "grabcut_minimum_contour_area_fraction",
        "contour_minimum_area_fraction",
        "contour_minimum_rectangularity",
        "minimum_agreement_iou",
    ):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if values["grabcut_border_fraction"] >= 0.5:
        raise ValueError("grabcut_border_fraction must be less than 0.5")
    for name in (
        "grabcut_erosion_kernel_fraction",
        "grabcut_close_kernel_fraction",
    ):
        if values[name] < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in integer_names:
        if values[name] < 0:
            raise ValueError(f"{name} must be non-negative")
    if values["grabcut_iterations"] < 1:
        raise ValueError("grabcut_iterations must be at least 1")
    if not 0.0 < values["grabcut_polygon_epsilon_fraction"] <= 0.25:
        raise ValueError("grabcut_polygon_epsilon_fraction must be greater than 0 and at most 0.25")
    if not 0.0 < values["contour_epsilon_max_fraction"] <= 0.25:
        raise ValueError("contour_epsilon_max_fraction must be greater than 0 and at most 0.25")
    weights = [values["grabcut_weight"], values["contour_weight"], values["agreement_weight"]]
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("fusion weights must be non-negative with a positive sum")
    return values


def _polygon_mask(shape: tuple[int, int], corners: Any) -> np.ndarray:
    output = np.zeros(shape, dtype=np.uint8)
    if corners is None:
        return output
    points = np.rint(np.asarray(corners, dtype=np.float32).reshape(-1, 2)).astype(np.int32)
    if len(points) >= 3:
        cv2.fillPoly(output, [points], 255)
    return output


def _polygon_iou(shape: tuple[int, int], first: Any, second: Any) -> float:
    first_mask = _polygon_mask(shape, first) > 0
    second_mask = _polygon_mask(shape, second) > 0
    union = int(np.count_nonzero(first_mask | second_mask))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(first_mask & second_mask) / union)


def _parent_parameters(values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    grabcut = {
        "border_fraction": values["grabcut_border_fraction"],
        "erosion_kernel_fraction": values["grabcut_erosion_kernel_fraction"],
        "erosion_iterations": values["grabcut_erosion_iterations"],
        "grabcut_iterations": values["grabcut_iterations"],
        "close_kernel_fraction": values["grabcut_close_kernel_fraction"],
        "close_iterations": values["grabcut_close_iterations"],
        "minimum_bbox_area_fraction": values["grabcut_minimum_bbox_area_fraction"],
        "minimum_contour_area_fraction": values["grabcut_minimum_contour_area_fraction"],
        "polygon_epsilon_fraction": values["grabcut_polygon_epsilon_fraction"],
    }
    contour = {
        "minimum_contour_area_fraction": values["contour_minimum_area_fraction"],
        "epsilon_max_fraction": values["contour_epsilon_max_fraction"],
        "minimum_rectangularity": values["contour_minimum_rectangularity"],
    }
    return grabcut, contour


def detect(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
) -> Candidate:
    """Generate geometry from GrabCut, then validate/refine it with contours.

    This is deliberately directional and is not an alias for ``contour_grabcut``:
    the returned geometry is GrabCut-derived whenever GrabCut succeeds. Contour
    Quadrilateral supplies independent geometric validation and an optional
    fallback only.
    """
    values = _parameters(parameters)
    if mask.ndim != 2:
        raise ValueError(f"GrabCut + Contour detector expects a 2-D mask, got shape {mask.shape}")

    grabcut_parameters, contour_parameters = _parent_parameters(values)
    grabcut = detector_grabcut.detect(
        image_bgr=image_bgr, mask=mask, parameters=grabcut_parameters
    )
    contour = detector_contour_quad.detect(
        image_bgr=image_bgr, mask=mask, parameters=contour_parameters
    )

    grabcut_ok = grabcut.status == "ok" and grabcut.corners is not None
    contour_ok = contour.status == "ok" and contour.corners is not None
    diagnostics: dict[str, Any] = {
        "parameters": values,
        "grabcut_status": grabcut.status,
        "contour_status": contour.status,
        "grabcut_score": round(float(grabcut.score), 8),
        "contour_score": round(float(contour.score), 8),
        "grabcut_corners": grabcut.corners,
        "contour_corners": contour.corners,
        "grabcut_reason": (grabcut.diagnostics or {}).get("reason"),
        "contour_reason": (contour.diagnostics or {}).get("reason"),
        "grabcut_refined_foreground_fraction": (grabcut.diagnostics or {}).get("refined_foreground_fraction"),
        "grabcut_contour_area_fraction": (grabcut.diagnostics or {}).get("contour_area_fraction"),
    }

    if not grabcut_ok:
        if contour_ok and not values["require_contour"]:
            diagnostics.update({
                "reason": "contour_fallback_without_grabcut",
                "fusion_mode": "contour_fallback",
                "selected_geometry": "contour_quad",
                "agreement_iou": 0.0,
            })
            score = round(float(contour.score), 6)
            return Candidate(METHOD, contour.bbox, contour.corners, score, score, diagnostics)
        diagnostics["reason"] = "no_grabcut_hypothesis"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics)

    if not contour_ok:
        if values["require_contour"]:
            diagnostics["reason"] = "no_contour_validation"
            return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics)
        diagnostics.update({
            "reason": "grabcut_accepted_without_contour",
            "fusion_mode": "grabcut_without_contour",
            "selected_geometry": "grabcut",
            "agreement_iou": 0.0,
        })
        score = round(float(grabcut.score), 6)
        return Candidate(METHOD, grabcut.bbox, grabcut.corners, score, score, diagnostics)

    agreement = _polygon_iou(mask.shape, grabcut.corners, contour.corners)
    diagnostics["agreement_iou"] = round(agreement, 8)
    if agreement < values["minimum_agreement_iou"]:
        diagnostics["reason"] = "insufficient_grabcut_contour_agreement"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics)

    weight_total = values["grabcut_weight"] + values["contour_weight"] + values["agreement_weight"]
    score = (
        values["grabcut_weight"] * float(grabcut.score)
        + values["contour_weight"] * float(contour.score)
        + values["agreement_weight"] * agreement
    ) / weight_total
    diagnostics.update({
        "fusion_mode": "grabcut_generated_contour_validated",
        "selected_geometry": "grabcut",
        "reason": "accepted",
    })
    rounded = round(float(score), 6)
    return Candidate(METHOD, grabcut.bbox, grabcut.corners, rounded, rounded, diagnostics)


def debug_images(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
    candidate_corners: list[list[float]] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    del mask, parameters
    base = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    grabcut_view = base.copy()
    contour_view = base.copy()
    agreement_view = base.copy()
    selected_view = base.copy()
    details = dict(diagnostics or {})

    grabcut_corners = details.get("grabcut_corners")
    contour_corners = details.get("contour_corners")
    if grabcut_corners is not None:
        points = np.rint(np.asarray(grabcut_corners)).astype(np.int32).reshape(-1, 2)
        cv2.polylines(grabcut_view, [points], True, (255, 0, 255), 4, cv2.LINE_AA)
        cv2.polylines(agreement_view, [points], True, (255, 0, 255), 4, cv2.LINE_AA)
    if contour_corners is not None:
        points = np.rint(np.asarray(contour_corners)).astype(np.int32).reshape(-1, 2)
        cv2.polylines(contour_view, [points], True, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.polylines(agreement_view, [points], True, (0, 255, 255), 4, cv2.LINE_AA)
    if candidate_corners is not None:
        points = np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1, 2)
        cv2.polylines(selected_view, [points], True, (0, 0, 255), 4, cv2.LINE_AA)

    return {
        "grabcut-candidate.png": grabcut_view,
        "contour-candidate.png": contour_view,
        "agreement-overlay.png": agreement_view,
        "selected-quadrilateral.png": selected_view,
    }


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
