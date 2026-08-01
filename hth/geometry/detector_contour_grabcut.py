from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from . import detector_contour_quad, detector_grabcut
from .model import Candidate

METHOD = "contour_grabcut"

BASELINE_PARAMETERS: dict[str, Any] = {
    "contour_minimum_area_fraction": 0.12,
    "contour_epsilon_max_fraction": 0.04,
    "contour_minimum_rectangularity": 0.55,
    "grabcut_border_fraction": 0.02,
    "grabcut_erosion_kernel_fraction": 0.015,
    "grabcut_iterations": 3,
    "minimum_agreement_iou": 0.30,
    "contour_weight": 0.45,
    "grabcut_weight": 0.25,
    "agreement_weight": 0.30,
    "require_grabcut": False,
}


def _parameters(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown Contour + GrabCut parameters: {', '.join(unknown)}")
        values.update(overrides)

    for name in (
        "contour_minimum_area_fraction",
        "contour_epsilon_max_fraction",
        "contour_minimum_rectangularity",
        "grabcut_border_fraction",
        "grabcut_erosion_kernel_fraction",
        "minimum_agreement_iou",
        "contour_weight",
        "grabcut_weight",
        "agreement_weight",
    ):
        values[name] = float(values[name])
    values["grabcut_iterations"] = int(values["grabcut_iterations"])
    values["require_grabcut"] = bool(values["require_grabcut"])

    for name in (
        "contour_minimum_area_fraction",
        "contour_minimum_rectangularity",
        "minimum_agreement_iou",
    ):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not 0.0 < values["contour_epsilon_max_fraction"] <= 0.25:
        raise ValueError("contour_epsilon_max_fraction must be greater than 0 and at most 0.25")
    if not 0.0 <= values["grabcut_border_fraction"] < 0.5:
        raise ValueError("grabcut_border_fraction must be between 0 and 0.5")
    if values["grabcut_erosion_kernel_fraction"] < 0.0:
        raise ValueError("grabcut_erosion_kernel_fraction must be non-negative")
    if values["grabcut_iterations"] < 1:
        raise ValueError("grabcut_iterations must be at least 1")
    weights = [values["contour_weight"], values["grabcut_weight"], values["agreement_weight"]]
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
    contour = {
        "minimum_contour_area_fraction": values["contour_minimum_area_fraction"],
        "epsilon_max_fraction": values["contour_epsilon_max_fraction"],
        "minimum_rectangularity": values["contour_minimum_rectangularity"],
    }
    grabcut = {
        "border_fraction": values["grabcut_border_fraction"],
        "erosion_kernel_fraction": values["grabcut_erosion_kernel_fraction"],
        "grabcut_iterations": values["grabcut_iterations"],
    }
    return contour, grabcut


def detect(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
) -> Candidate:
    values = _parameters(parameters)
    if mask.ndim != 2:
        raise ValueError(f"Contour + GrabCut detector expects a 2-D mask, got shape {mask.shape}")

    contour_parameters, grabcut_parameters = _parent_parameters(values)
    contour = detector_contour_quad.detect(
        image_bgr=image_bgr, mask=mask, parameters=contour_parameters
    )
    grabcut = detector_grabcut.detect(
        image_bgr=image_bgr, mask=mask, parameters=grabcut_parameters
    )

    contour_ok = contour.status == "ok" and contour.corners is not None
    grabcut_ok = grabcut.status == "ok" and grabcut.corners is not None
    diagnostics: dict[str, Any] = {
        "parameters": values,
        "contour_status": contour.status,
        "grabcut_status": grabcut.status,
        "contour_score": round(float(contour.score), 8),
        "grabcut_score": round(float(grabcut.score), 8),
        "contour_corners": contour.corners,
        "grabcut_corners": grabcut.corners,
        "contour_reason": (contour.diagnostics or {}).get("reason"),
        "grabcut_reason": (grabcut.diagnostics or {}).get("reason"),
    }

    if not contour_ok:
        diagnostics["reason"] = "no_contour_hypothesis"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics)
    if not grabcut_ok:
        if values["require_grabcut"]:
            diagnostics["reason"] = "no_grabcut_validation"
            return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics)
        diagnostics.update({
            "reason": "contour_fallback_without_grabcut",
            "agreement_iou": 0.0,
            "fusion_mode": "contour_fallback",
        })
        score = round(float(contour.score), 6)
        return Candidate(METHOD, contour.bbox, contour.corners, score, score, diagnostics)

    agreement = _polygon_iou(mask.shape, contour.corners, grabcut.corners)
    diagnostics["agreement_iou"] = round(agreement, 8)
    if agreement < values["minimum_agreement_iou"]:
        diagnostics["reason"] = "insufficient_contour_grabcut_agreement"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics)

    weight_total = values["contour_weight"] + values["grabcut_weight"] + values["agreement_weight"]
    score = (
        values["contour_weight"] * float(contour.score)
        + values["grabcut_weight"] * float(grabcut.score)
        + values["agreement_weight"] * agreement
    ) / weight_total
    diagnostics.update({
        "fusion_mode": "contour_generated_grabcut_validated",
        "selected_geometry": "contour_quad",
        "reason": "accepted",
    })
    rounded = round(float(score), 6)
    return Candidate(METHOD, contour.bbox, contour.corners, rounded, rounded, diagnostics)


def debug_images(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
    candidate_corners: list[list[float]] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    del parameters
    base = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    contour_view = base.copy()
    grabcut_view = base.copy()
    agreement = base.copy()
    selected = base.copy()
    details = dict(diagnostics or {})

    contour_corners = details.get("contour_corners")
    grabcut_corners = details.get("grabcut_corners")
    if contour_corners is not None:
        contour_points = np.rint(np.asarray(contour_corners)).astype(np.int32).reshape(-1, 2)
        cv2.polylines(contour_view, [contour_points], True, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.polylines(agreement, [contour_points], True, (0, 255, 255), 4, cv2.LINE_AA)
    if grabcut_corners is not None:
        grabcut_points = np.rint(np.asarray(grabcut_corners)).astype(np.int32).reshape(-1, 2)
        cv2.polylines(grabcut_view, [grabcut_points], True, (255, 0, 255), 4, cv2.LINE_AA)
        cv2.polylines(agreement, [grabcut_points], True, (255, 0, 255), 4, cv2.LINE_AA)
    if candidate_corners is not None:
        selected_points = np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1, 2)
        cv2.polylines(selected, [selected_points], True, (0, 0, 255), 4, cv2.LINE_AA)

    return {
        "contour-candidate.png": contour_view,
        "grabcut-candidate.png": grabcut_view,
        "agreement-overlay.png": agreement,
        "selected-quadrilateral.png": selected,
    }


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
