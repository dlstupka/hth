from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import detector_components
from .model import Candidate

METHOD = "contour_components"

BASELINE_PARAMETERS: dict[str, Any] = {
    "minimum_contour_area_fraction": 0.12,
    "close_kernel_fraction": 0.008,
    "close_iterations": 1,
    "epsilon_min_fraction": 0.008,
    "epsilon_max_fraction": 0.04,
    "epsilon_steps": 9,
    "minimum_rectangularity": 0.55,
    "component_minimum_area_fraction": 0.0015,
    "component_minimum_area_px": 25,
    "component_merge_area_ratio": 0.02,
    "component_merge_gap_fraction": 0.035,
    "component_minimum_bbox_area_fraction": 0.12,
    "component_minimum_selected_area_fraction": 0.04,
    "component_bbox_padding_fraction": 0.0,
    "component_close_fraction": 0.008,
    "component_dilate_fraction": 0.015,
    "minimum_component_score": 0.05,
    "area_weight": 0.25,
    "rectangularity_weight": 0.20,
    "angle_weight": 0.15,
    "component_weight": 0.40,
    "merge_fragmented_contours": True,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(
                f"Unknown Contour + Components parameters: {', '.join(unknown)}"
            )
        values.update(overrides)

    integer_names = {
        "close_iterations",
        "epsilon_steps",
        "component_minimum_area_px",
    }
    boolean_names = {"merge_fragmented_contours"}
    for name, value in list(values.items()):
        if name in boolean_names:
            values[name] = bool(value)
        elif name in integer_names:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Contour + Components parameter {name!r} must be numeric")
            values[name] = int(value)
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Contour + Components parameter {name!r} must be numeric")
            values[name] = float(value)

    if not 0.0 <= values["minimum_contour_area_fraction"] <= 1.0:
        raise ValueError("minimum_contour_area_fraction must be between 0 and 1")
    if not 0.0 <= values["close_kernel_fraction"] <= 0.25:
        raise ValueError("close_kernel_fraction must be between 0 and 0.25")
    if values["close_iterations"] < 0:
        raise ValueError("close_iterations must be non-negative")
    if not 0.0 < values["epsilon_min_fraction"] <= values["epsilon_max_fraction"] <= 0.25:
        raise ValueError("epsilon fractions must satisfy 0 < minimum <= maximum <= 0.25")
    if values["epsilon_steps"] < 1:
        raise ValueError("epsilon_steps must be at least 1")
    if not 0.0 <= values["minimum_rectangularity"] <= 1.0:
        raise ValueError("minimum_rectangularity must be between 0 and 1")
    if values["component_minimum_area_px"] < 1:
        raise ValueError("component_minimum_area_px must be at least 1")
    for name in (
        "component_minimum_area_fraction",
        "component_merge_area_ratio",
        "component_minimum_bbox_area_fraction",
        "component_minimum_selected_area_fraction",
        "minimum_component_score",
    ):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not 0.0 <= values["component_merge_gap_fraction"] <= 0.5:
        raise ValueError("component_merge_gap_fraction must be between 0 and 0.5")
    if not 0.0 <= values["component_bbox_padding_fraction"] <= 0.25:
        raise ValueError("component_bbox_padding_fraction must be between 0 and 0.25")
    for name in ("component_close_fraction", "component_dilate_fraction"):
        if not 0.0 <= values[name] <= 0.10:
            raise ValueError(f"{name} must be between 0 and 0.10")
    weights = [
        values[name]
        for name in ("area_weight", "rectangularity_weight", "angle_weight", "component_weight")
    ]
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("score weights must be non-negative with a positive sum")
    return values


def _component_parameters(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "minimum_component_area_fraction": values["component_minimum_area_fraction"],
        "minimum_component_area_px": values["component_minimum_area_px"],
        "merge_area_ratio": values["component_merge_area_ratio"],
        "merge_gap_fraction": values["component_merge_gap_fraction"],
        "minimum_bbox_area_fraction": values["component_minimum_bbox_area_fraction"],
        "minimum_selected_area_fraction": values["component_minimum_selected_area_fraction"],
        "bbox_padding_fraction": values["component_bbox_padding_fraction"],
        "morphology_close_fraction": values["component_close_fraction"],
        "morphology_dilate_fraction": values["component_dilate_fraction"],
    }


def _odd_kernel_size(fraction: float, width: int, height: int) -> int:
    if fraction <= 0.0:
        return 0
    size = max(3, int(round(min(width, height) * fraction)))
    return size if size % 2 else size + 1


def _order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


def _angle_score(corners: np.ndarray) -> float:
    scores: list[float] = []
    for index in range(4):
        previous = corners[(index - 1) % 4] - corners[index]
        following = corners[(index + 1) % 4] - corners[index]
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
        if denominator <= 0.0:
            return 0.0
        cosine = abs(float(np.dot(previous, following)) / denominator)
        scores.append(max(0.0, 1.0 - min(1.0, cosine)))
    return float(np.mean(scores))


def _component_context(
    *, image_bgr: np.ndarray, mask: np.ndarray, values: dict[str, Any]
) -> tuple[Candidate, np.ndarray, np.ndarray]:
    parameters = _component_parameters(values)
    component_candidate = detector_components.detect(
        image_bgr=image_bgr,
        mask=mask,
        parameters=parameters,
    )
    processed, _, _ = detector_components._morphology(mask, parameters)
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        (processed > 0).astype(np.uint8), connectivity=8
    )
    selected_labels = {
        int(label)
        for label in component_candidate.diagnostics.get("selected_component_labels", [])
    }
    selected_mask = np.zeros(mask.shape, dtype=np.uint8)
    for label in selected_labels:
        if 0 < label < count:
            selected_mask[labels == label] = 255
    return component_candidate, labels, selected_mask


def _component_evidence(
    corners: np.ndarray,
    component_candidate: Candidate,
    selected_mask: np.ndarray,
) -> tuple[float, dict[str, float]]:
    height, width = selected_mask.shape
    polygon_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(polygon_mask, np.rint(corners).astype(np.int32), 255)
    polygon_area = max(1, int(np.count_nonzero(polygon_mask)))
    selected_area = int(np.count_nonzero(selected_mask))
    contained_area = int(np.count_nonzero(cv2.bitwise_and(selected_mask, polygon_mask)))
    containment = contained_area / max(1, selected_area)
    component_density = contained_area / polygon_area
    density_score = min(1.0, component_density / 0.10)

    envelope_iou = 0.0
    spread_score = 0.0
    if component_candidate.bbox:
        left, top, right, bottom = [int(value) for value in component_candidate.bbox]
        envelope_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(envelope_mask, (left, top), (right, bottom), 255, -1)
        intersection = int(np.count_nonzero(cv2.bitwise_and(polygon_mask, envelope_mask)))
        union = int(np.count_nonzero(cv2.bitwise_or(polygon_mask, envelope_mask)))
        envelope_iou = intersection / max(1, union)
        envelope_area = max(1, (right - left) * (bottom - top))
        spread_score = min(1.0, envelope_area / polygon_area)

    score = float(
        0.40 * containment
        + 0.35 * envelope_iou
        + 0.15 * spread_score
        + 0.10 * density_score
    )
    return score, {
        "component_containment_score": float(containment),
        "component_envelope_iou": float(envelope_iou),
        "component_spread_score": float(spread_score),
        "component_density": float(component_density),
        "component_density_score": float(density_score),
        "selected_component_area_px": float(selected_area),
        "contained_component_area_px": float(contained_area),
    }


def detect(
    *, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None
) -> Candidate:
    values = _parameters(parameters)
    if mask.ndim != 2:
        raise ValueError(
            f"Contour + Components detector expects a 2-D mask, got shape {mask.shape}"
        )

    working = np.where(mask > 0, 255, 0).astype(np.uint8)
    height, width = working.shape
    image_area = float(width * height)
    close_size = _odd_kernel_size(values["close_kernel_fraction"], width, height)
    if close_size and values["close_iterations"]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        working = cv2.morphologyEx(
            working,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=values["close_iterations"],
        )

    component_candidate, _, selected_mask = _component_context(
        image_bgr=image_bgr,
        mask=mask,
        values=values,
    )
    contours, _ = cv2.findContours(working, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hypotheses: list[tuple[np.ndarray, str]] = [
        (contour, "external_contour") for contour in contours
    ]
    if values["merge_fragmented_contours"] and contours:
        points = np.concatenate(contours, axis=0)
        if len(points) >= 3:
            hypotheses.append((cv2.convexHull(points), "merged_convex_hull"))

    minimum_area = image_area * values["minimum_contour_area_fraction"]
    epsilons = np.linspace(
        values["epsilon_min_fraction"],
        values["epsilon_max_fraction"],
        values["epsilon_steps"],
    )
    weight_total = sum(
        values[name]
        for name in ("area_weight", "rectangularity_weight", "angle_weight", "component_weight")
    )
    best: dict[str, Any] | None = None
    quadrilateral_count = 0
    rejected_components = 0
    rejected_rectangularity = 0

    for contour, source in hypotheses:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < minimum_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        for epsilon_fraction in epsilons:
            approx = cv2.approxPolyDP(
                contour,
                float(epsilon_fraction) * perimeter,
                True,
            )
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quadrilateral_count += 1
            corners = _order_corners(approx.reshape(4, 2))
            quad_area = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2))))
            if quad_area <= 0.0:
                continue
            rectangularity = min(1.0, contour_area / quad_area)
            if rectangularity < values["minimum_rectangularity"]:
                rejected_rectangularity += 1
                continue
            component_score, component_metrics = _component_evidence(
                corners,
                component_candidate,
                selected_mask,
            )
            if component_score < values["minimum_component_score"]:
                rejected_components += 1
                continue
            area_score = min(1.0, quad_area / image_area)
            angle_score = _angle_score(corners)
            score = (
                values["area_weight"] * area_score
                + values["rectangularity_weight"] * rectangularity
                + values["angle_weight"] * angle_score
                + values["component_weight"] * component_score
            ) / weight_total
            if best is None or score > best["score"]:
                best = {
                    "corners": corners,
                    "score": float(score),
                    "source": source,
                    "epsilon_fraction": float(epsilon_fraction),
                    "contour_area": contour_area,
                    "quad_area": quad_area,
                    "area_score": area_score,
                    "rectangularity": rectangularity,
                    "angle_score": angle_score,
                    "component_score": component_score,
                    "component_metrics": component_metrics,
                }

    common_diagnostics = {
        "parameters": values,
        "external_contour_count": len(contours),
        "contour_hypothesis_count": len(hypotheses),
        "quadrilateral_count": quadrilateral_count,
        "rejected_rectangularity": rejected_rectangularity,
        "rejected_component_support": rejected_components,
        "component_detector_status": component_candidate.status,
        "component_detector_confidence": float(component_candidate.confidence),
        "component_detector_bbox": component_candidate.bbox,
        "component_detector_diagnostics": component_candidate.diagnostics,
    }
    if best is None:
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {**common_diagnostics, "reason": "no_supported_quadrilateral"},
            status="no_candidate",
        )

    corners = best["corners"]
    left = max(0, int(np.floor(np.min(corners[:, 0]))))
    top = max(0, int(np.floor(np.min(corners[:, 1]))))
    right = min(width, int(np.ceil(np.max(corners[:, 0]))))
    bottom = min(height, int(np.ceil(np.max(corners[:, 1]))))
    diagnostics = {
        **common_diagnostics,
        "source": best["source"],
        "epsilon_fraction": best["epsilon_fraction"],
        "contour_area": best["contour_area"],
        "quad_area": best["quad_area"],
        "area_score": best["area_score"],
        "rectangularity": best["rectangularity"],
        "angle_score": best["angle_score"],
        "component_score": best["component_score"],
        **best["component_metrics"],
    }
    return Candidate(
        METHOD,
        [left, top, right, bottom],
        corners.astype(float).tolist(),
        round(best["score"], 6),
        round(best["score"], 6),
        diagnostics,
    )


def debug_images(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: dict[str, Any] | None = None,
    candidate_corners: list[list[float]] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    values = _parameters(parameters)
    working = np.where(mask > 0, 255, 0).astype(np.uint8)
    contour_image = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(working, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 255), 2)

    component_candidate, labels, selected_mask = _component_context(
        image_bgr=image_bgr,
        mask=mask,
        values=values,
    )
    component_labels = detector_components._component_label_image(labels)
    selected_components = cv2.cvtColor(selected_mask, cv2.COLOR_GRAY2BGR)
    component_envelope = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if component_candidate.bbox:
        left, top, right, bottom = [int(value) for value in component_candidate.bbox]
        cv2.rectangle(component_envelope, (left, top), (right, bottom), (255, 0, 255), 3)

    evidence = component_envelope.copy()
    selected = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if candidate_corners is not None:
        corners = np.asarray(candidate_corners, dtype=np.float32).reshape(4, 2)
        polyline = np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(evidence, [polyline], True, (0, 255, 255), 4, cv2.LINE_AA)
        cv2.polylines(selected, [polyline], True, (0, 0, 255), 4, cv2.LINE_AA)
    if diagnostics and isinstance(diagnostics.get("component_detector_bbox"), list):
        left, top, right, bottom = [int(value) for value in diagnostics["component_detector_bbox"]]
        cv2.rectangle(evidence, (left, top), (right, bottom), (255, 0, 255), 3)

    return {
        "contour-hypotheses.png": contour_image,
        "component-labels.png": component_labels,
        "selected-components.png": selected_components,
        "component-envelope.png": component_envelope,
        "component-evidence.png": evidence,
        "selected-quadrilateral.png": selected,
    }


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
