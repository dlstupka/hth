from __future__ import annotations

from itertools import product
from typing import Any

import cv2
import numpy as np

from . import detector_gradient_vote, detector_polar_boundary_vote, detector_radial_edge
from .model import Candidate

METHOD = "border_fusion_quad"
CHILDREN = (
    ("radial_edge", detector_radial_edge.detect),
    ("polar_boundary_vote", detector_polar_boundary_vote.detect),
    ("gradient_vote", detector_gradient_vote.detect),
)

BASELINE_PARAMETERS: dict[str, Any] = {
    "gradient_percentile": 82.0,
    "minimum_side_gradient_support": 0.16,
    "minimum_child_candidates": 2,
    "minimum_distinct_sources": 2,
    "minimum_child_confidence": 0.0,
    "minimum_area_fraction": 0.18,
    "maximum_area_fraction": 0.98,
    "bbox_padding_fraction": 0.0,
    "gradient_weight": 0.45,
    "source_confidence_weight": 0.25,
    "source_diversity_weight": 0.15,
    "area_weight": 0.15,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown Border Fusion Quad parameters: {', '.join(unknown)}")
        values.update(overrides)
    values["minimum_child_candidates"] = int(values["minimum_child_candidates"])
    values["minimum_distinct_sources"] = int(values["minimum_distinct_sources"])
    for name in values:
        if name not in {"minimum_child_candidates", "minimum_distinct_sources"}:
            values[name] = float(values[name])
    if not 1 <= values["minimum_child_candidates"] <= len(CHILDREN):
        raise ValueError("minimum_child_candidates must be between 1 and 3")
    if not 1 <= values["minimum_distinct_sources"] <= len(CHILDREN):
        raise ValueError("minimum_distinct_sources must be between 1 and 3")
    if values["minimum_distinct_sources"] > values["minimum_child_candidates"]:
        raise ValueError("minimum_distinct_sources must not exceed minimum_child_candidates")
    if not 0.0 <= values["gradient_percentile"] <= 100.0:
        raise ValueError("gradient_percentile must be between 0 and 100")
    for name in (
        "minimum_side_gradient_support",
        "minimum_child_confidence",
        "minimum_area_fraction",
        "maximum_area_fraction",
        "bbox_padding_fraction",
    ):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if values["minimum_area_fraction"] > values["maximum_area_fraction"]:
        raise ValueError("minimum_area_fraction must not exceed maximum_area_fraction")
    weights = [values["gradient_weight"], values["source_confidence_weight"], values["source_diversity_weight"], values["area_weight"]]
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("score weights must be non-negative with at least one positive")
    return values


def _gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.ndim == 2:
        return image_bgr
    raise ValueError(f"Border Fusion Quad expects a 2-D or 3-D image, got {image_bgr.shape}")


def _order(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered[:, 0] + ordered[:, 1])), axis=0)


def _line(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x1, y1 = map(float, first)
    x2, y2 = map(float, second)
    line = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=np.float64)
    norm = float(np.hypot(line[0], line[1]))
    return line / max(norm, 1e-9)


def _intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    point = np.cross(first, second)
    if abs(float(point[2])) < 1e-8:
        return None
    return (point[:2] / point[2]).astype(np.float32)


def _sides(corners: np.ndarray) -> list[np.ndarray]:
    # Ordered TL, TR, BR, BL -> top, right, bottom, left.
    return [_line(corners[0], corners[1]), _line(corners[1], corners[2]), _line(corners[2], corners[3]), _line(corners[3], corners[0])]


def _candidate_summary(candidate: Candidate) -> dict[str, Any]:
    return {
        "method": candidate.method,
        "status": candidate.status,
        "bbox": candidate.bbox,
        "corners": candidate.corners,
        "confidence": float(candidate.confidence),
        "score": float(candidate.score),
        "diagnostics": candidate.diagnostics,
    }


def _side_gradient_support(magnitude: np.ndarray, first: np.ndarray, second: np.ndarray, threshold: float) -> float:
    count = max(8, int(round(float(np.linalg.norm(second - first)))))
    xs = np.clip(np.rint(np.linspace(first[0], second[0], count)).astype(int), 0, magnitude.shape[1] - 1)
    ys = np.clip(np.rint(np.linspace(first[1], second[1], count)).astype(int), 0, magnitude.shape[0] - 1)
    return float(np.mean(np.clip(magnitude[ys, xs] / max(threshold, 1e-6), 0.0, 1.0)))


def _run(image_bgr: np.ndarray, mask: np.ndarray, values: dict[str, Any]) -> dict[str, Any]:
    gray = _gray(image_bgr)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    threshold = float(np.percentile(magnitude, values["gradient_percentile"]))
    child_results: list[dict[str, Any]] = []
    available: list[dict[str, Any]] = []
    for method, detector in CHILDREN:
        candidate = detector(image_bgr=image_bgr, mask=mask)
        summary = _candidate_summary(candidate)
        child_results.append(summary)
        if candidate.status != "ok" or candidate.corners is None or candidate.confidence < values["minimum_child_confidence"]:
            continue
        corners = _order(np.asarray(candidate.corners, dtype=np.float32))
        available.append({"method": method, "candidate": candidate, "corners": corners, "sides": _sides(corners)})
    result = {"magnitude": magnitude, "threshold": threshold, "children": child_results, "available": available, "best": None, "evaluated_combinations": 0}
    if len(available) < values["minimum_child_candidates"]:
        return result
    height, width = gray.shape
    best: dict[str, Any] | None = None
    for source_indices in product(range(len(available)), repeat=4):
        if len(set(source_indices)) < values["minimum_distinct_sources"]:
            continue
        top = available[source_indices[0]]["sides"][0]
        right = available[source_indices[1]]["sides"][1]
        bottom = available[source_indices[2]]["sides"][2]
        left = available[source_indices[3]]["sides"][3]
        intersections = [_intersection(left, top), _intersection(top, right), _intersection(right, bottom), _intersection(bottom, left)]
        if any(point is None for point in intersections):
            continue
        corners = _order(np.asarray(intersections, dtype=np.float32))
        if not np.all(np.isfinite(corners)):
            continue
        area = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2))))
        area_fraction = area / float(width * height)
        if not values["minimum_area_fraction"] <= area_fraction <= values["maximum_area_fraction"]:
            continue
        supports = [
            _side_gradient_support(magnitude, corners[index], corners[(index + 1) % 4], threshold)
            for index in range(4)
        ]
        minimum_support = min(supports)
        if minimum_support < values["minimum_side_gradient_support"]:
            continue
        source_confidence = float(np.mean([available[index]["candidate"].confidence for index in source_indices]))
        diversity = len(set(source_indices)) / float(len(available))
        mean_gradient = float(np.mean(supports))
        total = values["gradient_weight"] + values["source_confidence_weight"] + values["source_diversity_weight"] + values["area_weight"]
        score = (
            mean_gradient * values["gradient_weight"]
            + source_confidence * values["source_confidence_weight"]
            + diversity * values["source_diversity_weight"]
            + area_fraction * values["area_weight"]
        ) / total
        result["evaluated_combinations"] += 1
        if best is None or score > best["score"]:
            best = {
                "corners": corners,
                "score": float(score),
                "area_fraction": area_fraction,
                "side_gradient_support": supports,
                "source_indices": list(source_indices),
                "source_methods": [available[index]["method"] for index in source_indices],
                "source_confidence": source_confidence,
                "source_diversity": diversity,
            }
    result["best"] = best
    return result


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    values = _parameters(parameters)
    run = _run(image_bgr, mask, values)
    diagnostics: dict[str, Any] = {
        "parameters": values,
        "children": run["children"],
        "available_child_candidates": len(run["available"]),
        "evaluated_side_combinations": run["evaluated_combinations"],
        "evidence": "cross_detector_side_hypothesis_fusion_with_gradient_validation",
    }
    if len(run["available"]) < values["minimum_child_candidates"]:
        diagnostics["reason"] = "insufficient_child_candidates"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    best = run["best"]
    if best is None:
        diagnostics["reason"] = "no_valid_fused_quadrilateral"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    diagnostics.update({
        "selected_side_sources": best["source_methods"],
        "side_gradient_support": best["side_gradient_support"],
        "source_confidence": best["source_confidence"],
        "source_diversity": best["source_diversity"],
        "area_fraction": best["area_fraction"],
    })
    height, width = _gray(image_bgr).shape
    corners = best["corners"]
    padding = int(round(min(height, width) * values["bbox_padding_fraction"]))
    x1, y1 = np.floor(corners.min(axis=0)).astype(int)
    x2, y2 = np.ceil(corners.max(axis=0)).astype(int)
    bbox = [int(max(0, x1 - padding)), int(max(0, y1 - padding)), int(min(width, x2 + padding)), int(min(height, y2 + padding))]
    return Candidate(METHOD, bbox, corners.astype(float).tolist(), best["score"], best["score"], diagnostics)


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None, candidate_corners: list[list[float]] | None = None, verbose: bool = False) -> dict[str, np.ndarray]:
    values = _parameters(parameters)
    run = _run(image_bgr, mask, values)
    gradient = cv2.normalize(run["magnitude"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    child_view = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]
    for index, item in enumerate(run["available"]):
        cv2.polylines(child_view, [np.rint(item["corners"]).astype(np.int32).reshape(-1, 1, 2)], True, colors[index % len(colors)], 2, cv2.LINE_AA)
    selected_view = child_view.copy()
    if candidate_corners is not None:
        cv2.polylines(selected_view, [np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 4, cv2.LINE_AA)
    images = {"fusion-gradient.png": gradient, "fusion-child-quads.png": child_view, "fusion-selected-quad.png": selected_view}
    if verbose and run["best"] is not None:
        side_view = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        corners = run["best"]["corners"]
        for index in range(4):
            cv2.line(side_view, tuple(np.rint(corners[index]).astype(int)), tuple(np.rint(corners[(index + 1) % 4]).astype(int)), colors[run["best"]["source_indices"][index] % len(colors)], 4, cv2.LINE_AA)
        images["fusion-side-sources.png"] = side_view
    return images


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
