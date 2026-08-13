from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .model import Candidate

METHOD = "projective_gradient_vote"

BASELINE_PARAMETERS: dict[str, Any] = {
    "gaussian_sigma": 1.2,
    "gradient_percentile": 82.0,
    "minimum_segment_fraction": 0.16,
    "angle_bin_degrees": 4.0,
    "family_tolerance_degrees": 16.0,
    "orthogonality_tolerance_degrees": 22.0,
    "minimum_side_support": 0.18,
    "minimum_area_fraction": 0.18,
    "maximum_area_fraction": 0.98,
    "maximum_corner_overshoot_fraction": 0.08,
    "bbox_padding_fraction": 0.0,
    "support_weight": 0.55,
    "geometry_weight": 0.30,
    "area_weight": 0.15,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown Projective Gradient Vote parameters: {', '.join(unknown)}")
        values.update(overrides)
    for name in values:
        values[name] = float(values[name])
    if values["gaussian_sigma"] < 0.0:
        raise ValueError("gaussian_sigma must be non-negative")
    if not 0.0 <= values["gradient_percentile"] <= 100.0:
        raise ValueError("gradient_percentile must be between 0 and 100")
    if not 0.0 < values["minimum_segment_fraction"] <= 1.0:
        raise ValueError("minimum_segment_fraction must be between 0 and 1")
    if not 1.0 <= values["angle_bin_degrees"] <= 30.0:
        raise ValueError("angle_bin_degrees must be between 1 and 30")
    for name in ("family_tolerance_degrees", "orthogonality_tolerance_degrees"):
        if not 1.0 <= values[name] <= 45.0:
            raise ValueError(f"{name} must be between 1 and 45")
    for name in (
        "minimum_side_support",
        "minimum_area_fraction",
        "maximum_area_fraction",
        "maximum_corner_overshoot_fraction",
        "bbox_padding_fraction",
    ):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if values["minimum_area_fraction"] > values["maximum_area_fraction"]:
        raise ValueError("minimum_area_fraction must not exceed maximum_area_fraction")
    weights = [values["support_weight"], values["geometry_weight"], values["area_weight"]]
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("score weights must be non-negative with at least one positive")
    return values


def _gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.ndim == 2:
        return image_bgr
    raise ValueError(f"Projective Gradient Vote expects a 2-D or 3-D image, got {image_bgr.shape}")


def _angle_distance(first: float, second: float) -> float:
    delta = abs(first - second) % np.pi
    return float(min(delta, np.pi - delta))


def _line_from_segment(segment: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = map(float, segment)
    line = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=np.float64)
    norm = float(np.hypot(line[0], line[1]))
    return line / max(norm, 1e-9)


def _intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    point = np.cross(first, second)
    if abs(float(point[2])) < 1e-8:
        return None
    return (point[:2] / point[2]).astype(np.float32)


def _order(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered[:, 0] + ordered[:, 1])), axis=0)


def _segment_support(magnitude: np.ndarray, segment: np.ndarray, threshold: float) -> float:
    x1, y1, x2, y2 = map(float, segment)
    count = max(8, int(round(np.hypot(x2 - x1, y2 - y1))))
    xs = np.clip(np.rint(np.linspace(x1, x2, count)).astype(int), 0, magnitude.shape[1] - 1)
    ys = np.clip(np.rint(np.linspace(y1, y2, count)).astype(int), 0, magnitude.shape[0] - 1)
    return float(np.mean(np.clip(magnitude[ys, xs] / max(threshold, 1e-6), 0.0, 1.0)))


def _evidence(image_bgr: np.ndarray, values: dict[str, Any]) -> dict[str, Any]:
    gray = _gray(image_bgr)
    working = cv2.GaussianBlur(gray, (0, 0), values["gaussian_sigma"]) if values["gaussian_sigma"] > 0 else gray
    gx = cv2.Sobel(working, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(working, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    threshold = float(np.percentile(magnitude, values["gradient_percentile"]))
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(working)[0]
    height, width = gray.shape
    minimum_length = min(height, width) * values["minimum_segment_fraction"]
    segments: list[dict[str, Any]] = []
    if detected is not None:
        for row in detected.reshape(-1, 4):
            length = float(np.hypot(row[2] - row[0], row[3] - row[1]))
            if length < minimum_length:
                continue
            angle = float(np.arctan2(row[3] - row[1], row[2] - row[0]) % np.pi)
            support = _segment_support(magnitude, row, threshold)
            segments.append({"xy": row.astype(np.float32), "length": length, "angle": angle, "support": support, "weight": length * max(support, 1e-3)})
    return {"gray": gray, "magnitude": magnitude, "threshold": threshold, "segments": segments}


def _orientation_families(segments: list[dict[str, Any]], values: dict[str, Any]) -> tuple[float, float] | None:
    if len(segments) < 4:
        return None
    bin_width = np.deg2rad(values["angle_bin_degrees"])
    bins = max(6, int(np.ceil(np.pi / bin_width)))
    histogram = np.zeros(bins, dtype=np.float64)
    for segment in segments:
        index = min(bins - 1, int(segment["angle"] / np.pi * bins))
        histogram[index] += segment["weight"]
    centers = (np.arange(bins) + 0.5) * np.pi / bins
    first_index = int(np.argmax(histogram))
    first = float(centers[first_index])
    target = (first + np.pi / 2.0) % np.pi
    tolerance = np.deg2rad(values["orthogonality_tolerance_degrees"])
    candidates = [i for i, angle in enumerate(centers) if _angle_distance(float(angle), target) <= tolerance]
    if not candidates:
        return None
    second_index = max(candidates, key=lambda index: histogram[index])
    if histogram[first_index] <= 0.0 or histogram[second_index] <= 0.0:
        return None
    return first, float(centers[second_index])


def _select_sides(segments: list[dict[str, Any]], orientation: float, values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    tolerance = np.deg2rad(values["family_tolerance_degrees"])
    normal = np.array([-np.sin(orientation), np.cos(orientation)], dtype=np.float64)
    family = []
    for segment in segments:
        if _angle_distance(segment["angle"], orientation) > tolerance:
            continue
        midpoint = 0.5 * (segment["xy"][:2] + segment["xy"][2:])
        item = dict(segment)
        item["offset"] = float(np.dot(midpoint, normal))
        family.append(item)
    if len(family) < 2:
        return None
    family.sort(key=lambda item: item["offset"])
    span = max(1, int(np.ceil(len(family) * 0.30)))
    low = max(family[:span], key=lambda item: item["weight"])
    high = max(family[-span:], key=lambda item: item["weight"])
    if low is high:
        return None
    return low, high


def _projective_candidate(image_bgr: np.ndarray, values: dict[str, Any]) -> dict[str, Any]:
    evidence = _evidence(image_bgr, values)
    segments = evidence["segments"]
    families = _orientation_families(segments, values)
    result = {**evidence, "families": families, "selected": [], "corners": None}
    if families is None:
        return result
    first = _select_sides(segments, families[0], values)
    second = _select_sides(segments, families[1], values)
    if first is None or second is None:
        return result
    a0, a1 = (_line_from_segment(item["xy"]) for item in first)
    b0, b1 = (_line_from_segment(item["xy"]) for item in second)
    intersections = [_intersection(a0, b0), _intersection(a1, b0), _intersection(a1, b1), _intersection(a0, b1)]
    if any(point is None for point in intersections):
        return result
    result["selected"] = [first[0], first[1], second[0], second[1]]
    result["corners"] = _order(np.asarray(intersections, dtype=np.float32))
    return result


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    del mask
    values = _parameters(parameters)
    run = _projective_candidate(image_bgr, values)
    height, width = run["gray"].shape
    diagnostics: dict[str, Any] = {
        "parameters": values,
        "segment_count": len(run["segments"]),
        "orientation_families_degrees": None if run["families"] is None else [float(np.rad2deg(angle)) for angle in run["families"]],
        "evidence": "long_segment_gradient_votes_with_projective_line_intersections",
    }
    corners = run["corners"]
    if corners is None:
        diagnostics["reason"] = "insufficient_projective_line_families"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    margin = min(height, width) * values["maximum_corner_overshoot_fraction"]
    if np.any(corners[:, 0] < -margin) or np.any(corners[:, 0] > width - 1 + margin) or np.any(corners[:, 1] < -margin) or np.any(corners[:, 1] > height - 1 + margin):
        diagnostics["reason"] = "corner_overshoot"
        diagnostics["corners"] = corners.astype(float).tolist()
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    area = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2))))
    area_fraction = area / float(width * height)
    diagnostics["area_fraction"] = area_fraction
    if not values["minimum_area_fraction"] <= area_fraction <= values["maximum_area_fraction"]:
        diagnostics["reason"] = "implausible_area"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    side_support = [float(item["support"]) for item in run["selected"]]
    diagnostics["side_support"] = side_support
    if min(side_support) < values["minimum_side_support"]:
        diagnostics["reason"] = "insufficient_side_support"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    first, second = run["families"]
    orth_error = abs(_angle_distance(first, second) - np.pi / 2.0)
    geometry = max(0.0, 1.0 - orth_error / max(np.deg2rad(values["orthogonality_tolerance_degrees"]), 1e-9))
    mean_support = float(np.mean(side_support))
    diagnostics["projective_geometry_score"] = float(geometry)
    total = values["support_weight"] + values["geometry_weight"] + values["area_weight"]
    score = (mean_support * values["support_weight"] + geometry * values["geometry_weight"] + area_fraction * values["area_weight"]) / total
    padding = int(round(min(height, width) * values["bbox_padding_fraction"]))
    x1, y1 = np.floor(corners.min(axis=0)).astype(int)
    x2, y2 = np.ceil(corners.max(axis=0)).astype(int)
    bbox = [int(max(0, x1 - padding)), int(max(0, y1 - padding)), int(min(width, x2 + padding)), int(min(height, y2 + padding))]
    return Candidate(METHOD, bbox, corners.astype(float).tolist(), float(score), float(score), diagnostics)


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None, candidate_corners: list[list[float]] | None = None, verbose: bool = False) -> dict[str, np.ndarray]:
    del mask
    values = _parameters(parameters)
    run = _projective_candidate(image_bgr, values)
    magnitude = cv2.normalize(run["magnitude"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    segments_view = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    for item in run["segments"]:
        x1, y1, x2, y2 = np.rint(item["xy"]).astype(int)
        cv2.line(segments_view, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)
    selected_view = segments_view.copy()
    for item in run["selected"]:
        x1, y1, x2, y2 = np.rint(item["xy"]).astype(int)
        cv2.line(selected_view, (x1, y1), (x2, y2), (255, 0, 255), 3, cv2.LINE_AA)
    if candidate_corners is not None:
        cv2.polylines(selected_view, [np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3, cv2.LINE_AA)
    images = {"projective-gradient.png": magnitude, "projective-line-votes.png": selected_view}
    if verbose:
        images["projective-segments.png"] = segments_view
    return images


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
