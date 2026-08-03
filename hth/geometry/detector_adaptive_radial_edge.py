from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .model import Candidate
from . import detector_radial_edge

METHOD = "adaptive_radial_edge"

BASELINE_PARAMETERS: dict[str, Any] = {
    **detector_radial_edge.BASELINE_PARAMETERS,
    "coarse_angle_step_degrees": 3.0,
    "refined_angle_step_degrees": 1.0,
    "weak_side_support_fraction": 0.55,
    "side_assignment_tolerance_fraction": 0.025,
    "maximum_refined_sides": 2,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown Adaptive Radial Edge Search parameters: {', '.join(unknown)}")
        values.update(overrides)
    values["ray_count"] = int(values["ray_count"])
    values["maximum_refined_sides"] = int(values["maximum_refined_sides"])
    for name in values:
        if name not in {"ray_count", "maximum_refined_sides"}:
            values[name] = float(values[name])
    if values["coarse_angle_step_degrees"] <= 0 or values["refined_angle_step_degrees"] <= 0:
        raise ValueError("angle steps must be positive")
    if values["refined_angle_step_degrees"] >= values["coarse_angle_step_degrees"]:
        raise ValueError("refined angle step must be smaller than coarse angle step")
    if not 0.0 <= values["weak_side_support_fraction"] <= 1.0:
        raise ValueError("weak_side_support_fraction must be between 0 and 1")
    if values["side_assignment_tolerance_fraction"] <= 0.0:
        raise ValueError("side_assignment_tolerance_fraction must be positive")
    if not 1 <= values["maximum_refined_sides"] <= 4:
        raise ValueError("maximum_refined_sides must be between 1 and 4")
    # Reuse the base detector's validation for shared parameters.
    detector_radial_edge._parameters({k: values[k] for k in detector_radial_edge.BASELINE_PARAMETERS})
    return values


def _gradient(image_bgr: np.ndarray, values: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    gray = detector_radial_edge._gray(image_bgr)
    height, width = gray.shape
    working = cv2.GaussianBlur(gray, (0, 0), values["gaussian_sigma"]) if values["gaussian_sigma"] > 0 else gray
    gx = cv2.Sobel(working, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(working, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
    max_radius = float(np.hypot(width, height) / 2.0)
    radii = np.arange(max(1.0, max_radius * values["minimum_radius_fraction"]), max_radius * values["maximum_radius_fraction"] + 1.0, 1.0, dtype=np.float32)
    threshold = float(np.percentile(magnitude, values["gradient_percentile"]))
    return magnitude, center, radii, threshold, gray


def _sample_angles(magnitude: np.ndarray, center: np.ndarray, radii: np.ndarray, threshold: float, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = magnitude.shape
    points, strengths, used_angles = [], [], []
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        samples = center[None, :] + radii[:, None] * direction[None, :]
        x = np.rint(samples[:, 0]).astype(int)
        y = np.rint(samples[:, 1]).astype(int)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not np.any(valid):
            continue
        values_on_ray = magnitude[y[valid], x[valid]]
        index = int(np.argmax(values_on_ray))
        strength = float(values_on_ray[index])
        if strength < threshold:
            continue
        points.append(samples[valid][index])
        strengths.append(strength)
        used_angles.append(float(angle))
    return np.asarray(points, dtype=np.float32), np.asarray(strengths, dtype=np.float32), np.asarray(used_angles, dtype=np.float32)


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    denom = float(np.dot(segment, segment))
    if denom <= 1e-9:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * segment)))


def _side_support(points: np.ndarray, corners: np.ndarray, diagonal: float, tolerance_fraction: float) -> tuple[list[int], list[float]]:
    counts = [0, 0, 0, 0]
    tolerance = max(2.0, diagonal * tolerance_fraction)
    for point in points:
        distances = [_point_segment_distance(point, corners[i], corners[(i + 1) % 4]) for i in range(4)]
        side = int(np.argmin(distances))
        if distances[side] <= tolerance:
            counts[side] += 1
    maximum = max(counts) if counts else 0
    fractions = [count / maximum if maximum else 0.0 for count in counts]
    return counts, fractions


def _ray_intersection_side(center: np.ndarray, direction: np.ndarray, corners: np.ndarray) -> int | None:
    best_t, best_side = float("inf"), None
    for side in range(4):
        a, b = corners[side], corners[(side + 1) % 4]
        segment = b - a
        matrix = np.array([[direction[0], -segment[0]], [direction[1], -segment[1]]], dtype=np.float64)
        det = float(np.linalg.det(matrix))
        if abs(det) < 1e-9:
            continue
        t, u = np.linalg.solve(matrix, np.asarray(a - center, dtype=np.float64))
        if t > 0.0 and 0.0 <= u <= 1.0 and t < best_t:
            best_t, best_side = float(t), side
    return best_side


def _fit(points: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(points.reshape(-1, 1, 2))
    return detector_radial_edge._order(cv2.boxPoints(rect))


def _run(image_bgr: np.ndarray, values: dict[str, Any]) -> dict[str, Any]:
    magnitude, center, radii, threshold, gray = _gradient(image_bgr, values)
    coarse_step = np.deg2rad(values["coarse_angle_step_degrees"])
    coarse_angles = np.arange(0.0, 2.0 * np.pi, coarse_step, dtype=np.float32)
    coarse_points, coarse_strengths, coarse_used = _sample_angles(magnitude, center, radii, threshold, coarse_angles)
    result = {
        "magnitude": magnitude, "center": center, "threshold": threshold,
        "coarse_points": coarse_points, "coarse_angles": coarse_used,
        "refined_points": np.empty((0, 2), dtype=np.float32), "refined_angles": np.empty((0,), dtype=np.float32),
        "weak_sides": [], "side_counts": [0, 0, 0, 0], "side_support_fractions": [0.0] * 4,
        "all_points": coarse_points, "coarse_corners": None, "final_corners": None,
        "mean_strength": 0.0,
    }
    if len(coarse_points) < 4:
        return result
    coarse_corners = _fit(coarse_points)
    result["coarse_corners"] = coarse_corners
    diagonal = float(np.hypot(*gray.shape[::-1]))
    counts, fractions = _side_support(coarse_points, coarse_corners, diagonal, values["side_assignment_tolerance_fraction"])
    weak = [i for i, fraction in sorted(enumerate(fractions), key=lambda item: item[1]) if fraction < values["weak_side_support_fraction"]]
    weak = weak[: values["maximum_refined_sides"]]
    result["side_counts"], result["side_support_fractions"], result["weak_sides"] = counts, fractions, weak
    refined_strengths = np.empty((0,), dtype=np.float32)
    if weak:
        refine_step = np.deg2rad(values["refined_angle_step_degrees"])
        refine_angles = []
        coarse_keys = {round(float(a), 6) for a in coarse_used}
        for angle in np.arange(0.0, 2.0 * np.pi, refine_step, dtype=np.float32):
            if round(float(angle), 6) in coarse_keys:
                continue
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            if _ray_intersection_side(center, direction, coarse_corners) in weak:
                refine_angles.append(float(angle))
        if refine_angles:
            refined_points, refined_strengths, refined_used = _sample_angles(magnitude, center, radii, threshold, np.asarray(refine_angles, dtype=np.float32))
            result["refined_points"], result["refined_angles"] = refined_points, refined_used
            if len(refined_points):
                result["all_points"] = np.vstack([coarse_points, refined_points])
    if len(result["all_points"]) >= 4:
        result["final_corners"] = _fit(result["all_points"])
    strengths = np.concatenate([coarse_strengths, refined_strengths]) if len(refined_strengths) else coarse_strengths
    result["mean_strength"] = 0.0 if len(strengths) == 0 else float(np.mean(np.clip(strengths / max(threshold, 1e-6), 0.0, 2.0)) / 2.0)
    return result


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    del mask
    values = _parameters(parameters)
    height, width = detector_radial_edge._gray(image_bgr).shape
    if min(height, width) < 16:
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "image_too_small", "parameters": values}, status="no_candidate")
    run = _run(image_bgr, values)
    points = run["all_points"]
    total_requested = max(1, int(round(360.0 / values["coarse_angle_step_degrees"])))
    support = len(run["coarse_points"]) / float(total_requested)
    diagnostics: dict[str, Any] = {
        "parameters": values,
        "coarse_supported_rays": int(len(run["coarse_points"])),
        "refined_supported_rays": int(len(run["refined_points"])),
        "total_supported_rays": int(len(points)),
        "ray_support": support,
        "mean_edge_strength": run["mean_strength"],
        "side_support_counts": run["side_counts"],
        "side_support_fractions": run["side_support_fractions"],
        "weak_sides": run["weak_sides"],
        "refinement_triggered": bool(run["weak_sides"]),
        "evidence": "adaptive_two_pass_radial_gradient_search",
    }
    if len(points) < 4 or support < values["minimum_ray_support"]:
        diagnostics["reason"] = "insufficient_ray_support"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    corners = run["final_corners"]
    area = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2))))
    area_fraction = area / float(width * height)
    diagnostics["area_fraction"] = area_fraction
    if not values["minimum_area_fraction"] <= area_fraction <= values["maximum_area_fraction"]:
        diagnostics["reason"] = "implausible_area"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    x, y, box_width, box_height = cv2.boundingRect(np.rint(corners).astype(np.int32).reshape(-1, 1, 2))
    rectangularity = min(1.0, area / max(float(box_width * box_height), 1.0))
    diagnostics["rectangularity"] = rectangularity
    total = values["area_weight"] + values["support_weight"] + values["rectangularity_weight"]
    score = (area_fraction * values["area_weight"] + (0.5 * support + 0.5 * run["mean_strength"]) * values["support_weight"] + rectangularity * values["rectangularity_weight"]) / total
    return Candidate(METHOD, [int(x), int(y), int(x + box_width), int(y + box_height)], corners.astype(float).tolist(), float(score), float(score), diagnostics)


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None, candidate_corners: list[list[float]] | None = None, verbose: bool = False) -> dict[str, np.ndarray]:
    del mask
    values = _parameters(parameters)
    run = _run(image_bgr, values)
    base = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    gradient = cv2.normalize(run["magnitude"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    points_view = base.copy()
    for point in np.rint(run["coarse_points"]).astype(np.int32):
        cv2.circle(points_view, tuple(point), 2, (0, 255, 255), -1)
    for point in np.rint(run["refined_points"]).astype(np.int32):
        cv2.circle(points_view, tuple(point), 2, (255, 0, 255), -1)
    if candidate_corners is not None:
        cv2.polylines(points_view, [np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3, cv2.LINE_AA)
    images = {"adaptive-radial-gradient.png": gradient, "adaptive-radial-edge-points.png": points_view}
    if verbose:
        center = tuple(np.rint(run["center"]).astype(int))
        weak_view = base.copy()
        refined_view = base.copy()
        if run["coarse_corners"] is not None:
            corners = np.rint(run["coarse_corners"]).astype(np.int32)
            for side in range(4):
                color = (0, 0, 255) if side in run["weak_sides"] else (0, 180, 0)
                cv2.line(weak_view, tuple(corners[side]), tuple(corners[(side + 1) % 4]), color, 4, cv2.LINE_AA)
        for point in np.rint(run["coarse_points"]).astype(np.int32):
            cv2.line(refined_view, center, tuple(point), (120, 120, 120), 1, cv2.LINE_AA)
        for point in np.rint(run["refined_points"]).astype(np.int32):
            cv2.line(refined_view, center, tuple(point), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(refined_view, tuple(point), 3, (0, 255, 255), -1)
        comparison = base.copy()
        if run["coarse_corners"] is not None:
            cv2.polylines(comparison, [np.rint(run["coarse_corners"]).astype(np.int32).reshape(-1, 1, 2)], True, (255, 160, 0), 2, cv2.LINE_AA)
        if run["final_corners"] is not None:
            cv2.polylines(comparison, [np.rint(run["final_corners"]).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3, cv2.LINE_AA)
        images["weak-side-support.png"] = weak_view
        images["refined-rays.png"] = refined_view
        images["coarse-vs-refined-overlay.png"] = comparison
    return images


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
