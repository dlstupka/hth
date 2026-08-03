from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .model import Candidate

METHOD = "radial_edge"

BASELINE_PARAMETERS: dict[str, Any] = {
    "gaussian_sigma": 1.2,
    "ray_count": 96,
    "minimum_radius_fraction": 0.18,
    "maximum_radius_fraction": 0.72,
    "gradient_percentile": 82.0,
    "minimum_ray_support": 0.45,
    "minimum_area_fraction": 0.18,
    "maximum_area_fraction": 0.98,
    "area_weight": 0.35,
    "support_weight": 0.45,
    "rectangularity_weight": 0.20,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown Radial Edge Search parameters: {', '.join(unknown)}")
        values.update(overrides)
    values["ray_count"] = int(values["ray_count"])
    for name in values:
        if name != "ray_count":
            values[name] = float(values[name])
    if values["ray_count"] < 16:
        raise ValueError("ray_count must be at least 16")
    if values["gaussian_sigma"] < 0.0:
        raise ValueError("gaussian_sigma must be non-negative")
    if not 0.0 <= values["minimum_radius_fraction"] < values["maximum_radius_fraction"] <= 1.0:
        raise ValueError("radius fractions must satisfy 0 <= minimum < maximum <= 1")
    if not 0.0 <= values["gradient_percentile"] <= 100.0:
        raise ValueError("gradient_percentile must be between 0 and 100")
    for name in ("minimum_ray_support", "minimum_area_fraction", "maximum_area_fraction"):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if values["minimum_area_fraction"] > values["maximum_area_fraction"]:
        raise ValueError("minimum_area_fraction must not exceed maximum_area_fraction")
    weights = [values["area_weight"], values["support_weight"], values["rectangularity_weight"]]
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("Radial-edge score weights must be non-negative with at least one positive")
    return values


def _gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.ndim == 2:
        return image_bgr
    raise ValueError(f"Radial Edge Search expects a 2-D or 3-D image, got {image_bgr.shape}")


def _order(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered[:, 0] + ordered[:, 1])), axis=0)


def _search(image_bgr: np.ndarray, values: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float]:
    gray = _gray(image_bgr)
    height, width = gray.shape
    working = cv2.GaussianBlur(gray, (0, 0), values["gaussian_sigma"]) if values["gaussian_sigma"] > 0 else gray
    gx = cv2.Sobel(working, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(working, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
    max_radius = float(np.hypot(width, height) / 2.0)
    radii = np.arange(max(1.0, max_radius * values["minimum_radius_fraction"]), max_radius * values["maximum_radius_fraction"] + 1.0, 1.0, dtype=np.float32)
    threshold = float(np.percentile(magnitude, values["gradient_percentile"]))
    points: list[np.ndarray] = []
    strengths: list[float] = []
    for angle in np.linspace(0.0, 2.0 * np.pi, values["ray_count"], endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        samples = center[None, :] + radii[:, None] * direction[None, :]
        x = np.rint(samples[:, 0]).astype(int)
        y = np.rint(samples[:, 1]).astype(int)
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not np.any(valid):
            continue
        values_on_ray = magnitude[y[valid], x[valid]]
        if values_on_ray.size == 0:
            continue
        index = int(np.argmax(values_on_ray))
        strength = float(values_on_ray[index])
        if strength < threshold:
            continue
        valid_samples = samples[valid]
        points.append(valid_samples[index])
        strengths.append(strength)
    support = len(points) / float(values["ray_count"])
    mean_strength = 0.0 if not strengths else float(np.mean(np.clip(np.asarray(strengths) / max(threshold, 1e-6), 0.0, 2.0)) / 2.0)
    return np.asarray(points, dtype=np.float32), magnitude, support, mean_strength


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    """Search outward from the image center for strong page-boundary transitions."""
    del mask
    values = _parameters(parameters)
    height, width = _gray(image_bgr).shape
    if min(height, width) < 16:
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "image_too_small", "parameters": values}, status="no_candidate")
    points, _magnitude, support, mean_strength = _search(image_bgr, values)
    diagnostics: dict[str, Any] = {
        "parameters": values,
        "supported_rays": int(len(points)),
        "ray_support": support,
        "mean_edge_strength": mean_strength,
        "evidence": "center_outward_radial_gradient_search",
    }
    if len(points) < 4 or support < values["minimum_ray_support"]:
        diagnostics["reason"] = "insufficient_ray_support"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    rect = cv2.minAreaRect(points.reshape(-1, 1, 2))
    corners = _order(cv2.boxPoints(rect))
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
    score = (area_fraction * values["area_weight"] + (0.5 * support + 0.5 * mean_strength) * values["support_weight"] + rectangularity * values["rectangularity_weight"]) / total
    return Candidate(METHOD, [int(x), int(y), int(x + box_width), int(y + box_height)], corners.astype(float).tolist(), float(score), float(score), diagnostics)


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None, candidate_corners: list[list[float]] | None = None, verbose: bool = False) -> dict[str, np.ndarray]:
    del mask
    values = _parameters(parameters)
    points, magnitude, _support, _strength = _search(image_bgr, values)
    gradient = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    overlay = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    for point in np.rint(points).astype(np.int32):
        cv2.circle(overlay, tuple(point), 2, (0, 255, 255), -1)
    if candidate_corners is not None:
        cv2.polylines(overlay, [np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3, cv2.LINE_AA)
    images = {"radial-gradient.png": gradient, "radial-edge-points.png": overlay}
    if verbose:
        height, width = _gray(image_bgr).shape
        center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
        max_radius = float(np.hypot(width, height) / 2.0)
        start_radius = max(1.0, max_radius * values["minimum_radius_fraction"])
        end_radius = max_radius * values["maximum_radius_fraction"]
        ray_view = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        accepted_view = ray_view.copy()
        accepted = {tuple(np.rint(point).astype(int)) for point in points}
        for angle in np.linspace(0.0, 2.0 * np.pi, values["ray_count"], endpoint=False):
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            start = tuple(np.rint(center + start_radius * direction).astype(int))
            end = tuple(np.rint(center + end_radius * direction).astype(int))
            cv2.line(ray_view, start, end, (170, 170, 170), 1, cv2.LINE_AA)
            cv2.line(accepted_view, start, end, (120, 120, 120), 1, cv2.LINE_AA)
        for point in np.rint(points).astype(np.int32):
            cv2.line(accepted_view, tuple(np.rint(center).astype(int)), tuple(point), (0, 200, 0), 1, cv2.LINE_AA)
            cv2.circle(accepted_view, tuple(point), 3, (0, 255, 255), -1)
        images["radial-search-rays.png"] = ray_view
        images["accepted-rays.png"] = accepted_view
    return images


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
