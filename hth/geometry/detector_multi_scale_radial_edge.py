from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .model import Candidate

METHOD = "multi_scale_radial_edge"

BASELINE_PARAMETERS: dict[str, Any] = {
    "base_sigma": 0.8,
    "scale_ratio": 2.0,
    "scale_count": 3,
    "ray_count": 144,
    "minimum_radius_fraction": 0.16,
    "maximum_radius_fraction": 0.78,
    "gradient_percentile": 82.0,
    "minimum_ray_support": 0.36,
    "minimum_area_fraction": 0.18,
    "maximum_area_fraction": 0.98,
    "bbox_padding_fraction": 0.0,
    "support_weight": 0.50,
    "strength_weight": 0.30,
    "area_weight": 0.20,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(
                f"Unknown Multi-Scale Radial Edge parameters: {', '.join(unknown)}"
            )
        values.update(overrides)
    values["scale_count"] = int(values["scale_count"])
    values["ray_count"] = int(values["ray_count"])
    for name in values:
        if name not in {"scale_count", "ray_count"}:
            values[name] = float(values[name])
    if values["base_sigma"] <= 0.0:
        raise ValueError("base_sigma must be positive")
    if values["scale_ratio"] <= 1.0:
        raise ValueError("scale_ratio must be greater than 1")
    if not 2 <= values["scale_count"] <= 5:
        raise ValueError("scale_count must be between 2 and 5")
    if values["ray_count"] < 16:
        raise ValueError("ray_count must be at least 16")
    if not 0.0 <= values["minimum_radius_fraction"] < values["maximum_radius_fraction"] <= 1.0:
        raise ValueError("radius fractions must satisfy 0 <= minimum < maximum <= 1")
    if not 0.0 <= values["gradient_percentile"] <= 100.0:
        raise ValueError("gradient_percentile must be between 0 and 100")
    for name in (
        "minimum_ray_support",
        "minimum_area_fraction",
        "maximum_area_fraction",
        "bbox_padding_fraction",
    ):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if values["minimum_area_fraction"] > values["maximum_area_fraction"]:
        raise ValueError("minimum_area_fraction must not exceed maximum_area_fraction")
    weights = [values["support_weight"], values["strength_weight"], values["area_weight"]]
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("score weights must be non-negative with at least one positive")
    return values


def _gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.ndim == 2:
        return image_bgr
    raise ValueError(f"Multi-Scale Radial Edge expects a 2-D or 3-D image, got {image_bgr.shape}")


def _order(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered[:, 0] + ordered[:, 1])), axis=0)


def _scale_space(gray: np.ndarray, values: dict[str, Any]) -> tuple[np.ndarray, list[float]]:
    normalized: list[np.ndarray] = []
    sigmas: list[float] = []
    for index in range(values["scale_count"]):
        sigma = values["base_sigma"] * (values["scale_ratio"] ** index)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        reference = float(np.percentile(magnitude, values["gradient_percentile"]))
        normalized.append(np.clip(magnitude / max(reference, 1e-6), 0.0, 2.0))
        sigmas.append(float(sigma))
    # Max fusion deliberately rewards a boundary that remains strong at any
    # physically useful blur scale instead of forcing all scales to agree.
    fused = np.max(np.stack(normalized, axis=0), axis=0).astype(np.float32)
    return fused, sigmas


def _search(image_bgr: np.ndarray, values: dict[str, Any]) -> dict[str, Any]:
    gray = _gray(image_bgr)
    height, width = gray.shape
    fused, sigmas = _scale_space(gray, values)
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
    max_radius = float(np.hypot(width, height) / 2.0)
    radii = np.arange(
        max(1.0, max_radius * values["minimum_radius_fraction"]),
        max_radius * values["maximum_radius_fraction"] + 1.0,
        1.0,
        dtype=np.float32,
    )
    threshold = float(np.percentile(fused, values["gradient_percentile"]))
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
        ray_values = fused[y[valid], x[valid]]
        if ray_values.size == 0:
            continue
        index = int(np.argmax(ray_values))
        strength = float(ray_values[index])
        if strength < threshold:
            continue
        points.append(samples[valid][index])
        strengths.append(strength)
    return {
        "fused": fused,
        "sigmas": sigmas,
        "points": np.asarray(points, dtype=np.float32),
        "support": len(points) / float(values["ray_count"]),
        "mean_strength": 0.0 if not strengths else float(np.mean(np.clip(strengths, 0.0, 2.0)) / 2.0),
    }


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    del mask
    values = _parameters(parameters)
    height, width = _gray(image_bgr).shape
    if min(height, width) < 16:
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "image_too_small", "parameters": values}, status="no_candidate")
    run = _search(image_bgr, values)
    points = run["points"]
    diagnostics: dict[str, Any] = {
        "parameters": values,
        "scale_sigmas": run["sigmas"],
        "supported_rays": int(len(points)),
        "ray_support": run["support"],
        "mean_fused_strength": run["mean_strength"],
        "evidence": "multi_scale_center_outward_gradient_search",
    }
    if len(points) < 4 or run["support"] < values["minimum_ray_support"]:
        diagnostics["reason"] = "insufficient_ray_support"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    corners = _order(cv2.boxPoints(cv2.minAreaRect(points.reshape(-1, 1, 2))))
    area = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2))))
    area_fraction = area / float(width * height)
    diagnostics["area_fraction"] = area_fraction
    if not values["minimum_area_fraction"] <= area_fraction <= values["maximum_area_fraction"]:
        diagnostics["reason"] = "implausible_area"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    padding = int(round(min(height, width) * values["bbox_padding_fraction"]))
    x, y, box_width, box_height = cv2.boundingRect(np.rint(corners).astype(np.int32).reshape(-1, 1, 2))
    bbox = [max(0, x - padding), max(0, y - padding), min(width, x + box_width + padding), min(height, y + box_height + padding)]
    total = values["support_weight"] + values["strength_weight"] + values["area_weight"]
    score = (
        run["support"] * values["support_weight"]
        + run["mean_strength"] * values["strength_weight"]
        + area_fraction * values["area_weight"]
    ) / total
    return Candidate(METHOD, bbox, corners.astype(float).tolist(), float(score), float(score), diagnostics)


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None, candidate_corners: list[list[float]] | None = None, verbose: bool = False) -> dict[str, np.ndarray]:
    del mask
    values = _parameters(parameters)
    run = _search(image_bgr, values)
    fused = cv2.normalize(run["fused"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    overlay = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    for point in np.rint(run["points"]).astype(np.int32):
        cv2.circle(overlay, tuple(point), 2, (0, 255, 255), -1)
    if candidate_corners is not None:
        cv2.polylines(overlay, [np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3, cv2.LINE_AA)
    images = {"multi-scale-gradient.png": fused, "multi-scale-radial-points.png": overlay}
    if verbose:
        scale_view = np.hstack([
            cv2.normalize(cv2.GaussianBlur(_gray(image_bgr), (0, 0), sigma), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            for sigma in run["sigmas"]
        ])
        images["scale-space.png"] = scale_view
    return images


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
