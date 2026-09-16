"""Shared conservative orientation/deskew primitives and production policy gates."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _analysis_gray(image: np.ndarray, maximum_dimension: int = 1400) -> np.ndarray:
    gray = _gray(image)
    scale = min(1.0, maximum_dimension / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray


def rotate_expand(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate without clipping, using a white expanded canvas and linear resampling."""
    if abs(angle) < 1e-12:
        return image.copy()
    height, width = image.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(float(matrix[0, 0]))
    sine = abs(float(matrix[0, 1]))
    target_width = max(1, int(math.ceil(height * sine + width * cosine)))
    target_height = max(1, int(math.ceil(height * cosine + width * sine)))
    matrix[0, 2] += (target_width - width) / 2.0
    matrix[1, 2] += (target_height - height) / 2.0
    value: int | tuple[int, ...] = 255 if image.ndim == 2 else tuple(255 for _ in range(image.shape[2]))
    return cv2.warpAffine(
        image,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=value,
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = float(np.sum(ordered_weights)) / 2.0
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left")])


def estimate_hough_lines(
    image: np.ndarray,
    maximum_degrees: float,
    deadband_degrees: float,
) -> dict[str, Any]:
    """Estimate baseline tilt from near-horizontal probabilistic Hough lines."""
    gray = _analysis_gray(image)
    edges = cv2.Canny(gray, 60, 180, apertureSize=3)
    minimum_length = max(30, int(gray.shape[1] * 0.08))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720.0,
        threshold=max(30, int(gray.shape[1] * 0.025)),
        minLineLength=minimum_length,
        maxLineGap=max(8, int(gray.shape[1] * 0.015)),
    )
    angles: list[float] = []
    lengths: list[float] = []
    if lines is not None:
        for line in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = (float(value) for value in line)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            while angle <= -90.0:
                angle += 180.0
            while angle > 90.0:
                angle -= 180.0
            if abs(angle) <= maximum_degrees:
                angles.append(angle)
                lengths.append(math.hypot(x2 - x1, y2 - y1))
    if not angles:
        return {
            "estimated_correction_degrees": 0.0,
            "applied_correction_degrees": 0.0,
            "confidence": 0.0,
            "line_count": 0,
            "weighted_mad_degrees": None,
            "boundary_limited": False,
        }
    angle_values = np.asarray(angles, dtype=np.float64)
    weights = np.asarray(lengths, dtype=np.float64)
    observed = _weighted_median(angle_values, weights)
    mad = _weighted_median(np.abs(angle_values - observed), weights)
    # Image-space y increases downward, while OpenCV's rotation argument uses
    # the opposite visual sign. Applying the image-space line angle levels it.
    correction = observed
    line_factor = min(1.0, len(angles) / 24.0)
    dispersion_factor = max(0.0, 1.0 - mad / max(maximum_degrees, 1e-9))
    confidence = line_factor * dispersion_factor
    boundary_limited = abs(correction) >= maximum_degrees - 0.05
    applied = 0.0 if abs(correction) < deadband_degrees else correction
    return {
        "estimated_correction_degrees": round(correction, 6),
        "applied_correction_degrees": round(applied, 6),
        "confidence": round(confidence, 6),
        "line_count": len(angles),
        "weighted_mad_degrees": round(mad, 6),
        "boundary_limited": boundary_limited,
    }


def evaluate_hough_policy(image: np.ndarray, policy: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a machine-readable conservative policy or return the pixels unchanged."""
    deskew = policy.get("deskew") or {}
    maximum = float(deskew["maximum_absolute_correction_degrees"])
    minimum = float(deskew["minimum_absolute_correction_degrees"])
    estimate = estimate_hough_lines(image, maximum, 0.0)
    angle = float(estimate["estimated_correction_degrees"])
    mad = estimate.get("weighted_mad_degrees")
    checks = {
        "minimum_correction": abs(angle) >= minimum,
        "maximum_correction": abs(angle) <= maximum,
        "confidence": float(estimate["confidence"]) >= float(deskew["minimum_confidence"]),
        "line_count": int(estimate["line_count"]) >= int(deskew["minimum_line_count"]),
        "weighted_mad": mad is not None and float(mad) <= float(deskew["maximum_weighted_mad_degrees"]),
        "not_boundary_limited": not bool(estimate["boundary_limited"]),
    }
    applied = all(checks.values())
    output = rotate_expand(image, angle) if applied else image.copy()
    reason = "all-safety-gates-passed" if applied else ",".join(name for name, passed in checks.items() if not passed)
    return output, {
        "estimator": "hough-lines",
        "decision": "apply" if applied else "preserve",
        "reason": reason,
        "estimated_correction_degrees": angle,
        "applied_correction_degrees": angle if applied else 0.0,
        "confidence": estimate["confidence"],
        "line_count": estimate["line_count"],
        "weighted_mad_degrees": mad,
        "boundary_limited": estimate["boundary_limited"],
        "safety_checks": checks,
    }
