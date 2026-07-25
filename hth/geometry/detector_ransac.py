from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from skimage.measure import LineModelND, ransac

from .common import bbox_from_points, candidate_score, valid_bbox
from .model import Candidate

METHOD = "ransac"

BASELINE_PARAMETERS: dict[str, int | float] = {
    "scan_samples": 220,
    "minimum_scan_foreground_fraction": 0.0125,
    "residual_threshold_fraction": 0.008,
    "max_trials": 400,
    "minimum_mean_inlier_ratio": 0.45,
    "minimum_bbox_area_fraction": 0.18,
    "bbox_padding_fraction": 0.0,
}

_EDGE_NAMES = ("left", "right", "top", "bottom")


@dataclass
class _Analysis:
    candidate: Candidate
    points: dict[str, np.ndarray]
    fitted: dict[str, tuple[LineModelND, np.ndarray]]
    corners: np.ndarray | None


def _parameters(overrides: dict[str, Any] | None) -> dict[str, int | float]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown RANSAC parameters: {', '.join(unknown)}")
        values.update(overrides)

    integer_names = {"scan_samples", "max_trials"}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"RANSAC parameter {name!r} must be numeric")
        values[name] = int(value) if name in integer_names else float(value)

    if int(values["scan_samples"]) < 20:
        raise ValueError("scan_samples must be at least 20")
    if int(values["max_trials"]) < 10:
        raise ValueError("max_trials must be at least 10")
    for name in (
        "minimum_scan_foreground_fraction",
        "residual_threshold_fraction",
        "minimum_mean_inlier_ratio",
        "minimum_bbox_area_fraction",
    ):
        if not 0.0 < float(values[name]) <= 1.0:
            raise ValueError(f"{name} must be greater than 0 and at most 1")
    if not 0.0 <= float(values["bbox_padding_fraction"]) <= 0.25:
        raise ValueError("bbox_padding_fraction must be between 0 and 0.25")
    return values


def _scan_boundary_points(
    mask: np.ndarray,
    *,
    scan_samples: int,
    minimum_scan_foreground_fraction: float,
) -> dict[str, np.ndarray]:
    height, width = mask.shape
    row_step = max(1, int(round(height / scan_samples)))
    col_step = max(1, int(round(width / scan_samples)))
    minimum_row_pixels = max(2, int(round(width * minimum_scan_foreground_fraction)))
    minimum_col_pixels = max(2, int(round(height * minimum_scan_foreground_fraction)))
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    top: list[tuple[float, float]] = []
    bottom: list[tuple[float, float]] = []

    for y in range(0, height, row_step):
        xs = np.flatnonzero(mask[y] > 0)
        if len(xs) >= minimum_row_pixels:
            left.append((float(xs[0]), float(y)))
            right.append((float(xs[-1]), float(y)))

    for x in range(0, width, col_step):
        ys = np.flatnonzero(mask[:, x] > 0)
        if len(ys) >= minimum_col_pixels:
            top.append((float(x), float(ys[0])))
            bottom.append((float(x), float(ys[-1])))

    return {
        "left": np.asarray(left, dtype=float).reshape(-1, 2),
        "right": np.asarray(right, dtype=float).reshape(-1, 2),
        "top": np.asarray(top, dtype=float).reshape(-1, 2),
        "bottom": np.asarray(bottom, dtype=float).reshape(-1, 2),
    }


def _fit_line(
    points: np.ndarray,
    *,
    residual_threshold: float,
    max_trials: int,
) -> tuple[LineModelND, np.ndarray] | None:
    if len(points) < 6:
        return None
    try:
        model, inliers = ransac(
            points,
            LineModelND,
            min_samples=2,
            residual_threshold=residual_threshold,
            max_trials=max_trials,
            stop_probability=0.999,
            rng=42,
        )
    except Exception:
        return None
    if model is None or inliers is None or int(np.sum(inliers)) < 4:
        return None
    return model, inliers


def _intersection(a: LineModelND, b: LineModelND) -> np.ndarray | None:
    origin_a, direction_a = np.asarray(a.origin), np.asarray(a.direction)
    origin_b, direction_b = np.asarray(b.origin), np.asarray(b.direction)
    matrix = np.column_stack((direction_a, -direction_b))
    if abs(float(np.linalg.det(matrix))) < 1e-8:
        return None
    t, _ = np.linalg.solve(matrix, origin_b - origin_a)
    return origin_a + direction_a * t


def _padded_bbox(bbox: list[int], fraction: float, width: int, height: int) -> list[int]:
    padding = int(round(min(width, height) * fraction))
    return [
        max(0, bbox[0] - padding),
        max(0, bbox[1] - padding),
        min(width, bbox[2] + padding),
        min(height, bbox[3] + padding),
    ]


def _analyze(mask: np.ndarray, parameters: dict[str, Any] | None) -> _Analysis:
    values = _parameters(parameters)
    if mask.ndim != 2:
        raise ValueError(f"RANSAC detector expects a 2-D mask, got shape {mask.shape}")

    height, width = mask.shape
    points = _scan_boundary_points(
        mask,
        scan_samples=int(values["scan_samples"]),
        minimum_scan_foreground_fraction=float(values["minimum_scan_foreground_fraction"]),
    )
    residual_threshold = max(
        1.0, min(width, height) * float(values["residual_threshold_fraction"])
    )
    fitted: dict[str, tuple[LineModelND, np.ndarray]] = {}
    for name, edge_points in points.items():
        result = _fit_line(
            edge_points,
            residual_threshold=residual_threshold,
            max_trials=int(values["max_trials"]),
        )
        if result is not None:
            fitted[name] = result

    base_diagnostics: dict[str, Any] = {
        "parameters": values,
        "residual_threshold_px": round(residual_threshold, 3),
        "sample_counts": {name: int(len(edge_points)) for name, edge_points in points.items()},
        "fitted_edges": sorted(fitted),
        "inlier_counts": {
            name: int(np.sum(fitted[name][1])) for name in fitted
        },
        "inlier_ratios": {
            name: round(float(np.mean(fitted[name][1])), 6) for name in fitted
        },
    }

    if set(fitted) != set(_EDGE_NAMES):
        candidate = Candidate(
            METHOD, None, None, 0.0, 0.0,
            {**base_diagnostics, "reason": "insufficient_edge_models"},
        )
        return _Analysis(candidate, points, fitted, None)

    tl = _intersection(fitted["left"][0], fitted["top"][0])
    tr = _intersection(fitted["right"][0], fitted["top"][0])
    br = _intersection(fitted["right"][0], fitted["bottom"][0])
    bl = _intersection(fitted["left"][0], fitted["bottom"][0])
    if any(point is None for point in (tl, tr, br, bl)):
        candidate = Candidate(
            METHOD, None, None, 0.0, 0.0,
            {**base_diagnostics, "reason": "parallel_edge_models"},
        )
        return _Analysis(candidate, points, fitted, None)

    corners = np.asarray([tl, tr, br, bl], dtype=float)
    bbox = _padded_bbox(
        bbox_from_points(corners, width, height),
        float(values["bbox_padding_fraction"]),
        width,
        height,
    )
    if not valid_bbox(bbox):
        candidate = Candidate(
            METHOD, None, None, 0.0, 0.0,
            {**base_diagnostics, "reason": "invalid_ransac_envelope"},
        )
        return _Analysis(candidate, points, fitted, corners)

    bbox_area_fraction = (
        (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / max(1, width * height)
    )
    mean_inlier_ratio = float(
        np.mean([np.mean(fitted[name][1]) for name in _EDGE_NAMES])
    )
    base_diagnostics.update({
        "mean_inlier_ratio": round(mean_inlier_ratio, 6),
        "bbox_area_fraction": round(bbox_area_fraction, 6),
        "corners": [[round(float(x), 3), round(float(y), 3)] for x, y in corners],
    })

    if mean_inlier_ratio < float(values["minimum_mean_inlier_ratio"]):
        candidate = Candidate(
            METHOD, None, None, 0.0, 0.0,
            {**base_diagnostics, "reason": "insufficient_mean_inlier_ratio"},
        )
        return _Analysis(candidate, points, fitted, corners)
    if bbox_area_fraction < float(values["minimum_bbox_area_fraction"]):
        candidate = Candidate(
            METHOD, None, None, 0.0, 0.0,
            {**base_diagnostics, "reason": "ransac_envelope_too_small"},
        )
        return _Analysis(candidate, points, fitted, corners)

    mask_score = candidate_score(mask, bbox)
    area_score = min(1.0, bbox_area_fraction / 0.65)
    combined = 0.55 * mask_score + 0.35 * mean_inlier_ratio + 0.10 * area_score
    base_diagnostics.update({
        "mask_score": round(mask_score, 6),
        "area_score": round(area_score, 6),
    })
    candidate = Candidate(
        METHOD,
        bbox,
        corners.tolist(),
        round(combined, 6),
        round(combined, 6),
        base_diagnostics,
    )
    return _Analysis(candidate, points, fitted, corners)


def detect(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> Candidate:
    del image_bgr
    return _analyze(mask, parameters).candidate


def _draw_points(image: np.ndarray, points: dict[str, np.ndarray], *, inliers: dict[str, np.ndarray] | None = None) -> None:
    colors = {
        "left": (255, 96, 96),
        "right": (96, 255, 96),
        "top": (96, 96, 255),
        "bottom": (255, 255, 96),
    }
    for name, edge_points in points.items():
        selected = edge_points if inliers is None or name not in inliers else edge_points[inliers[name]]
        for x, y in selected:
            cv2.circle(image, (int(round(x)), int(round(y))), 2, colors[name], -1)


def _line_endpoints(model: LineModelND, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    origin = np.asarray(model.origin, dtype=float)
    direction = np.asarray(model.direction, dtype=float)
    span = float(max(width, height) * 2)
    first = origin - direction * span
    second = origin + direction * span
    return (
        (int(round(first[0])), int(round(first[1]))),
        (int(round(second[0])), int(round(second[1]))),
    )


def debug_images(
    *,
    mask: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Return research-useful RANSAC intermediate images in analysis order."""
    analysis = _analyze(mask, parameters)
    height, width = mask.shape
    base = cv2.cvtColor((mask > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

    samples = base.copy()
    _draw_points(samples, analysis.points)

    models = base.copy()
    _draw_points(models, analysis.points)
    line_colors = {
        "left": (255, 96, 96),
        "right": (96, 255, 96),
        "top": (96, 96, 255),
        "bottom": (255, 255, 96),
    }
    for name, (model, _) in analysis.fitted.items():
        cv2.line(models, *_line_endpoints(model, width, height), line_colors[name], 2, cv2.LINE_AA)

    inliers = base.copy()
    _draw_points(
        inliers,
        analysis.points,
        inliers={name: values[1] for name, values in analysis.fitted.items()},
    )

    quadrilateral = base.copy()
    if analysis.corners is not None:
        polygon = np.round(analysis.corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(quadrilateral, [polygon], True, (255, 255, 255), 3, cv2.LINE_AA)
    if valid_bbox(analysis.candidate.bbox):
        left, top, right, bottom = (int(value) for value in analysis.candidate.bbox)
        cv2.rectangle(quadrilateral, (left, top), (right, bottom), (0, 255, 255), 2)

    return {
        "boundary-samples.png": samples,
        "fitted-edge-models.png": models,
        "ransac-inliers.png": inliers,
        "candidate-quadrilateral.png": quadrilateral,
    }


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
