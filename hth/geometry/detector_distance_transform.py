from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .model import Candidate

METHOD = "distance_transform"

BASELINE_PARAMETERS: dict[str, Any] = {
    "close_kernel_fraction": 0.008,
    "distance_threshold_fraction": 0.18,
    "minimum_core_area_fraction": 0.01,
    "minimum_component_core_overlap": 0.08,
    "minimum_bbox_area_fraction": 0.16,
    "minimum_rectangularity": 0.50,
    "bbox_padding_fraction": 0.0,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown Distance Transform parameters: {', '.join(unknown)}")
        values.update(overrides)
    for name in values:
        values[name] = float(values[name])
    if not 0.0 <= values["close_kernel_fraction"] <= 0.10:
        raise ValueError("close_kernel_fraction must be between 0 and 0.10")
    for name in (
        "distance_threshold_fraction", "minimum_core_area_fraction",
        "minimum_component_core_overlap", "minimum_bbox_area_fraction",
        "minimum_rectangularity", "bbox_padding_fraction",
    ):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if values["distance_threshold_fraction"] <= 0.0:
        raise ValueError("distance_threshold_fraction must be greater than 0")
    return values


def _binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Distance Transform expects a 2-D mask, got {mask.shape}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def detect(*, image_bgr: np.ndarray, mask: np.ndarray,
           parameters: dict[str, Any] | None = None) -> Candidate:
    """Use distance-transform interior support to select the dominant document region."""
    del image_bgr
    values = _parameters(parameters)
    working = _binary(mask)
    h, w = working.shape
    image_area = float(h * w)

    kernel_size = int(round(min(h, w) * values["close_kernel_fraction"]))
    if kernel_size > 0:
        kernel_size = max(3, kernel_size | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        working = cv2.morphologyEx(working, cv2.MORPH_CLOSE, kernel)

    if not np.any(working):
        return Candidate(
            METHOD, None, None, 0.0, 0.0,
            {"reason": "empty_foreground", "parameters": values},
            status="no_candidate",
        )

    distance = cv2.distanceTransform(working, cv2.DIST_L2, 5)
    maximum = float(distance.max())
    if maximum <= 0.0:
        return Candidate(
            METHOD, None, None, 0.0, 0.0,
            {"reason": "no_distance_core", "parameters": values},
            status="no_candidate",
        )

    core = distance >= (maximum * values["distance_threshold_fraction"])
    core_area_fraction = float(np.count_nonzero(core)) / image_area
    diagnostics = {
        "parameters": values,
        "maximum_distance_pixels": maximum,
        "core_area_fraction": core_area_fraction,
        "close_kernel_size": kernel_size,
        "evidence": "distance_transform_interior_core",
    }
    if core_area_fraction < values["minimum_core_area_fraction"]:
        diagnostics["reason"] = "core_too_small"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(working, connectivity=8)
    selected_labels: list[int] = []
    best_label = None
    best_overlap = -1.0
    for label in range(1, count):
        component = labels == label
        core_pixels = int(np.count_nonzero(core & component))
        component_pixels = int(stats[label, cv2.CC_STAT_AREA])
        overlap = core_pixels / max(component_pixels, 1)
        if core_pixels > 0 and overlap > best_overlap:
            best_overlap = overlap
            best_label = label
        if core_pixels > 0 and overlap >= values["minimum_component_core_overlap"]:
            selected_labels.append(label)

    if not selected_labels and best_label is not None:
        selected_labels = [best_label]
    if not selected_labels:
        diagnostics["reason"] = "no_core_supported_component"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    selected = np.isin(labels, selected_labels).astype(np.uint8) * 255
    points = cv2.findNonZero(selected)
    if points is None or len(points) < 4:
        diagnostics["reason"] = "insufficient_supported_geometry"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    hull = cv2.convexHull(points)
    hull_area = float(cv2.contourArea(hull))
    x, y, bw, bh = cv2.boundingRect(hull)
    bbox_area_fraction = float(bw * bh) / image_area
    rectangularity = min(1.0, hull_area / max(float(bw * bh), 1.0))
    diagnostics.update({
        "selected_component_count": len(selected_labels),
        "selected_labels": selected_labels,
        "bbox_area_fraction": bbox_area_fraction,
        "rectangularity": rectangularity,
    })
    if bbox_area_fraction < values["minimum_bbox_area_fraction"]:
        diagnostics["reason"] = "supported_region_too_small"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    if rectangularity < values["minimum_rectangularity"]:
        diagnostics["reason"] = "supported_region_not_rectangular"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    corners = _order_quad(cv2.boxPoints(cv2.minAreaRect(hull)))
    pad = int(round(min(h, w) * values["bbox_padding_fraction"]))
    bbox = [
        max(0, x - pad), max(0, y - pad),
        min(w, x + bw + pad), min(h, y + bh + pad),
    ]
    score = float(
        0.45 * rectangularity
        + 0.35 * min(1.0, bbox_area_fraction / max(values["minimum_bbox_area_fraction"], 1e-6))
        + 0.20 * min(1.0, core_area_fraction / max(values["minimum_core_area_fraction"], 1e-6))
    )
    return Candidate(
        METHOD, bbox, corners.astype(float).tolist(),
        score, score, diagnostics,
    )


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray,
                 parameters: dict[str, Any] | None = None,
                 candidate_corners: list[list[float]] | None = None,
                 verbose: bool = False) -> dict[str, np.ndarray]:
    del verbose
    values = _parameters(parameters)
    working = _binary(mask)
    distance = cv2.distanceTransform(working, cv2.DIST_L2, 5)
    maximum = float(distance.max())
    normalized = np.rint((distance / maximum) * 255.0).astype(np.uint8) if maximum > 0 else np.zeros_like(working)
    core = np.where(
        distance >= maximum * values["distance_threshold_fraction"], 255, 0
    ).astype(np.uint8) if maximum > 0 else np.zeros_like(working)
    overlay = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if candidate_corners is not None:
        corners = np.rint(np.asarray(candidate_corners, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [corners], True, (0, 0, 255), 3, cv2.LINE_AA)
    return {
        "distance-transform.png": normalized,
        "distance-core.png": core,
        "distance-candidate.png": overlay,
    }


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
