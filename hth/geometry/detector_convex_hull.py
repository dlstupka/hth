from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .model import Candidate

METHOD = "convex_hull"

BASELINE_PARAMETERS: dict[str, Any] = {
    "minimum_fragment_area_fraction": 0.0005,
    "close_kernel_fraction": 0.008,
    "close_iterations": 1,
    "minimum_hull_area_fraction": 0.16,
    "minimum_solidity": 0.55,
    "polygon_epsilon_fraction": 0.025,
    "bbox_padding_fraction": 0.0,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown Convex Hull parameters: {', '.join(unknown)}")
        values.update(overrides)
    for name in (
        "minimum_fragment_area_fraction", "close_kernel_fraction",
        "minimum_hull_area_fraction", "minimum_solidity",
        "polygon_epsilon_fraction", "bbox_padding_fraction",
    ):
        values[name] = float(values[name])
    values["close_iterations"] = int(values["close_iterations"])
    if not 0.0 <= values["minimum_fragment_area_fraction"] <= 0.25:
        raise ValueError("minimum_fragment_area_fraction must be between 0 and 0.25")
    if not 0.0 <= values["close_kernel_fraction"] <= 0.10:
        raise ValueError("close_kernel_fraction must be between 0 and 0.10")
    if values["close_iterations"] < 0 or values["close_iterations"] > 5:
        raise ValueError("close_iterations must be between 0 and 5")
    if not 0.0 < values["minimum_hull_area_fraction"] <= 1.0:
        raise ValueError("minimum_hull_area_fraction must be greater than 0 and at most 1")
    if not 0.0 <= values["minimum_solidity"] <= 1.0:
        raise ValueError("minimum_solidity must be between 0 and 1")
    if not 0.0 < values["polygon_epsilon_fraction"] <= 0.25:
        raise ValueError("polygon_epsilon_fraction must be greater than 0 and at most 0.25")
    if not 0.0 <= values["bbox_padding_fraction"] <= 0.25:
        raise ValueError("bbox_padding_fraction must be between 0 and 0.25")
    return values


def _binary(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Convex Hull expects a 2-D mask, got {mask.shape}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def _pad_bbox(bbox: list[int], shape: tuple[int, int], fraction: float) -> list[int]:
    x1, y1, x2, y2 = bbox
    pad = int(round(min(shape) * fraction))
    return [
        max(0, x1 - pad), max(0, y1 - pad),
        min(shape[1], x2 + pad), min(shape[0], y2 + pad),
    ]


def detect(*, image_bgr: np.ndarray, mask: np.ndarray,
           parameters: dict[str, Any] | None = None) -> Candidate:
    """Fit a document envelope to the convex hull of substantial foreground fragments."""
    del image_bgr
    values = _parameters(parameters)
    working = _binary(mask)
    h, w = working.shape
    image_area = float(h * w)

    kernel_size = int(round(min(h, w) * values["close_kernel_fraction"]))
    if kernel_size > 0 and values["close_iterations"] > 0:
        kernel_size = max(3, kernel_size | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        working = cv2.morphologyEx(
            working, cv2.MORPH_CLOSE, kernel,
            iterations=values["close_iterations"],
        )

    contours, _ = cv2.findContours(working, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minimum_fragment_area = image_area * values["minimum_fragment_area_fraction"]
    substantial = [c for c in contours if cv2.contourArea(c) >= minimum_fragment_area]
    if not substantial:
        return Candidate(
            METHOD, None, None, 0.0, 0.0,
            {"reason": "no_substantial_fragments", "parameters": values},
            status="no_candidate",
        )

    points = np.concatenate(substantial, axis=0)
    hull = cv2.convexHull(points)
    hull_area = float(cv2.contourArea(hull))
    hull_area_fraction = hull_area / image_area if image_area else 0.0
    foreground_area = float(sum(cv2.contourArea(c) for c in substantial))
    solidity = min(1.0, foreground_area / hull_area) if hull_area > 0 else 0.0

    diagnostics = {
        "parameters": values,
        "fragment_count": len(substantial),
        "foreground_area_fraction": foreground_area / image_area if image_area else 0.0,
        "hull_area_fraction": hull_area_fraction,
        "solidity": solidity,
        "close_kernel_size": kernel_size,
        "evidence": "convex_hull_of_substantial_foreground_fragments",
    }
    if hull_area_fraction < values["minimum_hull_area_fraction"]:
        diagnostics["reason"] = "hull_too_small"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    if solidity < values["minimum_solidity"]:
        diagnostics["reason"] = "insufficient_solidity"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    perimeter = cv2.arcLength(hull, True)
    approximation = cv2.approxPolyDP(
        hull, values["polygon_epsilon_fraction"] * perimeter, True
    )
    if len(approximation) == 4 and cv2.isContourConvex(approximation):
        corners = _order_quad(approximation.reshape(4, 2))
        corner_source = "approx_poly_dp"
    else:
        box = cv2.boxPoints(cv2.minAreaRect(hull))
        corners = _order_quad(box)
        corner_source = "min_area_rect"

    x, y, bw, bh = cv2.boundingRect(hull)
    bbox = _pad_bbox([x, y, x + bw, y + bh], working.shape, values["bbox_padding_fraction"])
    rectangularity = min(1.0, hull_area / max(float(bw * bh), 1.0))
    score = float(
        0.45 * min(1.0, hull_area_fraction / max(values["minimum_hull_area_fraction"], 1e-6))
        + 0.30 * solidity
        + 0.25 * rectangularity
    )
    diagnostics.update({
        "corner_source": corner_source,
        "rectangularity": rectangularity,
        "hull_point_count": int(len(hull)),
    })
    return Candidate(
        METHOD, bbox, corners.astype(float).tolist(),
        score, score, diagnostics,
    )


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray,
                 parameters: dict[str, Any] | None = None,
                 candidate_corners: list[list[float]] | None = None,
                 verbose: bool = False) -> dict[str, np.ndarray]:
    del parameters, verbose
    working = _binary(mask)
    overlay = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(working, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull = cv2.convexHull(np.concatenate(contours, axis=0))
        cv2.polylines(overlay, [hull], True, (0, 255, 255), 2, cv2.LINE_AA)
    if candidate_corners is not None:
        corners = np.rint(np.asarray(candidate_corners, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [corners], True, (0, 0, 255), 3, cv2.LINE_AA)
    return {"convex-hull.png": overlay}


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
