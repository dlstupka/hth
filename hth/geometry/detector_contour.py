from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .common import candidate_score
from .model import Candidate

METHOD = "contour"

BASELINE_PARAMETERS: dict[str, Any] = {
    "minimum_contour_area_fraction": 0.12,
    "polygon_epsilon_fraction": 0.018,
    "close_kernel_fraction": 0.0,
    "close_iterations": 0,
    "rectangularity_weight": 0.25,
    "bbox_padding_fraction": 0.0,
    "merge_fragmented_contours": False,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown Contour parameters: {', '.join(unknown)}")
        values.update(overrides)

    minimum_area = float(values["minimum_contour_area_fraction"])
    epsilon = float(values["polygon_epsilon_fraction"])
    close_fraction = float(values["close_kernel_fraction"])
    close_iterations = int(values["close_iterations"])
    rectangularity_weight = float(values["rectangularity_weight"])
    padding = float(values["bbox_padding_fraction"])
    merge_fragmented = bool(values["merge_fragmented_contours"])

    if not 0.0 <= minimum_area <= 1.0:
        raise ValueError("minimum_contour_area_fraction must be between 0 and 1")
    if not 0.0 < epsilon <= 0.25:
        raise ValueError("polygon_epsilon_fraction must be greater than 0 and at most 0.25")
    if not 0.0 <= close_fraction <= 0.25:
        raise ValueError("close_kernel_fraction must be between 0 and 0.25")
    if close_iterations < 0:
        raise ValueError("close_iterations must be non-negative")
    if not 0.0 <= rectangularity_weight <= 1.0:
        raise ValueError("rectangularity_weight must be between 0 and 1")
    if not 0.0 <= padding <= 0.25:
        raise ValueError("bbox_padding_fraction must be between 0 and 0.25")

    values["minimum_contour_area_fraction"] = minimum_area
    values["polygon_epsilon_fraction"] = epsilon
    values["close_kernel_fraction"] = close_fraction
    values["close_iterations"] = close_iterations
    values["rectangularity_weight"] = rectangularity_weight
    values["bbox_padding_fraction"] = padding
    values["merge_fragmented_contours"] = merge_fragmented
    return values


def _odd_kernel_size(fraction: float, width: int, height: int) -> int:
    if fraction <= 0.0:
        return 0
    size = max(3, int(round(min(width, height) * fraction)))
    if size % 2 == 0:
        size += 1
    return size


def _padded_bbox(
    bbox: list[int],
    padding_fraction: float,
    width: int,
    height: int,
) -> list[int]:
    x1, y1, x2, y2 = bbox
    padding = int(round(min(width, height) * padding_fraction))
    return [
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    ]


def detect(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> Candidate:
    """Fit the strongest document-like external contour in the supplied mask.

    This detector is intentionally black-box compatible with the regression
    framework: every tunable choice is supplied through ``parameters`` and the
    result is returned as a normal geometry ``Candidate``.
    """
    del image_bgr
    values = _parameters(parameters)

    if mask.ndim != 2:
        raise ValueError(f"Contour detector expects a 2-D mask, got shape {mask.shape}")

    working = np.where(mask > 0, 255, 0).astype(np.uint8)
    height, width = working.shape[:2]
    image_area = float(width * height)

    close_kernel_size = _odd_kernel_size(
        values["close_kernel_fraction"], width, height
    )
    if close_kernel_size and values["close_iterations"]:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_kernel_size, close_kernel_size)
        )
        working = cv2.morphologyEx(
            working,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=values["close_iterations"],
        )

    contours, _ = cv2.findContours(
        working, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    best: tuple[
        float,
        np.ndarray,
        list[int],
        np.ndarray,
        float,
        float,
        str,
    ] | None = None

    minimum_area = image_area * values["minimum_contour_area_fraction"]
    rectangularity_weight = values["rectangularity_weight"]

    def consider(contour: np.ndarray, source: str) -> None:
        nonlocal best
        area = float(cv2.contourArea(contour))
        if area < minimum_area:
            return

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            return

        approx = cv2.approxPolyDP(
            contour,
            values["polygon_epsilon_fraction"] * perimeter,
            True,
        )
        x, y, bbox_width, bbox_height = cv2.boundingRect(contour)
        raw_bbox = [x, y, x + bbox_width, y + bbox_height]
        bbox = _padded_bbox(
            raw_bbox,
            values["bbox_padding_fraction"],
            width,
            height,
        )

        coverage_score = float(candidate_score(working, bbox))
        rectangularity = area / max(1.0, float(bbox_width * bbox_height))
        combined = (
            coverage_score * (1.0 - rectangularity_weight)
            + min(1.0, rectangularity) * rectangularity_weight
        )

        if best is None or combined > best[0]:
            best = (
                combined,
                contour,
                bbox,
                approx,
                area,
                rectangularity,
                source,
            )

    for contour in contours:
        consider(contour, "external_contour")

    # Sparse title/index sheets often contain many small disconnected foreground
    # islands.  Optionally evaluate their convex hull as one document hypothesis.
    if best is None and values["merge_fragmented_contours"] and contours:
        points = np.concatenate(contours, axis=0)
        if len(points) >= 3:
            consider(cv2.convexHull(points), "merged_convex_hull")

    diagnostics = {
        "parameters": values,
        "external_contour_count": len(contours),
        "close_kernel_size": close_kernel_size,
        "mask_foreground_fraction": round(float(np.count_nonzero(working)) / image_area, 8),
    }

    if best is None:
        diagnostics["reason"] = "no_plausible_contour"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    combined, contour, bbox, approx, area, rectangularity, contour_source = best
    if len(approx) == 4:
        corners = [[float(point[0][0]), float(point[0][1])] for point in approx]
        corner_source = "approx_poly_dp"
    else:
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect).astype(float).tolist()
        corner_source = "minimum_area_rectangle"

    diagnostics.update(
        {
            "contour_area": round(area, 3),
            "contour_area_fraction": round(area / image_area, 8),
            "rectangularity": round(rectangularity, 8),
            "polygon_vertices": int(len(approx)),
            "corner_source": corner_source,
            "contour_source": contour_source,
        }
    )

    return Candidate(
        METHOD,
        bbox,
        corners,
        round(combined, 6),
        round(combined, 6),
        diagnostics,
    )
