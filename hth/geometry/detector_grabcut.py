from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np

from .common import candidate_score, valid_bbox
from .model import Candidate

METHOD = "grabcut"

BASELINE_PARAMETERS: dict[str, int | float] = {
    "border_fraction": 0.02,
    "erosion_kernel_fraction": 0.015,
    "erosion_iterations": 1,
    "grabcut_iterations": 3,
    "close_kernel_fraction": 0.02,
    "close_iterations": 1,
    "minimum_bbox_area_fraction": 0.10,
    "minimum_contour_area_fraction": 0.04,
    "polygon_epsilon_fraction": 0.018,
}


def _parameters(overrides: Mapping[str, Any] | None) -> dict[str, int | float]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown GrabCut parameters: {', '.join(unknown)}")
        values.update(overrides)

    integer_names = {
        "erosion_iterations",
        "grabcut_iterations",
        "close_iterations",
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"GrabCut parameter {name!r} must be numeric")
        if name in integer_names:
            values[name] = int(value)
        else:
            values[name] = float(value)

    if not 0.0 <= float(values["border_fraction"]) < 0.5:
        raise ValueError("border_fraction must be in [0, 0.5)")
    for name in ("erosion_kernel_fraction", "close_kernel_fraction"):
        if float(values[name]) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in integer_names:
        if int(values[name]) < 0:
            raise ValueError(f"{name} must be non-negative")
    if int(values["grabcut_iterations"]) < 1:
        raise ValueError("grabcut_iterations must be at least 1")
    for name in (
        "minimum_bbox_area_fraction",
        "minimum_contour_area_fraction",
        "polygon_epsilon_fraction",
    ):
        if float(values[name]) <= 0.0:
            raise ValueError(f"{name} must be positive")
    return values


def _odd_kernel_size(minimum_dimension: int, fraction: float) -> int:
    return max(3, (round(minimum_dimension * fraction) | 1))


def detect(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: Mapping[str, Any] | None = None,
) -> Candidate:
    """Refine the shared document mask with OpenCV GrabCut segmentation.

    ``parameters`` is intentionally an opaque mapping from the calibration
    runner's perspective. Omitting it preserves the v0.6.1 detector behavior.
    """
    params = _parameters(parameters)
    height, width = mask.shape
    minimum_dimension = min(width, height)
    foreground_fraction = float(np.mean(mask > 0))
    if foreground_fraction < 0.01:
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {
                "reason": "insufficient_initial_foreground",
                "initial_foreground_fraction": round(foreground_fraction, 6),
                "parameters": params,
            },
        )

    gc_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[mask > 0] = cv2.GC_PR_FGD

    border = max(1, round(minimum_dimension * float(params["border_fraction"])))
    if border:
        gc_mask[:border, :] = cv2.GC_BGD
        gc_mask[-border:, :] = cv2.GC_BGD
        gc_mask[:, :border] = cv2.GC_BGD
        gc_mask[:, -border:] = cv2.GC_BGD

    erosion_iterations = int(params["erosion_iterations"])
    if erosion_iterations:
        kernel_size = _odd_kernel_size(
            minimum_dimension, float(params["erosion_kernel_fraction"])
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        definite_foreground = cv2.erode(
            (mask > 0).astype(np.uint8), kernel, iterations=erosion_iterations
        )
        gc_mask[definite_foreground > 0] = cv2.GC_FGD
    else:
        kernel_size = 0

    if not np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)):
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {"reason": "no_grabcut_foreground_seed", "parameters": params},
        )
    if not np.any((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD)):
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {"reason": "no_grabcut_background_seed", "parameters": params},
        )

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        image_bgr,
        gc_mask,
        None,
        background_model,
        foreground_model,
        int(params["grabcut_iterations"]),
        cv2.GC_INIT_WITH_MASK,
    )

    refined = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    close_size = _odd_kernel_size(
        minimum_dimension, float(params["close_kernel_fraction"])
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_size, close_size)
    )
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=int(params["close_iterations"]),
    )

    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {"reason": "no_grabcut_contours", "parameters": params},
        )

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    x, y, box_width, box_height = cv2.boundingRect(contour)
    box = [x, y, x + box_width, y + box_height]
    image_area = max(1, width * height)
    area_fraction = area / image_area
    bbox_area_fraction = (box_width * box_height) / image_area

    if (
        not valid_bbox(box)
        or bbox_area_fraction < float(params["minimum_bbox_area_fraction"])
        or area_fraction < float(params["minimum_contour_area_fraction"])
    ):
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {
                "reason": "grabcut_region_too_small",
                "contour_area_fraction": round(area_fraction, 6),
                "bbox_area_fraction": round(bbox_area_fraction, 6),
                "parameters": params,
            },
        )

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(
        contour, float(params["polygon_epsilon_fraction"]) * perimeter, True
    )
    if len(approx) == 4:
        corners = [[float(point[0][0]), float(point[0][1])] for point in approx]
    else:
        corners = cv2.boxPoints(cv2.minAreaRect(contour)).astype(float).tolist()

    refined_score = candidate_score(refined, box)
    shared_mask_score = candidate_score(mask, box)
    rectangularity = area / max(1.0, box_width * box_height)
    combined = (
        0.45 * refined_score
        + 0.35 * shared_mask_score
        + 0.20 * min(1.0, rectangularity)
    )

    return Candidate(
        METHOD,
        box,
        corners,
        round(combined, 6),
        round(combined, 6),
        {
            "iterations": int(params["grabcut_iterations"]),
            "border_px": border,
            "erosion_kernel_px": kernel_size,
            "close_kernel_px": close_size,
            "initial_foreground_fraction": round(foreground_fraction, 6),
            "refined_foreground_fraction": round(float(np.mean(refined > 0)), 6),
            "contour_area_fraction": round(area_fraction, 6),
            "bbox_area_fraction": round(bbox_area_fraction, 6),
            "rectangularity": round(rectangularity, 6),
            "refined_mask_score": round(refined_score, 6),
            "shared_mask_score": round(shared_mask_score, 6),
            "parameters": params,
        },
    )
