from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .common import candidate_score, valid_bbox
from .model import Candidate

METHOD = "lsd"

BASELINE_PARAMETERS: dict[str, Any] = {
    "refine_mode": "std",
    "scale": 0.8,
    "minimum_length_fraction": 0.14,
    "axis_angle_tolerance_degrees": 18.0,
    "outer_percentile": 10.0,
    "minimum_bbox_area_fraction": 0.10,
    "bbox_padding_fraction": 0.0,
}

_REFINE_MODES = {
    "none": cv2.LSD_REFINE_NONE,
    "std": cv2.LSD_REFINE_STD,
    "adv": cv2.LSD_REFINE_ADV,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown LSD parameters: {', '.join(unknown)}")
        values.update(overrides)

    refine_mode = str(values["refine_mode"]).lower()
    scale = float(values["scale"])
    minimum_length = float(values["minimum_length_fraction"])
    angle_tolerance = float(values["axis_angle_tolerance_degrees"])
    outer_percentile = float(values["outer_percentile"])
    minimum_area = float(values["minimum_bbox_area_fraction"])
    padding = float(values["bbox_padding_fraction"])

    if refine_mode not in _REFINE_MODES:
        raise ValueError(
            "refine_mode must be one of: " + ", ".join(sorted(_REFINE_MODES))
        )
    if not 0.0 < scale <= 1.0:
        raise ValueError("scale must be greater than 0 and at most 1")
    if not 0.0 < minimum_length <= 1.0:
        raise ValueError("minimum_length_fraction must be greater than 0 and at most 1")
    if not 0.0 < angle_tolerance < 45.0:
        raise ValueError("axis_angle_tolerance_degrees must be greater than 0 and less than 45")
    if not 0.0 <= outer_percentile < 50.0:
        raise ValueError("outer_percentile must be between 0 and 50")
    if not 0.0 <= minimum_area <= 1.0:
        raise ValueError("minimum_bbox_area_fraction must be between 0 and 1")
    if not 0.0 <= padding <= 0.25:
        raise ValueError("bbox_padding_fraction must be between 0 and 0.25")

    values.update(
        {
            "refine_mode": refine_mode,
            "scale": scale,
            "minimum_length_fraction": minimum_length,
            "axis_angle_tolerance_degrees": angle_tolerance,
            "outer_percentile": outer_percentile,
            "minimum_bbox_area_fraction": minimum_area,
            "bbox_padding_fraction": padding,
        }
    )
    return values


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    if cumulative[-1] <= 0:
        return float(np.percentile(values, percentile))
    target = cumulative[-1] * percentile / 100.0
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _padded_bbox(
    bbox: list[int],
    padding_fraction: float,
    width: int,
    height: int,
) -> list[int]:
    padding = int(round(min(width, height) * padding_fraction))
    return [
        max(0, bbox[0] - padding),
        max(0, bbox[1] - padding),
        min(width, bbox[2] + padding),
        min(height, bbox[3] + padding),
    ]


def detect(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> Candidate:
    """Estimate a page envelope from OpenCV Line Segment Detector output."""
    values = _parameters(parameters)
    if mask.ndim != 2:
        raise ValueError(f"LSD detector expects a 2-D mask, got shape {mask.shape}")

    height, width = mask.shape
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    detector = cv2.createLineSegmentDetector(
        _REFINE_MODES[values["refine_mode"]],
        scale=values["scale"],
    )
    detected = detector.detect(gray)
    lines = detected[0] if detected else None
    if lines is None or np.asarray(lines).size == 0:
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {"reason": "no_line_segments", "parameters": values},
        )

    # OpenCV versions may return (N, 1, 4) or (N, 4).
    segments = np.asarray(lines, dtype=float).reshape(-1, 4)
    minimum_length = max(
        30.0, min(width, height) * values["minimum_length_fraction"]
    )
    angle_tolerance = values["axis_angle_tolerance_degrees"]
    vertical: list[tuple[float, float]] = []
    horizontal: list[tuple[float, float]] = []

    for x1, y1, x2, y2 in segments:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < minimum_length:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
        if angle > 90.0:
            angle = 180.0 - angle
        if angle >= 90.0 - angle_tolerance:
            vertical.append(((x1 + x2) / 2.0, length))
        elif angle <= angle_tolerance:
            horizontal.append(((y1 + y2) / 2.0, length))

    if len(vertical) < 2 or len(horizontal) < 2:
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {
                "reason": "insufficient_axis_segments",
                "parameters": values,
                "line_segments": int(len(segments)),
                "vertical_segments": len(vertical),
                "horizontal_segments": len(horizontal),
                "minimum_length_px": round(minimum_length, 3),
            },
        )

    vx = np.asarray([position for position, _ in vertical], dtype=float)
    vw = np.asarray([length for _, length in vertical], dtype=float)
    hy = np.asarray([position for position, _ in horizontal], dtype=float)
    hw = np.asarray([length for _, length in horizontal], dtype=float)

    lower = values["outer_percentile"]
    upper = 100.0 - lower
    left = int(round(_weighted_percentile(vx, vw, lower)))
    right = int(round(_weighted_percentile(vx, vw, upper)))
    top = int(round(_weighted_percentile(hy, hw, lower)))
    bottom = int(round(_weighted_percentile(hy, hw, upper)))
    raw_box = [max(0, left), max(0, top), min(width, right), min(height, bottom)]
    box = _padded_bbox(raw_box, values["bbox_padding_fraction"], width, height)

    if not valid_bbox(box):
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {"reason": "invalid_lsd_envelope", "parameters": values},
        )

    area_fraction = ((box[2] - box[0]) * (box[3] - box[1])) / max(1, width * height)
    if area_fraction < values["minimum_bbox_area_fraction"]:
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {
                "reason": "lsd_envelope_too_small",
                "parameters": values,
                "bbox_area_fraction": round(area_fraction, 6),
                "vertical_segments": len(vertical),
                "horizontal_segments": len(horizontal),
            },
        )

    mask_score = candidate_score(mask, box)
    support = min(1.0, (len(vertical) + len(horizontal)) / 20.0)
    area_score = min(1.0, area_fraction / 0.60)
    combined = 0.70 * mask_score + 0.20 * support + 0.10 * area_score
    corners = [
        [float(box[0]), float(box[1])],
        [float(box[2]), float(box[1])],
        [float(box[2]), float(box[3])],
        [float(box[0]), float(box[3])],
    ]
    return Candidate(
        METHOD,
        box,
        corners,
        round(combined, 6),
        round(combined, 6),
        {
            "parameters": values,
            "line_segments": int(len(segments)),
            "vertical_segments": len(vertical),
            "horizontal_segments": len(horizontal),
            "minimum_length_px": round(minimum_length, 3),
            "bbox_area_fraction": round(area_fraction, 6),
            "mask_score": round(mask_score, 6),
            "support_score": round(support, 6),
            "area_score": round(area_score, 6),
        },
    )
