from __future__ import annotations

import math

import cv2
import numpy as np

from .common import candidate_score, valid_bbox
from .model import Candidate

METHOD = "lsd"


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


def detect(*, image_bgr: np.ndarray, mask: np.ndarray) -> Candidate:
    """Estimate a page envelope from OpenCV Line Segment Detector output."""
    height, width = mask.shape
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(gray)
    lines = detected[0] if detected else None
    if lines is None or np.asarray(lines).size == 0:
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "no_line_segments"})

    # OpenCV versions may return (N, 1, 4) or (N, 4).
    segments = np.asarray(lines, dtype=float).reshape(-1, 4)
    minimum_length = max(30.0, min(width, height) * 0.14)
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
        if angle >= 72.0:
            vertical.append(((x1 + x2) / 2.0, length))
        elif angle <= 18.0:
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

    left = int(round(_weighted_percentile(vx, vw, 10.0)))
    right = int(round(_weighted_percentile(vx, vw, 90.0)))
    top = int(round(_weighted_percentile(hy, hw, 10.0)))
    bottom = int(round(_weighted_percentile(hy, hw, 90.0)))
    box = [max(0, left), max(0, top), min(width, right), min(height, bottom)]

    if not valid_bbox(box):
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "invalid_lsd_envelope"})

    area_fraction = ((box[2] - box[0]) * (box[3] - box[1])) / max(1, width * height)
    if area_fraction < 0.10:
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {
                "reason": "lsd_envelope_too_small",
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
