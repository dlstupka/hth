from __future__ import annotations

import math

import cv2
import numpy as np

from .common import candidate_score, valid_bbox
from .model import Candidate

METHOD = "hough"


def detect(*, image_bgr: np.ndarray, mask: np.ndarray) -> Candidate:
    height, width = mask.shape
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800,
        threshold=max(35, min(width, height) // 18),
        minLineLength=max(45, min(width, height) // 5),
        maxLineGap=max(20, min(width, height) // 18),
    )
    if lines is None:
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "no_hough_lines"})

    vertical, horizontal = [], []
    for item in np.asarray(lines, dtype=float).reshape(-1, 4):
        x1, y1, x2, y2 = map(float, item)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180
        if angle > 90:
            angle = 180 - angle
        if angle >= 68:
            vertical.append((x1, y1, x2, y2, length))
        elif angle <= 22:
            horizontal.append((x1, y1, x2, y2, length))

    if len(vertical) < 2 or len(horizontal) < 2:
        return Candidate(
            METHOD, None, None, 0.0, 0.0,
            {
                "reason": "insufficient_axis_lines",
                "vertical_lines": len(vertical),
                "horizontal_lines": len(horizontal),
            },
        )

    vx = np.asarray([((x1 + x2) / 2, length) for x1, _, x2, _, length in vertical])
    hy = np.asarray([((y1 + y2) / 2, length) for _, y1, _, y2, length in horizontal])
    left = int(round(np.percentile(vx[:, 0], 10)))
    right = int(round(np.percentile(vx[:, 0], 90)))
    top = int(round(np.percentile(hy[:, 0], 10)))
    bottom = int(round(np.percentile(hy[:, 0], 90)))
    box = [max(0, left), max(0, top), min(width, right), min(height, bottom)]

    if not valid_bbox(box):
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "invalid_line_envelope"})

    score = candidate_score(mask, box)
    support = min(1.0, (len(vertical) + len(horizontal)) / 24.0)
    combined = 0.72 * score + 0.28 * support
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
            "vertical_lines": len(vertical),
            "horizontal_lines": len(horizontal),
            "support_score": round(support, 6),
        },
    )
