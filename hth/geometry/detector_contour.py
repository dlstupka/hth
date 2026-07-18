from __future__ import annotations

import cv2
import numpy as np

from .common import candidate_score
from .model import Candidate

METHOD = "contour"


def detect(*, image_bgr: np.ndarray, mask: np.ndarray) -> Candidate:
    del image_bgr
    height, width = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[float, np.ndarray, list[int], np.ndarray] | None = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < width * height * 0.12:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        x, y, w, h = cv2.boundingRect(contour)
        box = [x, y, x + w, y + h]
        score = candidate_score(mask, box)
        rectangularity = area / max(1.0, w * h)
        combined = score * 0.75 + min(1.0, rectangularity) * 0.25
        if best is None or combined > best[0]:
            best = (combined, contour, box, approx)

    if best is None:
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "no_plausible_contour"})

    combined, contour, box, approx = best
    if len(approx) == 4:
        corners = [[float(p[0][0]), float(p[0][1])] for p in approx]
    else:
        rect = cv2.minAreaRect(contour)
        corners = cv2.boxPoints(rect).astype(float).tolist()

    return Candidate(
        METHOD,
        box,
        corners,
        round(combined, 6),
        round(combined, 6),
        {
            "contour_area": round(float(cv2.contourArea(contour)), 3),
            "polygon_vertices": int(len(approx)),
        },
    )
