from __future__ import annotations

import cv2
import numpy as np

from hth.geometry.detector_lsd import detect


def _page_image() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((600, 800, 3), 24, dtype=np.uint8)
    cv2.rectangle(image, (90, 55), (710, 545), (235, 235, 235), -1)
    cv2.rectangle(image, (90, 55), (710, 545), (255, 255, 255), 4)
    mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.rectangle(mask, (90, 55), (710, 545), 255, -1)
    return image, mask


def test_lsd_returns_candidate_for_rectangular_page() -> None:
    image, mask = _page_image()
    candidate = detect(image_bgr=image, mask=mask)
    assert candidate.method == "lsd"
    assert candidate.bbox is not None
    assert candidate.confidence > 0.5
    assert candidate.diagnostics["vertical_segments"] >= 2
    assert candidate.diagnostics["horizontal_segments"] >= 2


def test_lsd_cleanly_returns_no_candidate_for_blank_image() -> None:
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    mask = np.zeros((300, 400), dtype=np.uint8)
    candidate = detect(image_bgr=image, mask=mask)
    assert candidate.bbox is None
    assert candidate.status == "ok"  # registry normalizes this to no_candidate
    assert candidate.diagnostics["reason"] in {"no_line_segments", "insufficient_axis_segments"}
