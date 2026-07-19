from __future__ import annotations

import cv2
import numpy as np

from .common import candidate_score, valid_bbox
from .model import Candidate

METHOD = "grabcut"


def detect(*, image_bgr: np.ndarray, mask: np.ndarray) -> Candidate:
    """Refine the shared document mask with OpenCV GrabCut segmentation."""
    height, width = mask.shape
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
            },
        )

    gc_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[mask > 0] = cv2.GC_PR_FGD

    border = max(2, round(min(width, height) * 0.02))
    gc_mask[:border, :] = cv2.GC_BGD
    gc_mask[-border:, :] = cv2.GC_BGD
    gc_mask[:, :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD

    kernel_size = max(3, (round(min(width, height) * 0.015) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    definite_foreground = cv2.erode((mask > 0).astype(np.uint8), kernel, iterations=1)
    gc_mask[definite_foreground > 0] = cv2.GC_FGD

    if not np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)):
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "no_grabcut_foreground_seed"})
    if not np.any((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD)):
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "no_grabcut_background_seed"})

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        image_bgr,
        gc_mask,
        None,
        background_model,
        foreground_model,
        3,
        cv2.GC_INIT_WITH_MASK,
    )

    refined = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    close_size = max(3, (round(min(width, height) * 0.02) | 1))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "no_grabcut_contours"})

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    x, y, box_width, box_height = cv2.boundingRect(contour)
    box = [x, y, x + box_width, y + box_height]
    image_area = max(1, width * height)
    area_fraction = area / image_area
    bbox_area_fraction = (box_width * box_height) / image_area

    if not valid_bbox(box) or bbox_area_fraction < 0.10 or area_fraction < 0.04:
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
            },
        )

    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
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
            "iterations": 3,
            "initial_foreground_fraction": round(foreground_fraction, 6),
            "refined_foreground_fraction": round(float(np.mean(refined > 0)), 6),
            "contour_area_fraction": round(area_fraction, 6),
            "bbox_area_fraction": round(bbox_area_fraction, 6),
            "rectangularity": round(rectangularity, 6),
            "refined_mask_score": round(refined_score, 6),
            "shared_mask_score": round(shared_mask_score, 6),
        },
    )
