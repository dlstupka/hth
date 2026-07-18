from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def valid_bbox(box: Any) -> bool:
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


def resize_for_analysis(image: np.ndarray, maximum: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, maximum / max(width, height))
    if scale == 1.0:
        return image, 1.0
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def border_background_lab(image_bgr: np.ndarray, fraction: float = 0.035) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    height, width = lab.shape[:2]
    sx = max(2, round(width * fraction))
    sy = max(2, round(height * fraction))
    strips = np.concatenate(
        [
            lab[:sy, :, :].reshape(-1, 3),
            lab[height - sy :, :, :].reshape(-1, 3),
            lab[sy : height - sy, :sx, :].reshape(-1, 3),
            lab[sy : height - sy, width - sx :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(strips, axis=0)


def document_mask(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    bg = border_background_lab(image_bgr)
    distance = np.linalg.norm(lab.astype(np.float32) - bg.astype(np.float32), axis=2)

    border = np.concatenate(
        [
            distance[: max(2, distance.shape[0] // 30), :].ravel(),
            distance[-max(2, distance.shape[0] // 30) :, :].ravel(),
            distance[:, : max(2, distance.shape[1] // 30)].ravel(),
            distance[:, -max(2, distance.shape[1] // 30) :].ravel(),
        ]
    )
    threshold = max(18.0, float(np.percentile(border, 99)) + 5.0)
    mask = (distance >= threshold).astype(np.uint8) * 255

    short = max(3, (min(mask.shape) // 180) | 1)
    long = max(7, (min(mask.shape) // 70) | 1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (long, long))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (short, short))
    )

    return mask, {
        "background_lab": [round(float(v), 3) for v in bg],
        "color_distance_threshold": round(threshold, 3),
        "foreground_fraction": round(float(np.mean(mask > 0)), 6),
    }


def scale_bbox(box: list[int], inverse_scale: float, width: int, height: int) -> list[int]:
    return [
        max(0, min(width, round(box[0] * inverse_scale))),
        max(0, min(height, round(box[1] * inverse_scale))),
        max(0, min(width, round(box[2] * inverse_scale))),
        max(0, min(height, round(box[3] * inverse_scale))),
    ]


def bbox_from_points(points: np.ndarray, width: int, height: int) -> list[int] | None:
    if points.size == 0:
        return None
    xs = points[:, 0]
    ys = points[:, 1]
    box = [
        max(0, int(math.floor(float(xs.min())))),
        max(0, int(math.floor(float(ys.min())))),
        min(width, int(math.ceil(float(xs.max())))),
        min(height, int(math.ceil(float(ys.max())))),
    ]
    return box if valid_bbox(box) else None


def candidate_score(mask: np.ndarray, box: list[int] | None) -> float:
    if not valid_bbox(box):
        return 0.0
    height, width = mask.shape[:2]
    left, top, right, bottom = box
    inside = mask[top:bottom, left:right]
    if inside.size == 0:
        return 0.0

    interior = float(np.mean(inside > 0))
    area_fraction = ((right - left) * (bottom - top)) / max(1, width * height)

    outside_mask = np.ones(mask.shape, dtype=bool)
    outside_mask[top:bottom, left:right] = False
    outside_values = mask[outside_mask]
    outside_foreground = float(np.mean(outside_values > 0)) if outside_values.size else 0.0

    size_score = min(1.0, area_fraction / 0.55)
    score = 0.50 * interior + 0.30 * size_score + 0.20 * (1.0 - outside_foreground)
    return max(0.0, min(1.0, score))
