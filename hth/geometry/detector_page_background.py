from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .model import Candidate

METHOD = "page_background"

BASELINE_PARAMETERS: dict[str, Any] = {
    "border_band_fraction": 0.06,
    "color_distance_threshold": 3.0,
    "blur_sigma": 1.2,
    "close_kernel_fraction": 0.008,
    "open_kernel_fraction": 0.003,
    "minimum_border_background_fraction": 0.50,
    "minimum_page_area_fraction": 0.25,
    "maximum_page_area_fraction": 0.98,
    "minimum_rectangularity": 0.60,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, float]:
    values = dict(BASELINE_PARAMETERS)
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise ValueError(f"Unknown Page Background parameters: {', '.join(unknown)}")
    values.update(overrides)
    values = {name: float(value) for name, value in values.items()}
    if not 0.005 <= values["border_band_fraction"] <= 0.25:
        raise ValueError("border_band_fraction must be between 0.005 and 0.25")
    if values["color_distance_threshold"] <= 0.0:
        raise ValueError("color_distance_threshold must be positive")
    if values["blur_sigma"] < 0.0:
        raise ValueError("blur_sigma must be non-negative")
    for name in ("close_kernel_fraction", "open_kernel_fraction"):
        if not 0.0 <= values[name] <= 0.10:
            raise ValueError(f"{name} must be between 0 and 0.10")
    for name in ("minimum_border_background_fraction", "minimum_page_area_fraction", "maximum_page_area_fraction", "minimum_rectangularity"):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if values["minimum_page_area_fraction"] >= values["maximum_page_area_fraction"]:
        raise ValueError("minimum_page_area_fraction must be below maximum_page_area_fraction")
    return values


def _lab(image_bgr: np.ndarray, sigma: float) -> np.ndarray:
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    elif image_bgr.ndim != 3:
        raise ValueError(f"Page Background expects a 2-D or 3-D image, got {image_bgr.shape}")
    working = cv2.GaussianBlur(image_bgr, (0, 0), sigma) if sigma > 0 else image_bgr
    return cv2.cvtColor(working, cv2.COLOR_BGR2LAB).astype(np.float32)


def _border_mask(shape: tuple[int, int], fraction: float) -> np.ndarray:
    h, w = shape
    band = max(1, int(round(min(h, w) * fraction)))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:band, :] = 255
    mask[-band:, :] = 255
    mask[:, :band] = 255
    mask[:, -band:] = 255
    return mask


def _background_evidence(image_bgr: np.ndarray, values: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    lab = _lab(image_bgr, values["blur_sigma"])
    border_mask = _border_mask(lab.shape[:2], values["border_band_fraction"])
    samples = lab[border_mask > 0]
    median = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - median), axis=0)
    robust_scale = np.maximum(1.4826 * mad, np.array([6.0, 3.0, 3.0], dtype=np.float32))
    normalized = (lab - median.reshape(1, 1, 3)) / robust_scale.reshape(1, 1, 3)
    distance = np.sqrt(np.sum(normalized * normalized, axis=2))
    background = np.where(distance <= values["color_distance_threshold"], 255, 0).astype(np.uint8)
    border_fraction = float(np.count_nonzero(background[border_mask > 0])) / max(1, int(np.count_nonzero(border_mask)))
    diagnostics = {
        "background_lab_median": [float(x) for x in median],
        "background_lab_robust_scale": [float(x) for x in robust_scale],
        "border_background_fraction": border_fraction,
        "border_band_pixels": int(max(1, round(min(lab.shape[:2]) * values["border_band_fraction"]))),
    }
    return distance, background, border_mask, diagnostics


def _morph(mask: np.ndarray, values: dict[str, float]) -> np.ndarray:
    h, w = mask.shape
    result = mask.copy()
    for operation, name in ((cv2.MORPH_CLOSE, "close_kernel_fraction"), (cv2.MORPH_OPEN, "open_kernel_fraction")):
        fraction = values[name]
        if fraction <= 0:
            continue
        size = max(3, int(round(min(h, w) * fraction)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        result = cv2.morphologyEx(result, operation, kernel)
    return result


def _candidate_contour(foreground: np.ndarray, values: dict[str, float]) -> tuple[np.ndarray | None, dict[str, float]]:
    h, w = foreground.shape
    image_area = float(h * w)
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    diagonal = max(float(np.hypot(w, h)), 1.0)
    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_metrics: dict[str, float] = {}
    best_score = -1.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        area_fraction = area / image_area
        if not values["minimum_page_area_fraction"] <= area_fraction <= values["maximum_page_area_fraction"]:
            continue
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        rect_area = max(float(rw * rh), 1.0)
        rectangularity = min(1.0, area / rect_area)
        if rectangularity < values["minimum_rectangularity"]:
            continue
        centroid = np.array(rect[0], dtype=np.float32)
        center_score = max(0.0, 1.0 - float(np.linalg.norm(centroid - center)) / (0.5 * diagonal))
        score = 0.55 * rectangularity + 0.30 * area_fraction + 0.15 * center_score
        if score > best_score:
            best = contour
            best_score = score
            best_metrics = {
                "page_area_fraction": area_fraction,
                "rectangularity": rectangularity,
                "center_score": center_score,
                "proposal_score": score,
            }
    return best, best_metrics


def _proposal(image_bgr: np.ndarray, values: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    distance, background, border_mask, diagnostics = _background_evidence(image_bgr, values)
    foreground = _morph(cv2.bitwise_not(background), values)
    contour, metrics = _candidate_contour(foreground, values)
    diagnostics.update(metrics)
    return distance, background, border_mask, contour, diagnostics


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    """Infer the page by learning the capture background from image-border samples."""
    del mask
    values = _parameters(parameters)
    _distance, _background, _border_mask_image, contour, diagnostics = _proposal(image_bgr, values)
    diagnostics["parameters"] = values
    diagnostics["evidence"] = "robust_outer_border_background_model"
    if diagnostics["border_background_fraction"] < values["minimum_border_background_fraction"]:
        diagnostics["reason"] = "insufficient_coherent_border_background"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    if contour is None:
        diagnostics["reason"] = "no_plausible_non_background_page_region"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    rect = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rect).astype(np.float32)
    x, y, bw, bh = cv2.boundingRect(corners)
    h, w = _background.shape
    bbox = [max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)]
    border_quality = min(1.0, diagnostics["border_background_fraction"] / max(values["minimum_border_background_fraction"], 1e-6))
    score = float(np.clip(0.55 * diagnostics["proposal_score"] + 0.45 * border_quality, 0.0, 1.0))
    diagnostics["score_components"] = {"proposal": diagnostics["proposal_score"], "border_background": border_quality}
    return Candidate(METHOD, bbox, corners.astype(float).tolist(), score, score, diagnostics)


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None, candidate_corners: list[list[float]] | None = None, verbose: bool = False) -> dict[str, np.ndarray]:
    del mask
    values = _parameters(parameters)
    distance, background, border_mask, contour, diagnostics = _proposal(image_bgr, values)
    max_distance = max(values["color_distance_threshold"] * 2.0, 1.0)
    distance_image = np.rint(np.clip(distance / max_distance, 0.0, 1.0) * 255.0).astype(np.uint8)
    foreground = cv2.bitwise_not(background)
    overlay = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 255), 2)
    if candidate_corners is not None:
        corners = np.rint(np.asarray(candidate_corners, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [corners], True, (0, 0, 255), 3, cv2.LINE_AA)
    images = {
        "page-background-distance.png": distance_image,
        "page-background-mask.png": background,
        "page-background-candidate.png": overlay,
    }
    if verbose:
        border_view = overlay.copy()
        border_view[border_mask > 0] = (0.55 * border_view[border_mask > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        cv2.putText(border_view, f"border bg={diagnostics['border_background_fraction']:.3f}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        images["page-background-border-samples.png"] = border_view
        images["page-background-foreground.png"] = foreground
    return images


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
