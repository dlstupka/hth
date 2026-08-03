from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .model import Candidate

METHOD = "gradient_vote"

BASELINE_PARAMETERS: dict[str, Any] = {
    "gaussian_sigma": 1.2,
    "border_search_fraction": 0.42,
    "central_band_fraction": 0.86,
    "gradient_percentile": 82.0,
    "vote_smooth_fraction": 0.012,
    "minimum_vote_support": 0.16,
    "minimum_span_fraction": 0.45,
    "minimum_area_fraction": 0.20,
    "support_weight": 0.55,
    "area_weight": 0.25,
    "rectangularity_weight": 0.20,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown Gradient Boundary Voting parameters: {', '.join(unknown)}")
        values.update(overrides)
    for name in values:
        values[name] = float(values[name])
    if values["gaussian_sigma"] < 0.0:
        raise ValueError("gaussian_sigma must be non-negative")
    if not 0.1 <= values["border_search_fraction"] < 0.5:
        raise ValueError("border_search_fraction must be between 0.1 and 0.5")
    if not 0.1 <= values["central_band_fraction"] <= 1.0:
        raise ValueError("central_band_fraction must be between 0.1 and 1.0")
    if not 0.0 <= values["gradient_percentile"] <= 100.0:
        raise ValueError("gradient_percentile must be between 0 and 100")
    if not 0.0 <= values["vote_smooth_fraction"] <= 0.1:
        raise ValueError("vote_smooth_fraction must be between 0 and 0.1")
    for name in ("minimum_vote_support", "minimum_span_fraction", "minimum_area_fraction"):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    weights = [values["support_weight"], values["area_weight"], values["rectangularity_weight"]]
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("Gradient-vote score weights must be non-negative with at least one positive")
    return values


def _gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.ndim == 2:
        return image_bgr
    raise ValueError(f"Gradient Boundary Voting expects a 2-D or 3-D image, got {image_bgr.shape}")


def _smooth(profile: np.ndarray, fraction: float) -> np.ndarray:
    if fraction <= 0.0 or profile.size < 3:
        return profile.astype(np.float32)
    width = max(3, int(round(profile.size * fraction)))
    if width % 2 == 0:
        width += 1
    return cv2.GaussianBlur(profile.astype(np.float32).reshape(1, -1), (width, 1), 0).reshape(-1)


def _peak(profile: np.ndarray, start: int, end: int, smooth_fraction: float) -> tuple[int, float, float]:
    start = max(0, min(start, profile.size - 1))
    end = max(start + 1, min(end, profile.size))
    smoothed = _smooth(profile, smooth_fraction)
    region = smoothed[start:end]
    offset = int(np.argmax(region))
    index = start + offset
    peak = float(region[offset])
    baseline = float(np.median(region))
    prominence = 0.0 if peak <= 0.0 else max(0.0, (peak - baseline) / peak)
    return index, peak, prominence


def _order(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered[:, 0] + ordered[:, 1])), axis=0)


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    """Locate four page boundaries by accumulated horizontal and vertical gradient votes."""
    values = _parameters(parameters)
    gray = _gray(image_bgr)
    height, width = gray.shape
    if min(height, width) < 8:
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "image_too_small", "parameters": values}, status="no_candidate")

    working = gray
    if values["gaussian_sigma"] > 0.0:
        working = cv2.GaussianBlur(gray, (0, 0), values["gaussian_sigma"])
    gx = np.abs(cv2.Sobel(working, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(working, cv2.CV_32F, 0, 1, ksize=3))

    band = values["central_band_fraction"]
    x_margin = int(round(width * (1.0 - band) / 2.0))
    y_margin = int(round(height * (1.0 - band) / 2.0))
    vertical_profile = np.mean(gx[y_margin:max(y_margin + 1, height - y_margin), :], axis=0)
    horizontal_profile = np.mean(gy[:, x_margin:max(x_margin + 1, width - x_margin)], axis=1)

    search_x = max(2, int(round(width * values["border_search_fraction"])))
    search_y = max(2, int(round(height * values["border_search_fraction"])))
    left, left_peak, left_prom = _peak(vertical_profile, 0, search_x, values["vote_smooth_fraction"])
    right, right_peak, right_prom = _peak(vertical_profile, width - search_x, width, values["vote_smooth_fraction"])
    top, top_peak, top_prom = _peak(horizontal_profile, 0, search_y, values["vote_smooth_fraction"])
    bottom, bottom_peak, bottom_prom = _peak(horizontal_profile, height - search_y, height, values["vote_smooth_fraction"])

    span_x = (right - left) / float(width)
    span_y = (bottom - top) / float(height)
    if span_x < values["minimum_span_fraction"] or span_y < values["minimum_span_fraction"]:
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "insufficient_boundary_span", "span_x": span_x, "span_y": span_y, "parameters": values}, status="no_candidate")

    corners = _order(np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32))
    area_fraction = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2)))) / float(width * height)
    if area_fraction < values["minimum_area_fraction"]:
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "insufficient_area", "area_fraction": area_fraction, "parameters": values}, status="no_candidate")

    peaks = np.array([left_peak, right_peak, top_peak, bottom_peak], dtype=np.float32)
    threshold = float(np.percentile(np.concatenate([vertical_profile, horizontal_profile]), values["gradient_percentile"]))
    magnitude_support = float(np.mean(np.clip(peaks / max(threshold, 1e-6), 0.0, 1.0)))
    prominence_support = float(np.mean([left_prom, right_prom, top_prom, bottom_prom]))
    vote_support = 0.5 * magnitude_support + 0.5 * prominence_support
    if vote_support < values["minimum_vote_support"]:
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "insufficient_vote_support", "vote_support": vote_support, "parameters": values}, status="no_candidate")

    rectangularity = min(1.0, span_x * span_y / max(area_fraction, 1e-9))
    total = values["support_weight"] + values["area_weight"] + values["rectangularity_weight"]
    score = (vote_support * values["support_weight"] + area_fraction * values["area_weight"] + rectangularity * values["rectangularity_weight"]) / total
    return Candidate(
        METHOD,
        [int(left), int(top), int(right), int(bottom)],
        corners.astype(float).tolist(),
        float(score),
        float(score),
        {
            "parameters": values,
            "boundary_positions": {"left": int(left), "right": int(right), "top": int(top), "bottom": int(bottom)},
            "boundary_peak_strengths": peaks.astype(float).tolist(),
            "boundary_prominences": [left_prom, right_prom, top_prom, bottom_prom],
            "vote_support": vote_support,
            "area_fraction": area_fraction,
            "rectangularity": rectangularity,
            "evidence": "distributed_sobel_gradient_votes",
        },
    )


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None, candidate_corners: list[list[float]] | None = None, verbose: bool = False) -> dict[str, np.ndarray]:
    values = _parameters(parameters)
    gray = _gray(image_bgr)
    working = cv2.GaussianBlur(gray, (0, 0), values["gaussian_sigma"]) if values["gaussian_sigma"] > 0 else gray
    gx = np.abs(cv2.Sobel(working, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(working, cv2.CV_32F, 0, 1, ksize=3))
    magnitude = cv2.normalize(gx + gy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    overlay = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if candidate_corners is not None:
        cv2.polylines(overlay, [np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3, cv2.LINE_AA)
    images = {"gradient-magnitude.png": magnitude, "selected-boundaries.png": overlay}
    if verbose:
        vertical = cv2.normalize(gx, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        horizontal = cv2.normalize(gy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vote_map = np.zeros_like(overlay)
        if candidate_corners is not None:
            corners = np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(4, 2)
            for start, end in zip(corners, np.roll(corners, -1, axis=0)):
                cv2.line(vote_map, tuple(start), tuple(end), (0, 255, 255), 3, cv2.LINE_AA)
        images["vertical-gradient-votes.png"] = vertical
        images["horizontal-gradient-votes.png"] = horizontal
        images["vote-maxima.png"] = vote_map
    return images
