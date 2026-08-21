from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .model import Candidate

METHOD = "scantailor_page_frame"

# Clean-room HTH implementation of the page/content-frame ideas used by
# ScanTailor-class scan-processing workflows.  No ScanTailor source code is
# embedded or translated here; this detector intentionally remains an
# independent OpenCV implementation so it can be calibrated alongside the
# rest of HTH's detector catalog.
BASELINE_PARAMETERS: dict[str, Any] = {
    "illumination_sigma_fraction": 0.035,
    "ink_quantile": 0.82,
    "content_close_fraction": 0.012,
    "projection_smooth_fraction": 0.008,
    "boundary_search_margin_fraction": 0.06,
    "minimum_page_area_fraction": 0.20,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, float]:
    values = dict(BASELINE_PARAMETERS)
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise ValueError(f"Unknown ScanTailor Page Frame parameters: {', '.join(unknown)}")
    values.update(overrides)
    values = {name: float(value) for name, value in values.items()}
    if not 0.002 <= values["illumination_sigma_fraction"] <= 0.25:
        raise ValueError("illumination_sigma_fraction must be between 0.002 and 0.25")
    if not 0.50 <= values["ink_quantile"] <= 0.98:
        raise ValueError("ink_quantile must be between 0.50 and 0.98")
    if not 0.0 <= values["content_close_fraction"] <= 0.10:
        raise ValueError("content_close_fraction must be between 0 and 0.10")
    if not 0.0 <= values["projection_smooth_fraction"] <= 0.10:
        raise ValueError("projection_smooth_fraction must be between 0 and 0.10")
    if not 0.0 <= values["boundary_search_margin_fraction"] <= 0.30:
        raise ValueError("boundary_search_margin_fraction must be between 0 and 0.30")
    if not 0.0 <= values["minimum_page_area_fraction"] <= 1.0:
        raise ValueError("minimum_page_area_fraction must be between 0 and 1")
    return values


def _gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr.astype(np.uint8, copy=False)
    if image_bgr.ndim != 3:
        raise ValueError(f"ScanTailor Page Frame expects a 2-D or 3-D image, got {image_bgr.shape}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def _odd_size(value: float) -> int:
    size = max(3, int(round(value)))
    return size if size % 2 else size + 1


def _content_evidence(gray: np.ndarray, values: dict[str, float]) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int] | None]:
    h, w = gray.shape
    sigma = max(1.0, min(h, w) * values["illumination_sigma_fraction"])
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    # Historical pages can be lighter or darker than their local surround; the
    # absolute locally-normalized residual captures both polarities.
    residual = cv2.absdiff(gray, background)
    threshold = float(np.quantile(residual, values["ink_quantile"]))
    threshold = max(2.0, threshold)
    content = np.where(residual >= threshold, 255, 0).astype(np.uint8)

    fraction = values["content_close_fraction"]
    if fraction > 0.0:
        size = _odd_size(min(h, w) * fraction)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, kernel)

    # Suppress tiny isolated marks without destroying sparse handwritten pages.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(content, 8)
    cleaned = np.zeros_like(content)
    min_component = max(4, int(round(h * w * 0.00002)))
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_component:
            cleaned[labels == label] = 255
    ys, xs = np.nonzero(cleaned)
    if xs.size == 0:
        return residual, cleaned, None
    return residual, cleaned, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _smooth_projection(values: np.ndarray, sigma_pixels: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    if sigma_pixels <= 0.0:
        return vector[:, 0]
    smoothed = cv2.GaussianBlur(vector, (1, 0), sigmaX=0.0, sigmaY=max(0.5, sigma_pixels))
    return smoothed[:, 0]


def _peak(profile: np.ndarray, lo: int, hi: int, *, reverse: bool = False) -> tuple[int, float]:
    n = len(profile)
    lo = max(0, min(n - 1, int(lo)))
    hi = max(lo + 1, min(n, int(hi)))
    segment = profile[lo:hi]
    if segment.size == 0:
        return (hi - 1 if reverse else lo), 0.0
    peak_value = float(segment.max())
    candidates = np.flatnonzero(segment >= peak_value - 1e-6)
    offset = int(candidates[-1] if reverse else candidates[0])
    return lo + offset, peak_value


def _page_frame(image_bgr: np.ndarray, values: dict[str, float]):
    gray = _gray(image_bgr)
    h, w = gray.shape
    residual, content, content_box = _content_evidence(gray, values)
    if content_box is None:
        return residual, content, None, {}

    # Edge-energy projections are a practical page-frame analogue of the
    # one-dimensional scan analysis used in scan-processing tools: content
    # tells us which side of the page edge to search, while image gradients
    # select the strongest physical frame transition in that outer band.
    blur = cv2.GaussianBlur(gray, (3, 3), 0.8)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    x_profile = np.mean(np.abs(gx), axis=0)
    y_profile = np.mean(np.abs(gy), axis=1)
    smooth = min(h, w) * values["projection_smooth_fraction"]
    x_profile = _smooth_projection(x_profile, smooth)
    y_profile = _smooth_projection(y_profile, smooth)

    cx1, cy1, cx2, cy2 = content_box
    mx = max(2, int(round(w * values["boundary_search_margin_fraction"])))
    my = max(2, int(round(h * values["boundary_search_margin_fraction"])))
    guard_x = max(1, int(round(w * 0.005)))
    guard_y = max(1, int(round(h * 0.005)))

    left, left_energy = _peak(x_profile, guard_x, min(w, cx1 + mx), reverse=False)
    right, right_energy = _peak(x_profile, max(0, cx2 - mx), w - guard_x, reverse=True)
    top, top_energy = _peak(y_profile, guard_y, min(h, cy1 + my), reverse=False)
    bottom, bottom_energy = _peak(y_profile, max(0, cy2 - my), h - guard_y, reverse=True)

    if right <= left or bottom <= top:
        return residual, content, None, {"content_box": list(content_box), "reason": "invalid_frame_order"}

    area_fraction = float((right - left) * (bottom - top)) / float(h * w)
    energies = np.asarray([left_energy, right_energy, top_energy, bottom_energy], dtype=np.float32)
    mean_energy = float(energies.mean())
    balance = float(np.clip(1.0 - energies.std() / max(mean_energy, 1e-6), 0.0, 1.0))
    diagnostics = {
        "content_box": list(content_box),
        "page_area_fraction": area_fraction,
        "side_edge_energy": [float(v) for v in energies],
        "side_energy_balance": balance,
    }
    corners = np.asarray([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32)
    return residual, content, corners, diagnostics


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    del mask
    values = _parameters(parameters)
    _residual, _content, corners, diagnostics = _page_frame(image_bgr, values)
    diagnostics["parameters"] = values
    diagnostics["evidence"] = "scantailor_style_content_guided_page_frame"
    diagnostics["implementation"] = "HTH clean-room OpenCV implementation; no ScanTailor source code embedded"
    if corners is None:
        diagnostics.setdefault("reason", "no_content_guided_page_frame")
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    if diagnostics["page_area_fraction"] < values["minimum_page_area_fraction"]:
        diagnostics["reason"] = "page_frame_too_small"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    h, w = image_bgr.shape[:2]
    left, top = corners[0]
    right, bottom = corners[2]
    bbox = [max(0, int(round(left))), max(0, int(round(top))), min(w, int(round(right))), min(h, int(round(bottom)))]
    area_score = min(1.0, diagnostics["page_area_fraction"] / 0.60)
    score = float(np.clip(0.65 * diagnostics["side_energy_balance"] + 0.35 * area_score, 0.0, 1.0))
    diagnostics["score_components"] = {"side_energy_balance": diagnostics["side_energy_balance"], "page_area": area_score}
    return Candidate(METHOD, bbox, corners.astype(float).tolist(), score, score, diagnostics)


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None, candidate_corners: list[list[float]] | None = None, verbose: bool = False) -> dict[str, np.ndarray]:
    del mask, verbose
    values = _parameters(parameters)
    residual, content, corners, _diagnostics = _page_frame(image_bgr, values)
    overlay = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        cv2.polylines(overlay, [np.rint(corners).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2, cv2.LINE_AA)
    if candidate_corners is not None:
        cv2.polylines(overlay, [np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3, cv2.LINE_AA)
    residual_view = np.rint(np.clip(residual.astype(np.float32) * 4.0, 0, 255)).astype(np.uint8)
    return {
        "scantailor-local-contrast.png": residual_view,
        "scantailor-content-mask.png": content,
        "scantailor-page-frame.png": overlay,
    }


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
