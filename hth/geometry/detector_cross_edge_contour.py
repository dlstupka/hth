from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import detector_contour_quad
from .model import Candidate

METHOD = "cross_edge_contour"

BASELINE_PARAMETERS: dict[str, Any] = {
    "minimum_contour_area_fraction": 0.12,
    "epsilon_max_fraction": 0.04,
    "minimum_rectangularity": 0.55,
    "sample_offset_fraction": 0.008,
    "samples_per_edge": 48,
    "minimum_cross_edge_contrast": 0.045,
    "minimum_polarity_consistency": 0.55,
    "contour_weight": 0.45,
    "contrast_weight": 0.40,
    "polarity_weight": 0.15,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown Cross-Edge Contour parameters: {', '.join(unknown)}")
        values.update(overrides)
    for name in values:
        if name == "samples_per_edge":
            values[name] = int(values[name])
        else:
            values[name] = float(values[name])
    for name in ("minimum_contour_area_fraction", "minimum_rectangularity", "minimum_cross_edge_contrast", "minimum_polarity_consistency"):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not 0.0 < values["epsilon_max_fraction"] <= 0.25:
        raise ValueError("epsilon_max_fraction must be greater than 0 and at most 0.25")
    if not 0.0 < values["sample_offset_fraction"] <= 0.1:
        raise ValueError("sample_offset_fraction must be greater than 0 and at most 0.1")
    if values["samples_per_edge"] < 4:
        raise ValueError("samples_per_edge must be at least 4")
    weights = [values["contour_weight"], values["contrast_weight"], values["polarity_weight"]]
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("Cross-edge score weights must be non-negative with at least one positive")
    return values


def _gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.ndim == 2:
        return image_bgr
    raise ValueError(f"Cross-Edge Contour expects a 2-D or 3-D image, got {image_bgr.shape}")


def _bilinear(gray: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    x = np.clip(points[:, 0], 0, w - 1)
    y = np.clip(points[:, 1], 0, h - 1)
    x0 = np.floor(x).astype(int); x1 = np.minimum(x0 + 1, w - 1)
    y0 = np.floor(y).astype(int); y1 = np.minimum(y0 + 1, h - 1)
    wx = x - x0; wy = y - y0
    return ((1-wx)*(1-wy)*gray[y0,x0] + wx*(1-wy)*gray[y0,x1] + (1-wx)*wy*gray[y1,x0] + wx*wy*gray[y1,x1]).astype(np.float32)


def _cross_edge_evidence(gray: np.ndarray, corners: np.ndarray, offset: float, samples: int) -> tuple[float, float, list[float]]:
    center = corners.mean(axis=0)
    signed: list[float] = []
    side_contrast: list[float] = []
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length <= 1.0:
            side_contrast.append(0.0)
            continue
        tangent = edge / length
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        midpoint = (start + end) / 2.0
        if float(np.dot(normal, center - midpoint)) < 0.0:
            normal = -normal
        t = np.linspace(0.08, 0.92, samples, dtype=np.float32)
        base = start[None, :] + t[:, None] * edge[None, :]
        inside = _bilinear(gray, base + normal[None, :] * offset)
        outside = _bilinear(gray, base - normal[None, :] * offset)
        delta = inside - outside
        signed.extend(delta.astype(float).tolist())
        side_contrast.append(float(np.mean(np.abs(delta))) / 255.0)
    if not signed:
        return 0.0, 0.0, side_contrast
    values = np.asarray(signed, dtype=np.float32)
    contrast = float(np.mean(np.abs(values))) / 255.0
    positive = float(np.mean(values >= 0.0)); negative = 1.0 - positive
    polarity = max(positive, negative)
    return contrast, polarity, side_contrast


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    """Validate contour geometry by sampling intensity transitions across every proposed edge."""
    values = _parameters(parameters)
    contour_parameters = {
        "minimum_contour_area_fraction": values["minimum_contour_area_fraction"],
        "epsilon_max_fraction": values["epsilon_max_fraction"],
        "minimum_rectangularity": values["minimum_rectangularity"],
    }
    contour = detector_contour_quad.detect(image_bgr=image_bgr, mask=mask, parameters=contour_parameters)
    if contour.status != "ok" or contour.corners is None:
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "no_contour_hypothesis", "contour": contour.diagnostics, "parameters": values}, status="no_candidate")
    gray = _gray(image_bgr)
    corners = np.asarray(contour.corners, dtype=np.float32).reshape(4, 2)
    offset = max(1.0, min(gray.shape) * values["sample_offset_fraction"])
    contrast, polarity, side_contrast = _cross_edge_evidence(gray, corners, offset, values["samples_per_edge"])
    diagnostics = {
        "parameters": values,
        "contour_score": float(contour.score),
        "cross_edge_contrast": contrast,
        "polarity_consistency": polarity,
        "side_contrast": side_contrast,
        "sample_offset_pixels": offset,
        "evidence": "inside_outside_cross_boundary_intensity",
    }
    if contrast < values["minimum_cross_edge_contrast"]:
        diagnostics["reason"] = "insufficient_cross_edge_contrast"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    if polarity < values["minimum_polarity_consistency"]:
        diagnostics["reason"] = "inconsistent_cross_edge_polarity"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    total = values["contour_weight"] + values["contrast_weight"] + values["polarity_weight"]
    contrast_score = min(1.0, contrast / max(values["minimum_cross_edge_contrast"] * 3.0, 1e-6))
    score = (float(contour.score) * values["contour_weight"] + contrast_score * values["contrast_weight"] + polarity * values["polarity_weight"]) / total
    diagnostics["contrast_score"] = contrast_score
    return Candidate(METHOD, contour.bbox, contour.corners, float(score), float(score), diagnostics)


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None, candidate_corners: list[list[float]] | None = None) -> dict[str, np.ndarray]:
    overlay = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if candidate_corners is not None:
        corners = np.asarray(candidate_corners, dtype=np.float32).reshape(4, 2)
        cv2.polylines(overlay, [np.rint(corners).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3, cv2.LINE_AA)
    return {"cross-edge-selected-contour.png": overlay}
