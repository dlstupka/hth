from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import detector_contour_quad
from .model import Candidate

METHOD = "border_energy"

BASELINE_PARAMETERS: dict[str, Any] = {
    "minimum_contour_area_fraction": 0.12,
    "epsilon_max_fraction": 0.04,
    "minimum_rectangularity": 0.55,
    "gaussian_sigma": 1.2,
    "band_fraction": 0.008,
    "minimum_border_energy": 0.10,
    "minimum_side_consistency": 0.45,
    "contour_weight": 0.45,
    "energy_weight": 0.40,
    "consistency_weight": 0.15,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown Border Energy Validator parameters: {', '.join(unknown)}")
        values.update(overrides)
    for name in values:
        values[name] = float(values[name])
    for name in ("minimum_contour_area_fraction", "minimum_rectangularity", "minimum_border_energy", "minimum_side_consistency"):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if values["gaussian_sigma"] < 0.0:
        raise ValueError("gaussian_sigma must be non-negative")
    if not 0.0 < values["epsilon_max_fraction"] <= 0.25:
        raise ValueError("epsilon_max_fraction must be greater than 0 and at most 0.25")
    if not 0.0 < values["band_fraction"] <= 0.1:
        raise ValueError("band_fraction must be greater than 0 and at most 0.1")
    weights = [values["contour_weight"], values["energy_weight"], values["consistency_weight"]]
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("Border-energy score weights must be non-negative with at least one positive")
    return values


def _gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.ndim == 2:
        return image_bgr
    raise ValueError(f"Border Energy Validator expects a 2-D or 3-D image, got {image_bgr.shape}")


def _energy(gray: np.ndarray, corners: np.ndarray, sigma: float, band_width: int) -> tuple[float, float, list[float], np.ndarray]:
    working = cv2.GaussianBlur(gray, (0, 0), sigma) if sigma > 0 else gray
    gx = cv2.Sobel(working, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(working, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    scale = float(np.percentile(magnitude, 95.0))
    normalized = np.clip(magnitude / max(scale, 1e-6), 0.0, 1.0)
    side_energy: list[float] = []
    for index in range(4):
        edge_mask = np.zeros(gray.shape, dtype=np.uint8)
        start = tuple(np.rint(corners[index]).astype(int))
        end = tuple(np.rint(corners[(index + 1) % 4]).astype(int))
        cv2.line(edge_mask, start, end, 255, max(1, band_width), cv2.LINE_AA)
        selected = normalized[edge_mask > 0]
        side_energy.append(float(np.mean(selected)) if selected.size else 0.0)
    overall = float(np.mean(side_energy))
    consistency = 0.0 if overall <= 1e-9 else float(np.min(side_energy) / overall)
    return overall, min(1.0, consistency), side_energy, normalized


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    """Validate contour geometry by measuring gradient energy along all proposed borders."""
    values = _parameters(parameters)
    contour = detector_contour_quad.detect(
        image_bgr=image_bgr,
        mask=mask,
        parameters={
            "minimum_contour_area_fraction": values["minimum_contour_area_fraction"],
            "epsilon_max_fraction": values["epsilon_max_fraction"],
            "minimum_rectangularity": values["minimum_rectangularity"],
        },
    )
    if contour.status != "ok" or contour.corners is None:
        return Candidate(METHOD, None, None, 0.0, 0.0, {"reason": "no_contour_hypothesis", "contour": contour.diagnostics, "parameters": values}, status="no_candidate")
    gray = _gray(image_bgr)
    corners = np.asarray(contour.corners, dtype=np.float32).reshape(4, 2)
    band_width = max(1, int(round(min(gray.shape) * values["band_fraction"])))
    energy, consistency, side_energy, _normalized = _energy(gray, corners, values["gaussian_sigma"], band_width)
    diagnostics = {
        "parameters": values,
        "contour_score": float(contour.score),
        "border_energy": energy,
        "side_consistency": consistency,
        "side_energy": side_energy,
        "band_width_pixels": band_width,
        "evidence": "sobel_energy_along_candidate_borders",
    }
    if energy < values["minimum_border_energy"]:
        diagnostics["reason"] = "insufficient_border_energy"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    if consistency < values["minimum_side_consistency"]:
        diagnostics["reason"] = "inconsistent_border_energy"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    total = values["contour_weight"] + values["energy_weight"] + values["consistency_weight"]
    score = (float(contour.score) * values["contour_weight"] + energy * values["energy_weight"] + consistency * values["consistency_weight"]) / total
    return Candidate(METHOD, contour.bbox, contour.corners, float(score), float(score), diagnostics)


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None, candidate_corners: list[list[float]] | None = None, verbose: bool = False) -> dict[str, np.ndarray]:
    del mask
    values = _parameters(parameters)
    gray = _gray(image_bgr)
    overlay = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    energy_image = np.zeros_like(gray)
    if candidate_corners is not None:
        corners = np.asarray(candidate_corners, dtype=np.float32).reshape(4, 2)
        band_width = max(1, int(round(min(gray.shape) * values["band_fraction"])))
        _energy_value, _consistency, _side_energy, normalized = _energy(gray, corners, values["gaussian_sigma"], band_width)
        energy_image = np.rint(normalized * 255.0).astype(np.uint8)
        cv2.polylines(overlay, [np.rint(corners).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3, cv2.LINE_AA)
    images = {"border-energy.png": energy_image, "validated-border.png": overlay}
    if verbose and candidate_corners is not None:
        corners = np.asarray(candidate_corners, dtype=np.float32).reshape(4, 2)
        band_width = max(1, int(round(min(gray.shape) * values["band_fraction"])))
        _energy_value, _consistency, side_energy, _normalized = _energy(gray, corners, values["gaussian_sigma"], band_width)
        band_view = overlay.copy()
        score_view = overlay.copy()
        labels = ("top", "right", "bottom", "left")
        palette = ((0,255,0),(0,255,255),(0,165,255),(0,0,255))
        for index in range(4):
            start = tuple(np.rint(corners[index]).astype(int))
            end = tuple(np.rint(corners[(index + 1) % 4]).astype(int))
            cv2.line(band_view, start, end, palette[index], max(1, band_width), cv2.LINE_AA)
            midpoint = tuple(np.rint((corners[index] + corners[(index + 1) % 4]) / 2.0).astype(int))
            cv2.putText(score_view, f"{labels[index]}={side_energy[index]:.3f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.55, palette[index], 2, cv2.LINE_AA)
        images["border-sampling-bands.png"] = band_view
        images["side-energy-scores.png"] = score_view
    return images


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
