from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .model import Candidate

METHOD = "contour_projection"

BASELINE_PARAMETERS: dict[str, Any] = {
    "minimum_contour_area_fraction": 0.12,
    "close_kernel_fraction": 0.008,
    "close_iterations": 1,
    "epsilon_min_fraction": 0.008,
    "epsilon_max_fraction": 0.04,
    "epsilon_steps": 9,
    "minimum_rectangularity": 0.55,
    "projection_margin_fraction": 0.06,
    "projection_threshold_block_fraction": 0.08,
    "projection_threshold_c": 9.0,
    "minimum_projection_score": 0.08,
    "area_weight": 0.25,
    "rectangularity_weight": 0.25,
    "angle_weight": 0.20,
    "projection_weight": 0.30,
    "merge_fragmented_contours": True,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown Contour + Projection parameters: {', '.join(unknown)}")
        values.update(overrides)
    for name in (
        "minimum_contour_area_fraction", "close_kernel_fraction",
        "epsilon_min_fraction", "epsilon_max_fraction", "minimum_rectangularity",
        "projection_margin_fraction", "projection_threshold_block_fraction",
        "projection_threshold_c", "minimum_projection_score", "area_weight",
        "rectangularity_weight", "angle_weight", "projection_weight",
    ):
        values[name] = float(values[name])
    values["close_iterations"] = int(values["close_iterations"])
    values["epsilon_steps"] = int(values["epsilon_steps"])
    values["merge_fragmented_contours"] = bool(values["merge_fragmented_contours"])
    if not 0 <= values["minimum_contour_area_fraction"] <= 1:
        raise ValueError("minimum_contour_area_fraction must be between 0 and 1")
    if not 0 <= values["close_kernel_fraction"] <= 0.25:
        raise ValueError("close_kernel_fraction must be between 0 and 0.25")
    if values["close_iterations"] < 0:
        raise ValueError("close_iterations must be non-negative")
    if not 0 < values["epsilon_min_fraction"] <= values["epsilon_max_fraction"] <= 0.25:
        raise ValueError("epsilon fractions must satisfy 0 < minimum <= maximum <= 0.25")
    if values["epsilon_steps"] < 1:
        raise ValueError("epsilon_steps must be at least 1")
    if not 0 <= values["minimum_rectangularity"] <= 1:
        raise ValueError("minimum_rectangularity must be between 0 and 1")
    if not 0 <= values["projection_margin_fraction"] < 0.5:
        raise ValueError("projection_margin_fraction must be between 0 and 0.5")
    if not 0 < values["projection_threshold_block_fraction"] <= 0.5:
        raise ValueError("projection_threshold_block_fraction must be greater than 0 and at most 0.5")
    if not 0 <= values["minimum_projection_score"] <= 1:
        raise ValueError("minimum_projection_score must be between 0 and 1")
    weights = [values[k] for k in ("area_weight", "rectangularity_weight", "angle_weight", "projection_weight")]
    if any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("score weights must be non-negative with a positive sum")
    return values


def _odd_kernel_size(fraction: float, width: int, height: int) -> int:
    if fraction <= 0:
        return 0
    size = max(3, int(round(min(width, height) * fraction)))
    return size if size % 2 else size + 1


def _order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


def _angle_score(corners: np.ndarray) -> float:
    scores: list[float] = []
    for index in range(4):
        previous = corners[(index - 1) % 4] - corners[index]
        following = corners[(index + 1) % 4] - corners[index]
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(following))
        if denominator <= 0:
            return 0.0
        cosine = abs(float(np.dot(previous, following)) / denominator)
        scores.append(max(0.0, 1.0 - min(1.0, cosine)))
    return float(np.mean(scores))


def _warp_gray(image_bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    top = np.linalg.norm(corners[1] - corners[0])
    bottom = np.linalg.norm(corners[2] - corners[3])
    left = np.linalg.norm(corners[3] - corners[0])
    right = np.linalg.norm(corners[2] - corners[1])
    width = max(32, int(round(max(top, bottom))))
    height = max(32, int(round(max(left, right))))
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    return cv2.warpPerspective(gray, transform, (width, height), flags=cv2.INTER_LINEAR)


def _projection_evidence(image_bgr: np.ndarray, corners: np.ndarray, values: dict[str, Any]) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
    warped = _warp_gray(image_bgr, corners)
    margin_y = int(round(warped.shape[0] * values["projection_margin_fraction"]))
    margin_x = int(round(warped.shape[1] * values["projection_margin_fraction"]))
    interior = warped[margin_y:warped.shape[0] - margin_y or None, margin_x:warped.shape[1] - margin_x or None]
    if interior.size == 0 or min(interior.shape) < 8:
        empty = np.zeros_like(warped)
        return 0.0, {"horizontal_band_score": 0.0, "vertical_coverage_score": 0.0, "ink_fraction": 0.0}, warped, empty
    block = _odd_kernel_size(values["projection_threshold_block_fraction"], interior.shape[1], interior.shape[0])
    block = max(3, block)
    binary = cv2.adaptiveThreshold(interior, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, values["projection_threshold_c"])
    row_density = np.mean(binary > 0, axis=1).astype(np.float32)
    col_density = np.mean(binary > 0, axis=0).astype(np.float32)
    ink_fraction = float(np.mean(binary > 0))
    row_mean = float(np.mean(row_density))
    row_std = float(np.std(row_density))
    horizontal_band_score = min(1.0, row_std / max(row_mean, 0.015))
    active_columns = float(np.mean(col_density > max(0.01, float(np.percentile(col_density, 35)))))
    vertical_coverage_score = min(1.0, active_columns / 0.75)
    ink_score = max(0.0, 1.0 - abs(ink_fraction - 0.16) / 0.16)
    score = float(0.55 * horizontal_band_score + 0.30 * vertical_coverage_score + 0.15 * ink_score)
    projection = np.zeros((max(96, binary.shape[0]), binary.shape[1]), dtype=np.uint8)
    normalized = row_density / max(float(np.max(row_density)), 1e-9)
    for y, fraction in enumerate(normalized):
        cv2.line(projection, (0, y), (int(round(fraction * (projection.shape[1] - 1))), y), 255, 1)
    metrics = {
        "horizontal_band_score": horizontal_band_score,
        "vertical_coverage_score": vertical_coverage_score,
        "ink_fraction": ink_fraction,
    }
    return score, metrics, warped, binary


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    values = _parameters(parameters)
    if mask.ndim != 2:
        raise ValueError(f"Contour + Projection detector expects a 2-D mask, got shape {mask.shape}")
    working = np.where(mask > 0, 255, 0).astype(np.uint8)
    height, width = working.shape
    image_area = float(width * height)
    close_size = _odd_kernel_size(values["close_kernel_fraction"], width, height)
    if close_size and values["close_iterations"]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        working = cv2.morphologyEx(working, cv2.MORPH_CLOSE, kernel, iterations=values["close_iterations"])
    contours, _ = cv2.findContours(working, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hypotheses: list[tuple[np.ndarray, str]] = [(contour, "external_contour") for contour in contours]
    if values["merge_fragmented_contours"] and contours:
        points = np.concatenate(contours, axis=0)
        if len(points) >= 3:
            hypotheses.append((cv2.convexHull(points), "merged_convex_hull"))
    minimum_area = image_area * values["minimum_contour_area_fraction"]
    epsilons = np.linspace(values["epsilon_min_fraction"], values["epsilon_max_fraction"], values["epsilon_steps"])
    weight_total = sum(values[name] for name in ("area_weight", "rectangularity_weight", "angle_weight", "projection_weight"))
    best: dict[str, Any] | None = None
    quadrilateral_count = rejected_projection = rejected_rectangularity = 0
    for contour, source in hypotheses:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < minimum_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        for epsilon_fraction in epsilons:
            approx = cv2.approxPolyDP(contour, float(epsilon_fraction) * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quadrilateral_count += 1
            corners = _order_corners(approx.reshape(4, 2))
            quad_area = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2))))
            if quad_area <= 0:
                continue
            rectangularity = min(1.0, contour_area / quad_area)
            if rectangularity < values["minimum_rectangularity"]:
                rejected_rectangularity += 1
                continue
            projection_score, projection_metrics, _, _ = _projection_evidence(image_bgr, corners, values)
            if projection_score < values["minimum_projection_score"]:
                rejected_projection += 1
                continue
            area_score = min(1.0, quad_area / image_area)
            angle_score = _angle_score(corners)
            score = (area_score * values["area_weight"] + rectangularity * values["rectangularity_weight"] + angle_score * values["angle_weight"] + projection_score * values["projection_weight"]) / weight_total
            x, y, w, h = cv2.boundingRect(np.rint(corners).astype(np.int32).reshape(-1, 1, 2))
            candidate = {"score": score, "bbox": [x, y, x + w, y + h], "corners": corners.astype(float).tolist(), "source": source, "epsilon": float(epsilon_fraction), "area_score": area_score, "rectangularity": rectangularity, "angle_score": angle_score, "projection_score": projection_score, **projection_metrics}
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    diagnostics: dict[str, Any] = {
        "parameters": values, "external_contour_count": len(contours),
        "contour_hypothesis_count": len(hypotheses), "quadrilateral_count": quadrilateral_count,
        "rejected_rectangularity": rejected_rectangularity, "rejected_projection": rejected_projection,
        "close_kernel_size": close_size,
    }
    if best is None:
        diagnostics["reason"] = "no_projection_verified_quadrilateral"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    diagnostics.update({
        "contour_source": best["source"], "epsilon_fraction": round(best["epsilon"], 8),
        "area_score": round(best["area_score"], 8), "rectangularity": round(best["rectangularity"], 8),
        "angle_score": round(best["angle_score"], 8), "projection_score": round(best["projection_score"], 8),
        "horizontal_band_score": round(best["horizontal_band_score"], 8),
        "vertical_coverage_score": round(best["vertical_coverage_score"], 8),
        "ink_fraction": round(best["ink_fraction"], 8),
    })
    score = round(float(best["score"]), 6)
    return Candidate(METHOD, best["bbox"], best["corners"], score, score, diagnostics)


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None, candidate_corners: list[list[float]] | None = None) -> dict[str, np.ndarray]:
    values = _parameters(parameters)
    working = np.where(mask > 0, 255, 0).astype(np.uint8)
    contour_image = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(working, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 255), 2)
    selected = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if candidate_corners is None:
        blank = np.zeros_like(working)
        return {"contour-hypotheses.png": contour_image, "warped-candidate.png": blank, "projection-binary.png": blank, "horizontal-projection.png": blank, "selected-quadrilateral.png": selected}
    corners = np.asarray(candidate_corners, dtype=np.float32).reshape(4, 2)
    _, _, warped, binary = _projection_evidence(image_bgr, corners, values)
    row_density = np.mean(binary > 0, axis=1).astype(np.float32) if binary.size else np.zeros(1, dtype=np.float32)
    projection = np.zeros_like(binary)
    if projection.size:
        normalized = row_density / max(float(np.max(row_density)), 1e-9)
        for y, fraction in enumerate(normalized):
            cv2.line(projection, (0, y), (int(round(fraction * (projection.shape[1] - 1))), y), 255, 1)
    cv2.polylines(selected, [np.rint(corners).astype(np.int32)], True, (0, 0, 255), 4, cv2.LINE_AA)
    return {"contour-hypotheses.png": contour_image, "warped-candidate.png": warped, "projection-binary.png": binary, "horizontal-projection.png": projection, "selected-quadrilateral.png": selected}


__all__ = ["BASELINE_PARAMETERS", "METHOD", "debug_images", "detect"]
