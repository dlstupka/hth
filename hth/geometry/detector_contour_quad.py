from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .model import Candidate

METHOD = "contour_quad"

BASELINE_PARAMETERS: dict[str, Any] = {
    "minimum_contour_area_fraction": 0.12,
    "close_kernel_fraction": 0.008,
    "close_iterations": 1,
    "epsilon_min_fraction": 0.008,
    "epsilon_max_fraction": 0.04,
    "epsilon_steps": 9,
    "minimum_rectangularity": 0.55,
    "area_weight": 0.35,
    "rectangularity_weight": 0.30,
    "angle_weight": 0.20,
    "edge_support_weight": 0.15,
    "edge_support_dilation_fraction": 0.004,
    "merge_fragmented_contours": True,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(
                f"Unknown Contour Quadrilateral parameters: {', '.join(unknown)}"
            )
        values.update(overrides)

    floats = (
        "minimum_contour_area_fraction",
        "close_kernel_fraction",
        "epsilon_min_fraction",
        "epsilon_max_fraction",
        "minimum_rectangularity",
        "area_weight",
        "rectangularity_weight",
        "angle_weight",
        "edge_support_weight",
        "edge_support_dilation_fraction",
    )
    for name in floats:
        values[name] = float(values[name])
    values["close_iterations"] = int(values["close_iterations"])
    values["epsilon_steps"] = int(values["epsilon_steps"])
    values["merge_fragmented_contours"] = bool(values["merge_fragmented_contours"])

    if not 0.0 <= values["minimum_contour_area_fraction"] <= 1.0:
        raise ValueError("minimum_contour_area_fraction must be between 0 and 1")
    if not 0.0 <= values["close_kernel_fraction"] <= 0.25:
        raise ValueError("close_kernel_fraction must be between 0 and 0.25")
    if values["close_iterations"] < 0:
        raise ValueError("close_iterations must be non-negative")
    if not 0.0 < values["epsilon_min_fraction"] <= 0.25:
        raise ValueError("epsilon_min_fraction must be greater than 0 and at most 0.25")
    if not values["epsilon_min_fraction"] <= values["epsilon_max_fraction"] <= 0.25:
        raise ValueError("epsilon_max_fraction must be between epsilon_min_fraction and 0.25")
    if values["epsilon_steps"] < 1:
        raise ValueError("epsilon_steps must be at least 1")
    if not 0.0 <= values["minimum_rectangularity"] <= 1.0:
        raise ValueError("minimum_rectangularity must be between 0 and 1")
    if not 0.0 <= values["edge_support_dilation_fraction"] <= 0.1:
        raise ValueError("edge_support_dilation_fraction must be between 0 and 0.1")

    weights = [
        values["area_weight"],
        values["rectangularity_weight"],
        values["angle_weight"],
        values["edge_support_weight"],
    ]
    if any(weight < 0.0 for weight in weights):
        raise ValueError("quadrilateral score weights must be non-negative")
    if sum(weights) <= 0.0:
        raise ValueError("at least one quadrilateral score weight must be positive")
    return values


def _odd_kernel_size(fraction: float, width: int, height: int) -> int:
    if fraction <= 0.0:
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
        if denominator <= 0.0:
            return 0.0
        cosine = abs(float(np.dot(previous, following)) / denominator)
        scores.append(max(0.0, 1.0 - min(1.0, cosine)))
    return float(sum(scores) / len(scores))


def _edge_support(image_bgr: np.ndarray, working: np.ndarray, corners: np.ndarray, dilation: int) -> float:
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    elif image_bgr.ndim == 2:
        gray = image_bgr
    else:
        raise ValueError(f"Contour quadrilateral detector expects a 2-D or 3-D image, got {image_bgr.shape}")

    image_edges = cv2.Canny(gray, 50, 150)
    mask_edges = cv2.Canny(working, 50, 150)
    evidence = cv2.bitwise_or(image_edges, mask_edges)

    polygon_edge = np.zeros_like(working)
    cv2.polylines(
        polygon_edge,
        [np.rint(corners).astype(np.int32).reshape(-1, 1, 2)],
        True,
        255,
        1,
        cv2.LINE_AA,
    )
    if dilation > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation, dilation))
        evidence = cv2.dilate(evidence, kernel, iterations=1)
    expected = int(np.count_nonzero(polygon_edge))
    if expected == 0:
        return 0.0
    return float(np.count_nonzero(cv2.bitwise_and(evidence, polygon_edge))) / expected


def detect(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> Candidate:
    """Detect a document quadrilateral using contour geometry plus edge evidence."""
    values = _parameters(parameters)
    if mask.ndim != 2:
        raise ValueError(
            f"Contour quadrilateral detector expects a 2-D mask, got shape {mask.shape}"
        )

    working = np.where(mask > 0, 255, 0).astype(np.uint8)
    height, width = working.shape
    image_area = float(width * height)

    close_kernel_size = _odd_kernel_size(values["close_kernel_fraction"], width, height)
    if close_kernel_size and values["close_iterations"]:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_kernel_size, close_kernel_size)
        )
        working = cv2.morphologyEx(
            working,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=values["close_iterations"],
        )

    contours, _ = cv2.findContours(working, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_hypotheses: list[tuple[np.ndarray, str]] = [
        (contour, "external_contour") for contour in contours
    ]
    if values["merge_fragmented_contours"] and contours:
        points = np.concatenate(contours, axis=0)
        if len(points) >= 3:
            contour_hypotheses.append((cv2.convexHull(points), "merged_convex_hull"))

    minimum_area = image_area * values["minimum_contour_area_fraction"]
    epsilons = np.linspace(
        values["epsilon_min_fraction"],
        values["epsilon_max_fraction"],
        values["epsilon_steps"],
    )
    dilation = _odd_kernel_size(values["edge_support_dilation_fraction"], width, height)
    weight_total = sum(
        values[name]
        for name in (
            "area_weight",
            "rectangularity_weight",
            "angle_weight",
            "edge_support_weight",
        )
    )

    best: dict[str, Any] | None = None
    quadrilateral_count = 0
    rejected_nonconvex = 0
    rejected_rectangularity = 0

    for contour, source in contour_hypotheses:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < minimum_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue

        for epsilon_fraction in epsilons:
            approx = cv2.approxPolyDP(contour, float(epsilon_fraction) * perimeter, True)
            if len(approx) != 4:
                continue
            quadrilateral_count += 1
            if not cv2.isContourConvex(approx):
                rejected_nonconvex += 1
                continue

            corners = _order_corners(approx.reshape(4, 2))
            quad_area = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2))))
            if quad_area <= 0.0:
                continue
            rectangularity = min(1.0, contour_area / quad_area)
            if rectangularity < values["minimum_rectangularity"]:
                rejected_rectangularity += 1
                continue

            area_score = min(1.0, quad_area / image_area)
            angle_score = _angle_score(corners)
            edge_support = _edge_support(image_bgr, working, corners, dilation)
            score = (
                area_score * values["area_weight"]
                + rectangularity * values["rectangularity_weight"]
                + angle_score * values["angle_weight"]
                + edge_support * values["edge_support_weight"]
            ) / weight_total

            x, y, bbox_width, bbox_height = cv2.boundingRect(
                np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
            )
            candidate = {
                "score": float(score),
                "bbox": [x, y, x + bbox_width, y + bbox_height],
                "corners": corners.astype(float).tolist(),
                "contour_source": source,
                "epsilon_fraction": float(epsilon_fraction),
                "contour_area": contour_area,
                "quad_area": quad_area,
                "area_score": area_score,
                "rectangularity": rectangularity,
                "angle_score": angle_score,
                "edge_support": edge_support,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    diagnostics: dict[str, Any] = {
        "parameters": values,
        "external_contour_count": len(contours),
        "contour_hypothesis_count": len(contour_hypotheses),
        "quadrilateral_count": quadrilateral_count,
        "rejected_nonconvex": rejected_nonconvex,
        "rejected_rectangularity": rejected_rectangularity,
        "close_kernel_size": close_kernel_size,
        "edge_support_dilation_size": dilation,
        "mask_foreground_fraction": round(float(np.count_nonzero(working)) / image_area, 8),
    }

    if best is None:
        diagnostics["reason"] = "no_plausible_quadrilateral"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    diagnostics.update(
        {
            "contour_source": best["contour_source"],
            "epsilon_fraction": round(best["epsilon_fraction"], 8),
            "contour_area": round(best["contour_area"], 3),
            "quadrilateral_area": round(best["quad_area"], 3),
            "area_score": round(best["area_score"], 8),
            "rectangularity": round(best["rectangularity"], 8),
            "angle_score": round(best["angle_score"], 8),
            "edge_support": round(best["edge_support"], 8),
        }
    )
    score = round(best["score"], 6)
    return Candidate(METHOD, best["bbox"], best["corners"], score, score, diagnostics)
