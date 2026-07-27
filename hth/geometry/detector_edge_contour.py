from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .model import Candidate

METHOD = "edge_contour"

BASELINE_PARAMETERS: dict[str, Any] = {
    "minimum_contour_area_fraction": 0.12,
    "close_kernel_fraction": 0.008,
    "close_iterations": 1,
    "epsilon_min_fraction": 0.008,
    "epsilon_max_fraction": 0.04,
    "epsilon_steps": 9,
    "minimum_rectangularity": 0.55,
    "lsd_refine_mode": "std",
    "lsd_scale": 0.8,
    "minimum_segment_length_fraction": 0.06,
    "edge_support_dilation_fraction": 0.006,
    "minimum_edge_support": 0.12,
    "area_weight": 0.25,
    "rectangularity_weight": 0.25,
    "angle_weight": 0.20,
    "edge_support_weight": 0.30,
    "merge_fragmented_contours": True,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown Edge-Contour parameters: {', '.join(unknown)}")
        values.update(overrides)

    for name in (
        "minimum_contour_area_fraction", "close_kernel_fraction",
        "epsilon_min_fraction", "epsilon_max_fraction", "minimum_rectangularity",
        "lsd_scale", "minimum_segment_length_fraction",
        "edge_support_dilation_fraction", "minimum_edge_support", "area_weight",
        "rectangularity_weight", "angle_weight", "edge_support_weight",
    ):
        values[name] = float(values[name])
    values["close_iterations"] = int(values["close_iterations"])
    values["epsilon_steps"] = int(values["epsilon_steps"])
    values["merge_fragmented_contours"] = bool(values["merge_fragmented_contours"])
    values["lsd_refine_mode"] = str(values["lsd_refine_mode"]).lower()

    if not 0.0 <= values["minimum_contour_area_fraction"] <= 1.0:
        raise ValueError("minimum_contour_area_fraction must be between 0 and 1")
    if not 0.0 <= values["close_kernel_fraction"] <= 0.25:
        raise ValueError("close_kernel_fraction must be between 0 and 0.25")
    if values["close_iterations"] < 0:
        raise ValueError("close_iterations must be non-negative")
    if not 0.0 < values["epsilon_min_fraction"] <= values["epsilon_max_fraction"] <= 0.25:
        raise ValueError("epsilon fractions must satisfy 0 < min <= max <= 0.25")
    if values["epsilon_steps"] < 1:
        raise ValueError("epsilon_steps must be at least 1")
    if not 0.0 <= values["minimum_rectangularity"] <= 1.0:
        raise ValueError("minimum_rectangularity must be between 0 and 1")
    if values["lsd_refine_mode"] not in {"none", "std", "adv"}:
        raise ValueError("lsd_refine_mode must be none, std, or adv")
    if not 0.1 <= values["lsd_scale"] <= 1.0:
        raise ValueError("lsd_scale must be between 0.1 and 1.0")
    if not 0.0 <= values["minimum_segment_length_fraction"] <= 1.0:
        raise ValueError("minimum_segment_length_fraction must be between 0 and 1")
    if not 0.0 <= values["edge_support_dilation_fraction"] <= 0.1:
        raise ValueError("edge_support_dilation_fraction must be between 0 and 0.1")
    if not 0.0 <= values["minimum_edge_support"] <= 1.0:
        raise ValueError("minimum_edge_support must be between 0 and 1")
    weights = [values[name] for name in ("area_weight", "rectangularity_weight", "angle_weight", "edge_support_weight")]
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("Edge-Contour score weights must be non-negative with at least one positive")
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
    return np.roll(ordered, -int(np.argmin(ordered[:, 0] + ordered[:, 1])), axis=0)


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
    return float(sum(scores) / 4.0)


def _lsd_evidence(image_bgr: np.ndarray, values: dict[str, Any]) -> tuple[np.ndarray, int, int]:
    if image_bgr.ndim == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    elif image_bgr.ndim == 2:
        gray = image_bgr
    else:
        raise ValueError(f"Edge-Contour detector expects a 2-D or 3-D image, got {image_bgr.shape}")
    refine = {"none": cv2.LSD_REFINE_NONE, "std": cv2.LSD_REFINE_STD, "adv": cv2.LSD_REFINE_ADV}[values["lsd_refine_mode"]]
    detector = cv2.createLineSegmentDetector(refine=refine, scale=values["lsd_scale"])
    detected = detector.detect(gray)[0]
    evidence = np.zeros_like(gray)
    total = 0
    retained = 0
    minimum_length = min(gray.shape) * values["minimum_segment_length_fraction"]
    if detected is not None:
        for raw in detected:
            x1, y1, x2, y2 = map(float, raw.reshape(4))
            total += 1
            if float(np.hypot(x2 - x1, y2 - y1)) < minimum_length:
                continue
            retained += 1
            cv2.line(evidence, (round(x1), round(y1)), (round(x2), round(y2)), 255, 1, cv2.LINE_AA)
    return evidence, total, retained


def _edge_support(evidence: np.ndarray, corners: np.ndarray, dilation: int) -> float:
    expanded = evidence
    if dilation > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation, dilation))
        expanded = cv2.dilate(evidence, kernel, iterations=1)
    expected = np.zeros_like(evidence)
    cv2.polylines(expected, [np.rint(corners).astype(np.int32).reshape(-1, 1, 2)], True, 255, 1, cv2.LINE_AA)
    count = int(np.count_nonzero(expected))
    return 0.0 if count == 0 else float(np.count_nonzero(cv2.bitwise_and(expanded, expected))) / count


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    """Generate contour quadrilaterals and verify them with independent LSD evidence."""
    values = _parameters(parameters)
    if mask.ndim != 2:
        raise ValueError(f"Edge-Contour detector expects a 2-D mask, got shape {mask.shape}")
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

    evidence, segment_count, retained_segment_count = _lsd_evidence(image_bgr, values)
    dilation = _odd_kernel_size(values["edge_support_dilation_fraction"], width, height)
    minimum_area = image_area * values["minimum_contour_area_fraction"]
    epsilons = np.linspace(values["epsilon_min_fraction"], values["epsilon_max_fraction"], values["epsilon_steps"])
    weight_total = sum(values[name] for name in ("area_weight", "rectangularity_weight", "angle_weight", "edge_support_weight"))
    best: dict[str, Any] | None = None
    quad_count = rejected_edge_support = 0

    for contour, source in hypotheses:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < minimum_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        for epsilon_fraction in epsilons:
            approx = cv2.approxPolyDP(contour, float(epsilon_fraction) * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad_count += 1
            corners = _order_corners(approx.reshape(4, 2))
            quad_area = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2))))
            if quad_area <= 0.0:
                continue
            rectangularity = min(1.0, contour_area / quad_area)
            if rectangularity < values["minimum_rectangularity"]:
                continue
            support = _edge_support(evidence, corners, dilation)
            if support < values["minimum_edge_support"]:
                rejected_edge_support += 1
                continue
            area_score = min(1.0, quad_area / image_area)
            angle_score = _angle_score(corners)
            score = (area_score * values["area_weight"] + rectangularity * values["rectangularity_weight"] + angle_score * values["angle_weight"] + support * values["edge_support_weight"]) / weight_total
            x, y, w, h = cv2.boundingRect(np.rint(corners).astype(np.int32).reshape(-1, 1, 2))
            candidate = {"score": score, "bbox": [x, y, x + w, y + h], "corners": corners.astype(float).tolist(), "source": source, "epsilon": float(epsilon_fraction), "area_score": area_score, "rectangularity": rectangularity, "angle_score": angle_score, "edge_support": support}
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    diagnostics: dict[str, Any] = {
        "parameters": values,
        "external_contour_count": len(contours),
        "contour_hypothesis_count": len(hypotheses),
        "quadrilateral_count": quad_count,
        "lsd_segment_count": segment_count,
        "retained_lsd_segment_count": retained_segment_count,
        "rejected_edge_support": rejected_edge_support,
        "close_kernel_size": close_size,
        "edge_support_dilation_size": dilation,
    }
    if best is None:
        diagnostics["reason"] = "no_edge_verified_quadrilateral"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    diagnostics.update({
        "contour_source": best["source"], "epsilon_fraction": round(best["epsilon"], 8),
        "area_score": round(best["area_score"], 8), "rectangularity": round(best["rectangularity"], 8),
        "angle_score": round(best["angle_score"], 8), "edge_support": round(best["edge_support"], 8),
    })
    score = round(float(best["score"]), 6)
    return Candidate(METHOD, best["bbox"], best["corners"], score, score, diagnostics)
