from __future__ import annotations

from itertools import product
from typing import Any

import cv2
import numpy as np

from . import (
    detector_border_fusion_quad,
    detector_multi_scale_radial_edge,
    detector_page_background,
    detector_signed_polar_boundary_vote,
)
from .model import Candidate

METHOD = "msre_bfq_spbv_pbg"

CHILD_CALIBRATIONS: dict[str, dict[str, Any]] = {
    "multi_scale_radial_edge": {
        "parameter_set_id": "ddb7623ebb92",
        "parameters": {
            "base_sigma": 1.0,
            "scale_ratio": 3.5,
            "scale_count": 4,
            "ray_count": 176,
            "gradient_percentile": 96.875,
        },
    },
    "border_fusion_quad": {
        "parameter_set_id": "2370e6cea486",
        "parameters": {
            "gradient_percentile": 68,
            "minimum_side_gradient_support": 0.06,
            "minimum_area_fraction": 0.08,
            "gradient_weight": 0.15,
            "source_confidence_weight": 0.4,
            "source_diversity_weight": 0.0,
        },
    },
    "signed_polar_boundary_vote": {
        "parameter_set_id": "8ddbe5f468cd",
        "parameters": {
            "ray_count": 72,
            "inner_radius_fraction": 0.2,
            "outer_radius_fraction": 0.6,
            "gradient_percentile": 95.0,
            "minimum_support_fraction": 0.35,
            "bbox_padding_fraction": 0.0,
            "polarity": "absolute",
        },
    },
    "page_background": {
        "parameter_set_id": "afbe81a796a1",
        "parameters": {
            "border_band_fraction": 0.015,
            "color_distance_threshold": 11.5,
            "blur_sigma": 0.0,
            "close_kernel_fraction": 0.0,
            "open_kernel_fraction": 0.0035,
            "minimum_border_background_fraction": 0.15,
            "minimum_page_area_fraction": 0.15,
        },
    },
}

CHILDREN = (
    ("multi_scale_radial_edge", detector_multi_scale_radial_edge.detect),
    ("border_fusion_quad", detector_border_fusion_quad.detect),
    ("signed_polar_boundary_vote", detector_signed_polar_boundary_vote.detect),
    ("page_background", detector_page_background.detect),
)

BASELINE_PARAMETERS: dict[str, Any] = {
    "gradient_percentile": 76.0,
    "minimum_side_gradient_support": 0.03,
    "consensus_tolerance_fraction": 0.012,
    "minimum_side_consensus": 0.50,
    "consensus_weight": 0.60,
    "gradient_weight": 0.25,
    "source_diversity_weight": 0.15,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, float]:
    values = dict(BASELINE_PARAMETERS)
    overrides = overrides or {}
    unknown = sorted(set(overrides) - set(values))
    if unknown:
        raise ValueError(f"Unknown Fusion Gen1 parameters: {', '.join(unknown)}")
    values.update(overrides)
    values = {name: float(value) for name, value in values.items()}
    if not 0.0 <= values["gradient_percentile"] <= 100.0:
        raise ValueError("gradient_percentile must be between 0 and 100")
    for name in ("minimum_side_gradient_support", "minimum_side_consensus"):
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if values["consensus_tolerance_fraction"] <= 0.0:
        raise ValueError("consensus_tolerance_fraction must be positive")
    weights = [values["consensus_weight"], values["gradient_weight"], values["source_diversity_weight"]]
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("fusion score weights must be non-negative with at least one positive")
    return values


def _gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 3:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if image_bgr.ndim == 2:
        return image_bgr
    raise ValueError(f"Fusion Gen1 expects a 2-D or 3-D image, got {image_bgr.shape}")


def _order(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered[:, 0] + ordered[:, 1])), axis=0)


def _line(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    x1, y1 = map(float, first)
    x2, y2 = map(float, second)
    line = np.array([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dtype=np.float64)
    norm = float(np.hypot(line[0], line[1]))
    return line / max(norm, 1e-9)


def _intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    point = np.cross(first, second)
    if abs(float(point[2])) < 1e-8:
        return None
    return (point[:2] / point[2]).astype(np.float32)


def _sides(corners: np.ndarray) -> list[np.ndarray]:
    return [
        _line(corners[0], corners[1]),
        _line(corners[1], corners[2]),
        _line(corners[2], corners[3]),
        _line(corners[3], corners[0]),
    ]


def _child_parameters(method: str) -> dict[str, Any]:
    module = {
        "multi_scale_radial_edge": detector_multi_scale_radial_edge,
        "border_fusion_quad": detector_border_fusion_quad,
        "signed_polar_boundary_vote": detector_signed_polar_boundary_vote,
        "page_background": detector_page_background,
    }[method]
    values = dict(module.BASELINE_PARAMETERS)
    values.update(CHILD_CALIBRATIONS[method]["parameters"])
    return values


def _candidate_summary(candidate: Candidate, method: str) -> dict[str, Any]:
    return {
        "method": method,
        "parameter_set_id": CHILD_CALIBRATIONS[method]["parameter_set_id"],
        "status": candidate.status,
        "bbox": candidate.bbox,
        "corners": candidate.corners,
        "confidence": float(candidate.confidence),
        "score": float(candidate.score),
        "diagnostics": candidate.diagnostics,
    }


def _side_gradient_support(magnitude: np.ndarray, first: np.ndarray, second: np.ndarray, threshold: float) -> float:
    count = max(8, int(round(float(np.linalg.norm(second - first)))))
    xs = np.clip(np.rint(np.linspace(first[0], second[0], count)).astype(int), 0, magnitude.shape[1] - 1)
    ys = np.clip(np.rint(np.linspace(first[1], second[1], count)).astype(int), 0, magnitude.shape[0] - 1)
    return float(np.mean(np.clip(magnitude[ys, xs] / max(threshold, 1e-6), 0.0, 1.0)))


def _line_disagreement(reference_line: np.ndarray, first: np.ndarray, second: np.ndarray, diagonal: float) -> float:
    sample_count = 9
    points = np.column_stack((
        np.linspace(first[0], second[0], sample_count),
        np.linspace(first[1], second[1], sample_count),
        np.ones(sample_count),
    ))
    return float(np.mean(np.abs(points @ reference_line)) / max(diagonal, 1.0))


def _run(image_bgr: np.ndarray, mask: np.ndarray, values: dict[str, float]) -> dict[str, Any]:
    gray = _gray(image_bgr)
    h, w = gray.shape
    diagonal = float(np.hypot(w, h))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    gradient_threshold = float(np.percentile(magnitude, values["gradient_percentile"]))

    child_results: list[dict[str, Any]] = []
    available: list[dict[str, Any]] = []
    for method, detector in CHILDREN:
        candidate = detector(image_bgr=image_bgr, mask=mask, parameters=_child_parameters(method))
        child_results.append(_candidate_summary(candidate, method))
        if candidate.status != "ok" or candidate.corners is None:
            continue
        corners = _order(np.asarray(candidate.corners, dtype=np.float32))
        available.append({"method": method, "candidate": candidate, "corners": corners, "sides": _sides(corners)})

    result = {
        "magnitude": magnitude,
        "gradient_threshold": gradient_threshold,
        "children": child_results,
        "available": available,
        "evaluated_combinations": 0,
        "best": None,
    }
    if len(available) < 2:
        return result

    best: dict[str, Any] | None = None
    tolerance = values["consensus_tolerance_fraction"]
    for source_indices in product(range(len(available)), repeat=4):
        if len(set(source_indices)) < 2:
            continue

        selected_lines = [
            available[source_indices[0]]["sides"][0],
            available[source_indices[1]]["sides"][1],
            available[source_indices[2]]["sides"][2],
            available[source_indices[3]]["sides"][3],
        ]
        top, right, bottom, left = selected_lines
        intersections = [
            _intersection(left, top),
            _intersection(top, right),
            _intersection(right, bottom),
            _intersection(bottom, left),
        ]
        if any(point is None for point in intersections):
            continue
        corners = _order(np.asarray(intersections, dtype=np.float32))
        if not np.all(np.isfinite(corners)):
            continue

        area_fraction = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2)))) / float(w * h)
        if not 0.10 <= area_fraction <= 0.995:
            continue
        if (
            np.any(corners[:, 0] < -0.10 * w)
            or np.any(corners[:, 0] > 1.10 * w)
            or np.any(corners[:, 1] < -0.10 * h)
            or np.any(corners[:, 1] > 1.10 * h)
        ):
            continue

        side_consensus: list[float] = []
        gradient_support: list[float] = []
        for side_index in range(4):
            first = corners[side_index]
            second = corners[(side_index + 1) % 4]
            agreements = []
            for child in available:
                disagreement = _line_disagreement(child["sides"][side_index], first, second, diagonal)
                agreements.append(float(np.exp(-((disagreement / tolerance) ** 2))))
            side_consensus.append(float(np.mean(agreements)))
            gradient_support.append(_side_gradient_support(magnitude, first, second, gradient_threshold))

        if min(side_consensus) < values["minimum_side_consensus"]:
            continue
        if min(gradient_support) < values["minimum_side_gradient_support"]:
            continue

        consensus = float(np.mean(side_consensus))
        gradient = float(np.mean(gradient_support))
        diversity = len(set(source_indices)) / float(min(4, len(available)))
        total_weight = values["consensus_weight"] + values["gradient_weight"] + values["source_diversity_weight"]
        score = (
            consensus * values["consensus_weight"]
            + gradient * values["gradient_weight"]
            + diversity * values["source_diversity_weight"]
        ) / total_weight
        result["evaluated_combinations"] += 1

        if best is None or score > best["score"]:
            best = {
                "corners": corners,
                "score": float(score),
                "source_indices": list(source_indices),
                "source_methods": [available[index]["method"] for index in source_indices],
                "side_consensus": side_consensus,
                "gradient_support": gradient_support,
                "mean_consensus": consensus,
                "mean_gradient_support": gradient,
                "source_diversity": diversity,
                "area_fraction": area_fraction,
            }

    result["best"] = best
    return result


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    values = _parameters(parameters)
    run = _run(image_bgr, mask, values)
    diagnostics: dict[str, Any] = {
        "parameters": values,
        "child_calibrations": CHILD_CALIBRATIONS,
        "children": run["children"],
        "available_child_candidates": len(run["available"]),
        "evaluated_side_combinations": run["evaluated_combinations"],
        "evidence": "calibrated_cross_family_side_hypothesis_consensus",
    }
    if len(run["available"]) < 2:
        diagnostics["reason"] = "insufficient_child_hypotheses"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    best = run["best"]
    if best is None:
        diagnostics["reason"] = "no_consensus_fused_quadrilateral"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    diagnostics.update({
        "selected_side_sources": best["source_methods"],
        "side_consensus": best["side_consensus"],
        "side_gradient_support": best["gradient_support"],
        "mean_consensus": best["mean_consensus"],
        "mean_gradient_support": best["mean_gradient_support"],
        "source_diversity": best["source_diversity"],
        "area_fraction": best["area_fraction"],
    })
    corners = best["corners"]
    x1, y1 = np.floor(corners.min(axis=0)).astype(int)
    x2, y2 = np.ceil(corners.max(axis=0)).astype(int)
    h, w = _gray(image_bgr).shape
    bbox = [int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))]
    return Candidate(METHOD, bbox, corners.astype(float).tolist(), best["score"], best["score"], diagnostics)


def debug_images(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None, candidate_corners: list[list[float]] | None = None, verbose: bool = False) -> dict[str, np.ndarray]:
    values = _parameters(parameters)
    run = _run(image_bgr, mask, values)
    gradient = cv2.normalize(run["magnitude"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    base = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]

    child_view = base.copy()
    for index, child in enumerate(run["available"]):
        cv2.polylines(child_view, [np.rint(child["corners"]).astype(np.int32).reshape(-1, 1, 2)], True, colors[index % len(colors)], 2, cv2.LINE_AA)

    selected_view = child_view.copy()
    if candidate_corners is not None:
        cv2.polylines(selected_view, [np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 4, cv2.LINE_AA)

    images = {
        "fusion-gen1-gradient.png": gradient,
        "fusion-gen1-child-quads.png": child_view,
        "fusion-gen1-selected-quad.png": selected_view,
    }
    if verbose and run["best"] is not None:
        side_view = base.copy()
        corners = run["best"]["corners"]
        for side_index in range(4):
            source_index = run["best"]["source_indices"][side_index]
            cv2.line(side_view, tuple(np.rint(corners[side_index]).astype(int)), tuple(np.rint(corners[(side_index + 1) % 4]).astype(int)), colors[source_index % len(colors)], 5, cv2.LINE_AA)
            cv2.putText(
                side_view,
                f"{side_index}:{run['best']['source_methods'][side_index]}",
                tuple(np.rint((corners[side_index] + corners[(side_index + 1) % 4]) / 2).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colors[source_index % len(colors)],
                1,
                cv2.LINE_AA,
            )
        images["fusion-gen1-side-sources.png"] = side_view
    return images


__all__ = ["BASELINE_PARAMETERS", "CHILD_CALIBRATIONS", "METHOD", "debug_images", "detect"]
