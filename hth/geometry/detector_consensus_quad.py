from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import detector_contour_quad, detector_edge_contour
from .model import Candidate

METHOD = "consensus_quad"

BASELINE_PARAMETERS: dict[str, Any] = {
    "minimum_polygon_iou": 0.90,
    "maximum_mean_corner_distance_fraction": 0.025,
    "contour_quad_weight": 0.50,
    "edge_contour_weight": 0.50,
    "minimum_consensus_confidence": 0.20,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(f"Unknown Consensus Quad parameters: {', '.join(unknown)}")
        values.update(overrides)
    for name in values:
        values[name] = float(values[name])
    if not 0.0 <= values["minimum_polygon_iou"] <= 1.0:
        raise ValueError("minimum_polygon_iou must be between 0 and 1")
    if not 0.0 <= values["maximum_mean_corner_distance_fraction"] <= 1.0:
        raise ValueError("maximum_mean_corner_distance_fraction must be between 0 and 1")
    if not 0.0 <= values["minimum_consensus_confidence"] <= 1.0:
        raise ValueError("minimum_consensus_confidence must be between 0 and 1")
    weights = (values["contour_quad_weight"], values["edge_contour_weight"])
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("consensus weights must be non-negative with a positive sum")
    return values


def _corners(candidate: Candidate) -> np.ndarray | None:
    if candidate.status != "ok" or candidate.corners is None:
        return None
    points = np.asarray(candidate.corners, dtype=np.float32)
    if points.shape != (4, 2):
        return None
    return points


def _polygon_iou(first: np.ndarray, second: np.ndarray, shape: tuple[int, int]) -> float:
    height, width = shape
    first_mask = np.zeros((height, width), dtype=np.uint8)
    second_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(first_mask, np.rint(first).astype(np.int32), 255)
    cv2.fillConvexPoly(second_mask, np.rint(second).astype(np.int32), 255)
    intersection = int(np.count_nonzero(cv2.bitwise_and(first_mask, second_mask)))
    union = int(np.count_nonzero(cv2.bitwise_or(first_mask, second_mask)))
    return float(intersection / union) if union else 0.0


def _candidate_summary(candidate: Candidate) -> dict[str, Any]:
    return {
        "method": candidate.method,
        "status": candidate.status,
        "bbox": candidate.bbox,
        "corners": candidate.corners,
        "confidence": candidate.confidence,
        "score": candidate.score,
        "diagnostics": candidate.diagnostics,
    }


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    values = _parameters(parameters)
    contour_quad = detector_contour_quad.detect(image_bgr=image_bgr, mask=mask)
    edge_contour = detector_edge_contour.detect(image_bgr=image_bgr, mask=mask)
    cq_corners = _corners(contour_quad)
    ec_corners = _corners(edge_contour)

    diagnostics: dict[str, Any] = {
        "parameters": values,
        "voters": {
            "contour_quad": _candidate_summary(contour_quad),
            "edge_contour": _candidate_summary(edge_contour),
        },
        "available_votes": int(cq_corners is not None) + int(ec_corners is not None),
    }
    if cq_corners is None or ec_corners is None:
        diagnostics["reason"] = "insufficient_quad_votes"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    height, width = mask.shape[:2]
    diagonal = float(np.hypot(width, height))
    polygon_iou = _polygon_iou(cq_corners, ec_corners, (height, width))
    corner_distances = np.linalg.norm(cq_corners - ec_corners, axis=1)
    mean_corner_distance_fraction = float(np.mean(corner_distances) / diagonal) if diagonal else 1.0
    diagnostics.update({
        "polygon_iou": polygon_iou,
        "mean_corner_distance_px": float(np.mean(corner_distances)),
        "maximum_corner_distance_px": float(np.max(corner_distances)),
        "mean_corner_distance_fraction": mean_corner_distance_fraction,
    })

    if polygon_iou < values["minimum_polygon_iou"]:
        diagnostics["reason"] = "polygon_iou_below_minimum"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")
    if mean_corner_distance_fraction > values["maximum_mean_corner_distance_fraction"]:
        diagnostics["reason"] = "corner_distance_above_maximum"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    cq_weight = values["contour_quad_weight"] * max(0.0, float(contour_quad.confidence))
    ec_weight = values["edge_contour_weight"] * max(0.0, float(edge_contour.confidence))
    if cq_weight + ec_weight <= 0.0:
        cq_weight = values["contour_quad_weight"]
        ec_weight = values["edge_contour_weight"]
    consensus = (cq_corners * cq_weight + ec_corners * ec_weight) / (cq_weight + ec_weight)
    x1, y1 = np.floor(consensus.min(axis=0)).astype(int)
    x2, y2 = np.ceil(consensus.max(axis=0)).astype(int)
    agreement = 0.5 * polygon_iou + 0.5 * max(0.0, 1.0 - mean_corner_distance_fraction / max(values["maximum_mean_corner_distance_fraction"], 1e-9))
    source_confidence = (cq_weight * contour_quad.confidence + ec_weight * edge_contour.confidence) / (cq_weight + ec_weight)
    confidence = float(max(0.0, min(1.0, source_confidence * agreement)))
    diagnostics.update({
        "reason": "consensus",
        "normalized_voter_weights": {
            "contour_quad": float(cq_weight / (cq_weight + ec_weight)),
            "edge_contour": float(ec_weight / (cq_weight + ec_weight)),
        },
        "agreement_score": float(agreement),
        "source_confidence": float(source_confidence),
    })
    if confidence < values["minimum_consensus_confidence"]:
        diagnostics["reason"] = "consensus_confidence_below_minimum"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    return Candidate(
        method=METHOD,
        bbox=[int(x1), int(y1), int(x2), int(y2)],
        corners=consensus.astype(float).tolist(),
        confidence=confidence,
        score=confidence,
        diagnostics=diagnostics,
    )


def debug_images(*, image_bgr: np.ndarray, diagnostics: dict[str, Any], candidate_corners: list[list[float]] | None) -> dict[str, np.ndarray]:
    voters = diagnostics.get("voters") if isinstance(diagnostics, dict) else {}
    voters = voters if isinstance(voters, dict) else {}

    def canvas() -> np.ndarray:
        return image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    cq_image = canvas()
    ec_image = canvas()
    agreement_image = canvas()
    selected_image = canvas()

    def draw(image: np.ndarray, corners: Any, color: tuple[int, int, int], thickness: int) -> None:
        if corners is None:
            return
        points = np.asarray(corners, dtype=np.float32)
        if points.shape == (4, 2):
            cv2.polylines(image, [np.rint(points).astype(np.int32)], True, color, thickness, cv2.LINE_AA)

    cq = voters.get("contour_quad") if isinstance(voters.get("contour_quad"), dict) else {}
    ec = voters.get("edge_contour") if isinstance(voters.get("edge_contour"), dict) else {}
    draw(cq_image, cq.get("corners"), (0, 255, 255), 4)
    draw(ec_image, ec.get("corners"), (255, 0, 255), 4)
    draw(agreement_image, cq.get("corners"), (0, 255, 255), 3)
    draw(agreement_image, ec.get("corners"), (255, 0, 255), 3)
    draw(selected_image, candidate_corners, (0, 0, 255), 4)
    return {
        "contour-quad-vote.png": cq_image,
        "edge-contour-vote.png": ec_image,
        "agreement-overlay.png": agreement_image,
        "selected-consensus.png": selected_image,
    }
