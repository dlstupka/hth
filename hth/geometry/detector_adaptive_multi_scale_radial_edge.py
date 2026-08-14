from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from . import detector_adaptive_radial_edge, detector_multi_scale_radial_edge
from .model import Candidate

METHOD = "adaptive_multi_scale_radial_edge"
PARENT_METHOD = "multi_scale_radial_edge"
PARENT_PARAMETER_SET_ID = "ddb7623ebb92"

# Keep MSRE's calibrated multiscale evidence fixed.  AMSRE Gen1 changes only
# the angular sampling policy: a coarse pass measures per-side support, then a
# finer angular pass is allocated only to weak sides.
BASELINE_PARAMETERS: dict[str, Any] = {
    "base_sigma": 1.0,
    "scale_ratio": 3.5,
    "scale_count": 4,
    "minimum_radius_fraction": 0.16,
    "maximum_radius_fraction": 0.78,
    "gradient_percentile": 96.875,
    "minimum_ray_support": 0.36,
    "minimum_area_fraction": 0.18,
    "maximum_area_fraction": 0.98,
    "bbox_padding_fraction": 0.0,
    "support_weight": 0.50,
    "strength_weight": 0.30,
    "area_weight": 0.20,
    "coarse_angle_step_degrees": 360.0 / 176.0,
    "refined_angle_step_degrees": 0.50,
    "weak_side_support_fraction": 0.55,
    "side_assignment_tolerance_fraction": 0.025,
    "maximum_refined_sides": 2,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, Any]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(
                f"Unknown Adaptive Multi-Scale Radial Edge parameters: {', '.join(unknown)}"
            )
        values.update(overrides)

    values["scale_count"] = int(values["scale_count"])
    values["maximum_refined_sides"] = int(values["maximum_refined_sides"])
    for name in values:
        if name not in {"scale_count", "maximum_refined_sides"}:
            values[name] = float(values[name])

    # Reuse MSRE's validation for every fixed multiscale/scoring control.  A
    # representative ray_count is supplied only for validation because AMSRE's
    # angular population is expressed by coarse_angle_step_degrees instead.
    parent_values = {
        name: values[name]
        for name in detector_multi_scale_radial_edge.BASELINE_PARAMETERS
        if name != "ray_count"
    }
    parent_values["ray_count"] = 176
    detector_multi_scale_radial_edge._parameters(parent_values)

    if values["coarse_angle_step_degrees"] <= 0.0:
        raise ValueError("coarse_angle_step_degrees must be positive")
    if values["refined_angle_step_degrees"] <= 0.0:
        raise ValueError("refined_angle_step_degrees must be positive")
    if values["refined_angle_step_degrees"] >= values["coarse_angle_step_degrees"]:
        raise ValueError("refined angle step must be smaller than coarse angle step")
    if not 0.0 <= values["weak_side_support_fraction"] <= 1.0:
        raise ValueError("weak_side_support_fraction must be between 0 and 1")
    if values["side_assignment_tolerance_fraction"] <= 0.0:
        raise ValueError("side_assignment_tolerance_fraction must be positive")
    if not 0 <= values["maximum_refined_sides"] <= 4:
        raise ValueError("maximum_refined_sides must be between 0 and 4")
    return values


def _gray(image_bgr: np.ndarray) -> np.ndarray:
    return detector_multi_scale_radial_edge._gray(image_bgr)


def _sample_angles(
    fused: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    threshold: float,
    angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return detector_adaptive_radial_edge._sample_angles(
        fused, center, radii, threshold, angles
    )


def _run(image_bgr: np.ndarray, values: dict[str, Any]) -> dict[str, Any]:
    gray = _gray(image_bgr)
    height, width = gray.shape
    fused, sigmas = detector_multi_scale_radial_edge._scale_space(gray, values)
    center = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=np.float32)
    max_radius = float(np.hypot(width, height) / 2.0)
    radii = np.arange(
        max(1.0, max_radius * values["minimum_radius_fraction"]),
        max_radius * values["maximum_radius_fraction"] + 1.0,
        1.0,
        dtype=np.float32,
    )
    threshold = float(np.percentile(fused, values["gradient_percentile"]))

    coarse_step = np.deg2rad(values["coarse_angle_step_degrees"])
    coarse_angles = np.arange(0.0, 2.0 * np.pi, coarse_step, dtype=np.float32)
    coarse_points, coarse_strengths, coarse_used = _sample_angles(
        fused, center, radii, threshold, coarse_angles
    )

    result: dict[str, Any] = {
        "fused": fused,
        "sigmas": sigmas,
        "center": center,
        "threshold": threshold,
        "requested_coarse_rays": int(len(coarse_angles)),
        "coarse_points": coarse_points,
        "coarse_angles": coarse_used,
        "refined_points": np.empty((0, 2), dtype=np.float32),
        "refined_angles": np.empty((0,), dtype=np.float32),
        "weak_sides": [],
        "side_eligible_rays": [0, 0, 0, 0],
        "side_support_counts": [0, 0, 0, 0],
        "side_support_fractions": [0.0] * 4,
        "all_points": coarse_points,
        "coarse_corners": None,
        "final_corners": None,
        "mean_strength": 0.0,
    }
    if len(coarse_points) < 4:
        return result

    coarse_corners = detector_adaptive_radial_edge._fit(coarse_points)
    result["coarse_corners"] = coarse_corners
    diagonal = float(np.hypot(width, height))
    eligible, counts, fractions = detector_adaptive_radial_edge._side_support(
        coarse_points,
        coarse_used,
        coarse_angles,
        center,
        coarse_corners,
        diagonal,
        values["side_assignment_tolerance_fraction"],
    )
    weak = [
        side
        for side, fraction in sorted(enumerate(fractions), key=lambda item: item[1])
        if fraction < values["weak_side_support_fraction"]
    ][: values["maximum_refined_sides"]]
    result["side_eligible_rays"] = eligible
    result["side_support_counts"] = counts
    result["side_support_fractions"] = fractions
    result["weak_sides"] = weak

    refined_strengths = np.empty((0,), dtype=np.float32)
    if weak:
        refine_step = np.deg2rad(values["refined_angle_step_degrees"])
        coarse_keys = {round(float(angle), 6) for angle in coarse_used}
        refine_angles: list[float] = []
        for angle in np.arange(0.0, 2.0 * np.pi, refine_step, dtype=np.float32):
            if round(float(angle), 6) in coarse_keys:
                continue
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            if (
                detector_adaptive_radial_edge._ray_intersection_side(
                    center, direction, coarse_corners
                )
                in weak
            ):
                refine_angles.append(float(angle))
        if refine_angles:
            refined_points, refined_strengths, refined_used = _sample_angles(
                fused,
                center,
                radii,
                threshold,
                np.asarray(refine_angles, dtype=np.float32),
            )
            result["refined_points"] = refined_points
            result["refined_angles"] = refined_used
            if len(refined_points):
                result["all_points"] = np.vstack([coarse_points, refined_points])

    if len(result["all_points"]) >= 4:
        result["final_corners"] = detector_adaptive_radial_edge._fit(result["all_points"])

    strengths = (
        np.concatenate([coarse_strengths, refined_strengths])
        if len(refined_strengths)
        else coarse_strengths
    )
    result["mean_strength"] = (
        0.0
        if len(strengths) == 0
        else float(np.mean(np.clip(strengths, 0.0, 2.0)) / 2.0)
    )
    return result


def detect(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> Candidate:
    del mask
    values = _parameters(parameters)
    height, width = _gray(image_bgr).shape
    if min(height, width) < 16:
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {"reason": "image_too_small", "parameters": values},
            status="no_candidate",
        )

    run = _run(image_bgr, values)
    points = run["all_points"]
    requested = max(1, run["requested_coarse_rays"])
    support = len(run["coarse_points"]) / float(requested)
    diagnostics: dict[str, Any] = {
        "parameters": values,
        "parent_detector": PARENT_METHOD,
        "parent_parameter_set_id": PARENT_PARAMETER_SET_ID,
        "scale_sigmas": run["sigmas"],
        "requested_coarse_rays": requested,
        "coarse_supported_rays": int(len(run["coarse_points"])),
        "refined_supported_rays": int(len(run["refined_points"])),
        "total_supported_rays": int(len(points)),
        "ray_support": support,
        "mean_fused_strength": run["mean_strength"],
        "side_eligible_rays": run["side_eligible_rays"],
        "side_support_counts": run["side_support_counts"],
        "side_support_fractions": run["side_support_fractions"],
        "weak_sides": run["weak_sides"],
        "refinement_triggered": bool(run["weak_sides"]),
        "evidence": "adaptive_angular_sampling_over_fixed_multi_scale_radial_evidence",
    }
    if len(points) < 4 or support < values["minimum_ray_support"]:
        diagnostics["reason"] = "insufficient_ray_support"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    corners = run["final_corners"]
    area = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2))))
    area_fraction = area / float(width * height)
    diagnostics["area_fraction"] = area_fraction
    if not values["minimum_area_fraction"] <= area_fraction <= values["maximum_area_fraction"]:
        diagnostics["reason"] = "implausible_area"
        return Candidate(METHOD, None, None, 0.0, 0.0, diagnostics, status="no_candidate")

    padding = int(round(min(height, width) * values["bbox_padding_fraction"]))
    x, y, box_width, box_height = cv2.boundingRect(
        np.rint(corners).astype(np.int32).reshape(-1, 1, 2)
    )
    bbox = [
        max(0, x - padding),
        max(0, y - padding),
        min(width, x + box_width + padding),
        min(height, y + box_height + padding),
    ]
    total = values["support_weight"] + values["strength_weight"] + values["area_weight"]
    score = (
        support * values["support_weight"]
        + run["mean_strength"] * values["strength_weight"]
        + area_fraction * values["area_weight"]
    ) / total
    return Candidate(
        METHOD,
        bbox,
        corners.astype(float).tolist(),
        float(score),
        float(score),
        diagnostics,
    )


def debug_images(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: dict[str, Any] | None = None,
    candidate_corners: list[list[float]] | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    del mask
    values = _parameters(parameters)
    run = _run(image_bgr, values)
    fused = cv2.normalize(run["fused"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    base = image_bgr.copy() if image_bgr.ndim == 3 else cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    points_view = base.copy()
    for point in np.rint(run["coarse_points"]).astype(np.int32):
        cv2.circle(points_view, tuple(point), 2, (0, 255, 255), -1)
    for point in np.rint(run["refined_points"]).astype(np.int32):
        cv2.circle(points_view, tuple(point), 2, (255, 0, 255), -1)
    if candidate_corners is not None:
        cv2.polylines(
            points_view,
            [np.rint(np.asarray(candidate_corners)).astype(np.int32).reshape(-1, 1, 2)],
            True,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    images = {
        "adaptive-multi-scale-gradient.png": fused,
        "adaptive-multi-scale-radial-points.png": points_view,
    }
    if verbose:
        gray = _gray(image_bgr)
        images["adaptive-multi-scale-space.png"] = np.hstack(
            [
                cv2.normalize(
                    cv2.GaussianBlur(gray, (0, 0), sigma),
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX,
                ).astype(np.uint8)
                for sigma in run["sigmas"]
            ]
        )

        pass1 = base.copy()
        if run["coarse_corners"] is not None:
            cv2.polylines(
                pass1,
                [np.rint(run["coarse_corners"]).astype(np.int32).reshape(-1, 1, 2)],
                True,
                (255, 160, 0),
                3,
                cv2.LINE_AA,
            )
        images["adaptive-multi-scale-pass1-fit.png"] = pass1

        side_support = base.copy()
        if run["coarse_corners"] is not None:
            corners = np.rint(run["coarse_corners"]).astype(np.int32)
            side_names = ("top", "right", "bottom", "left")
            for side in range(4):
                weak = side in run["weak_sides"]
                color = (0, 0, 255) if weak else (0, 180, 0)
                cv2.line(
                    side_support,
                    tuple(corners[side]),
                    tuple(corners[(side + 1) % 4]),
                    color,
                    5,
                    cv2.LINE_AA,
                )
                midpoint = np.rint(
                    (corners[side] + corners[(side + 1) % 4]) / 2.0
                ).astype(int)
                label = (
                    f"{side_names[side]} {run['side_support_counts'][side]}/"
                    f"{run['side_eligible_rays'][side]} "
                    f"{run['side_support_fractions'][side]:.0%} "
                    f"{'WEAK' if weak else 'STRONG'}"
                )
                cv2.putText(
                    side_support,
                    label,
                    (int(midpoint[0] + 8), int(midpoint[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        images["adaptive-multi-scale-side-support.png"] = side_support

        pass2 = base.copy()
        for point in np.rint(run["refined_points"]).astype(np.int32):
            cv2.circle(pass2, tuple(point), 3, (255, 0, 255), -1)
        if run["final_corners"] is not None:
            cv2.polylines(
                pass2,
                [np.rint(run["final_corners"]).astype(np.int32).reshape(-1, 1, 2)],
                True,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
        images["adaptive-multi-scale-pass2-fit.png"] = pass2
    return images


__all__ = [
    "BASELINE_PARAMETERS",
    "METHOD",
    "PARENT_METHOD",
    "PARENT_PARAMETER_SET_ID",
    "debug_images",
    "detect",
]
