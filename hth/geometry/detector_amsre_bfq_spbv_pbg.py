from __future__ import annotations

from typing import Any

import numpy as np

from . import (
    detector_adaptive_multi_scale_radial_edge,
    detector_border_fusion_quad,
    detector_page_background,
    detector_signed_polar_boundary_vote,
)
from .detector_side_consensus_fusion import (
    COMMON_CHILD_CALIBRATIONS,
    SideConsensusFusion,
    debug_images as _debug_images,
    detect as _detect,
)
from .model import Candidate

METHOD = "amsre_bfq_spbv_pbg"
CHILD_CALIBRATIONS: dict[str, dict[str, Any]] = {
    "adaptive_multi_scale_radial_edge": {
        "parameter_set_id": "21ea516c3c5a",
        "parameters": {
            "coarse_angle_step_degrees": 2.0454545454545454,
            "maximum_refined_sides": 3,
            "refined_angle_step_degrees": 0.35,
            "side_assignment_tolerance_fraction": 0.0075,
            "weak_side_support_fraction": 0.65,
        },
    },
    **COMMON_CHILD_CALIBRATIONS,
}
BASELINE_PARAMETERS: dict[str, Any] = {
    "minimum_side_consensus": 0.1,
    "consensus_tolerance_fraction": 0.012664,
    "gradient_weight": 0.25,
    "gradient_percentile": 76.0,
    "consensus_weight": 0.6,
    "source_diversity_weight": 0.15,
    "minimum_side_gradient_support": 0.03,
}

_CHILD_MODULES = {
    "adaptive_multi_scale_radial_edge": detector_adaptive_multi_scale_radial_edge,
    "border_fusion_quad": detector_border_fusion_quad,
    "signed_polar_boundary_vote": detector_signed_polar_boundary_vote,
    "page_background": detector_page_background,
}
_FAMILY = SideConsensusFusion(
    method=METHOD,
    label="Fusion Gen2",
    baseline_parameters=BASELINE_PARAMETERS,
    child_calibrations=CHILD_CALIBRATIONS,
    children=tuple((name, module.detect) for name, module in _CHILD_MODULES.items()),
    child_parameter_defaults={
        name: module.BASELINE_PARAMETERS for name, module in _CHILD_MODULES.items()
    },
    debug_prefix="fusion-gen2",
)


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None) -> Candidate:
    return _detect(_FAMILY, image_bgr=image_bgr, mask=mask, parameters=parameters)


def debug_images(
    *, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None,
    candidate_corners: list[list[float]] | None = None, verbose: bool = False,
) -> dict[str, np.ndarray]:
    return _debug_images(
        _FAMILY, image_bgr=image_bgr, mask=mask, parameters=parameters,
        candidate_corners=candidate_corners, verbose=verbose,
    )


__all__ = ["BASELINE_PARAMETERS", "CHILD_CALIBRATIONS", "METHOD", "debug_images", "detect"]
