from __future__ import annotations

from typing import Any

import numpy as np

from . import (
    detector_border_fusion_quad,
    detector_multi_scale_radial_edge,
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
    **COMMON_CHILD_CALIBRATIONS,
}
BASELINE_PARAMETERS: dict[str, Any] = {
    "gradient_percentile": 76.0,
    "minimum_side_gradient_support": 0.03,
    "consensus_tolerance_fraction": 0.012,
    "minimum_side_consensus": 0.50,
    "consensus_weight": 0.60,
    "gradient_weight": 0.25,
    "source_diversity_weight": 0.15,
}

_CHILD_MODULES = {
    "multi_scale_radial_edge": detector_multi_scale_radial_edge,
    "border_fusion_quad": detector_border_fusion_quad,
    "signed_polar_boundary_vote": detector_signed_polar_boundary_vote,
    "page_background": detector_page_background,
}
_FAMILY = SideConsensusFusion(
    method=METHOD,
    label="Fusion Gen1",
    baseline_parameters=BASELINE_PARAMETERS,
    child_calibrations=CHILD_CALIBRATIONS,
    children=tuple((name, module.detect) for name, module in _CHILD_MODULES.items()),
    child_parameter_defaults={
        name: module.BASELINE_PARAMETERS for name, module in _CHILD_MODULES.items()
    },
    debug_prefix="fusion-gen1",
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
