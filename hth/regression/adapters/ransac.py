"""RANSAC adapter for the detector-agnostic regression framework."""
from __future__ import annotations

from math import prod
from typing import Any

import numpy as np

from hth.geometry.model import Candidate
from hth.geometry.registry import run_registered_detector


def detect(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> Candidate:
    return run_registered_detector(
        "ransac", image_bgr=image_bgr, mask=mask, parameters=parameters
    )


def pre_regression_report_sections(config: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = config.get("parameters", {})

    def values(name: str) -> list[Any]:
        definition = parameters.get(name, {})
        configured = definition.get("values", []) if isinstance(definition, dict) else []
        return list(configured)

    fit_names = (
        "residual_threshold_fraction",
        "max_trials",
        "minimum_mean_inlier_ratio",
    )
    candidate_names = ("minimum_bbox_area_fraction", "bbox_padding_fraction")
    fit_variants = prod(len(values(name)) or 1 for name in fit_names)
    candidate_variants = prod(len(values(name)) or 1 for name in candidate_names)

    return [
        {
            "title": "RANSAC Boundary Sampling Algorithm",
            "rows": [
                ("Boundary observations", "first and last foreground pixel per scan line"),
                ("Edge families", "left, right, top, bottom"),
                ("Scan scaling basis", "analysis image width and height"),
                ("Line model", "2-D total-least-squares line"),
                ("Random seed", 42),
            ],
        },
        {
            "title": "RANSAC Boundary Sampling Search Space",
            "rows": [
                ("Scan samples", ", ".join(map(str, values("scan_samples")))),
                ("Minimum foreground fractions", ", ".join(map(str, values("minimum_scan_foreground_fraction")))),
                ("Sampling variants", (len(values("scan_samples")) or 1) * (len(values("minimum_scan_foreground_fraction")) or 1)),
            ],
        },
        {
            "title": "RANSAC Line Fitting Search Space",
            "rows": [
                ("Residual threshold fractions", ", ".join(map(str, values("residual_threshold_fraction")))),
                ("Maximum trials", ", ".join(map(str, values("max_trials")))),
                ("Minimum mean inlier ratios", ", ".join(map(str, values("minimum_mean_inlier_ratio")))),
                ("Line-fitting variants", fit_variants),
            ],
        },
        {
            "title": "RANSAC Detector Configuration",
            "rows": [
                ("Minimum envelope fractions", ", ".join(map(str, values("minimum_bbox_area_fraction")))),
                ("Bounding-box padding fractions", ", ".join(map(str, values("bbox_padding_fraction")))),
                ("Candidate variants", candidate_variants),
            ],
        },
    ]


__all__ = ["detect", "pre_regression_report_sections"]
