"""Connected-components adapter for the detector-agnostic regression framework."""
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
        "components",
        image_bgr=image_bgr,
        mask=mask,
        parameters=parameters,
    )


def pre_regression_report_sections(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe tunable Connected Components stages before evaluation begins."""
    parameters = config.get("parameters", {})

    def values(name: str) -> list[Any]:
        definition = parameters.get(name, {})
        configured = definition.get("values", []) if isinstance(definition, dict) else []
        return list(configured)

    close_values = values("morphology_close_fraction")
    dilate_values = values("morphology_dilate_fraction")
    morphology_variants = prod((len(close_values) or 1, len(dilate_values) or 1))

    return [
        {
            "title": "Morphology Preprocessing Tuning",
            "rows": [
                ("Operation sequence", "closing -> dilation"),
                ("Closing kernel fractions", ", ".join(map(str, close_values)) or "baseline only"),
                ("Dilation kernel fractions", ", ".join(map(str, dilate_values)) or "baseline only"),
                ("Morphology variants", morphology_variants),
                ("Kernel scaling basis", "minimum image dimension"),
                ("Kernel shape", "rectangular, odd-sized"),
            ],
        },
        {
            "title": "Connected Components Candidate Tuning",
            "rows": [
                ("Component area fractions", ", ".join(map(str, values("minimum_component_area_fraction")))),
                ("Component pixel floors", ", ".join(map(str, values("minimum_component_area_px")))),
                ("Merge area ratios", ", ".join(map(str, values("merge_area_ratio")))),
                ("Merge gap fractions", ", ".join(map(str, values("merge_gap_fraction")))),
                ("Minimum envelope fractions", ", ".join(map(str, values("minimum_bbox_area_fraction")))),
                ("Minimum selected-area fractions", ", ".join(map(str, values("minimum_selected_area_fraction")))),
                ("Bounding-box padding fractions", ", ".join(map(str, values("bbox_padding_fraction")))),
            ],
        },
    ]


__all__ = ["detect", "pre_regression_report_sections"]
