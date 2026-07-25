"""Line Segment Detector adapter for the detector-agnostic regression framework."""
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
        "lsd",
        image_bgr=image_bgr,
        mask=mask,
        parameters=parameters,
    )


def pre_regression_report_sections(config: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = config.get("parameters", {})

    def values(name: str) -> list[Any]:
        definition = parameters.get(name, {})
        configured = definition.get("values", []) if isinstance(definition, dict) else []
        return list(configured)

    extraction_names = ("refine_mode", "scale")
    classification_names = (
        "minimum_length_fraction",
        "axis_angle_tolerance_degrees",
    )
    envelope_names = (
        "outer_percentile",
        "minimum_bbox_area_fraction",
        "bbox_padding_fraction",
    )

    def variants(names: tuple[str, ...]) -> int:
        return prod(len(values(name)) or 1 for name in names)

    return [
        {
            "title": "Line Segment Detection Algorithm",
            "rows": [
                ("Implementation", "OpenCV Line Segment Detector"),
                ("Input", "Gaussian-blurred grayscale analysis image"),
                ("Segment weighting", "detected segment length"),
                ("Envelope model", "axis-aligned weighted outer percentiles"),
                ("Edge families", "horizontal and vertical"),
            ],
        },
        {
            "title": "Line Segment Extraction Search Space",
            "rows": [
                ("Refinement modes", ", ".join(map(str, values("refine_mode")))),
                ("Image scales", ", ".join(map(str, values("scale")))),
                ("Extraction variants", variants(extraction_names)),
            ],
        },
        {
            "title": "Axis Segment Classification Search Space",
            "rows": [
                ("Minimum length fractions", ", ".join(map(str, values("minimum_length_fraction")))),
                ("Axis-angle tolerances", ", ".join(map(str, values("axis_angle_tolerance_degrees")))),
                ("Classification variants", variants(classification_names)),
            ],
        },
        {
            "title": "Line Segment Detector Configuration",
            "rows": [
                ("Outer percentiles", ", ".join(map(str, values("outer_percentile")))),
                ("Minimum envelope fractions", ", ".join(map(str, values("minimum_bbox_area_fraction")))),
                ("Bounding-box padding fractions", ", ".join(map(str, values("bbox_padding_fraction")))),
                ("Envelope variants", variants(envelope_names)),
            ],
        },
    ]


__all__ = ["detect", "pre_regression_report_sections"]
