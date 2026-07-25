"""Probabilistic Hough-lines adapter for detector regression."""
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
        "hough", image_bgr=image_bgr, mask=mask, parameters=parameters
    )


def pre_regression_report_sections(config: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = config.get("parameters", {})

    def values(name: str) -> list[Any]:
        definition = parameters.get(name, {})
        configured = definition.get("values", []) if isinstance(definition, dict) else []
        return list(configured)

    def variants(names: tuple[str, ...]) -> int:
        return prod(len(values(name)) or 1 for name in names)

    extraction_names = (
        "canny_low_threshold",
        "hough_threshold_fraction",
        "minimum_length_fraction",
        "maximum_gap_fraction",
    )
    classification_names = ("axis_angle_tolerance_degrees",)
    envelope_names = ("outer_percentile", "bbox_padding_fraction")

    return [
        {
            "title": "Probabilistic Hough Transform Algorithm",
            "rows": [
                ("Implementation", "OpenCV HoughLinesP"),
                ("Input", "masked Canny edges from Gaussian-blurred grayscale image"),
                ("Line weighting", "detected segment length"),
                ("Envelope model", "axis-aligned weighted outer percentiles"),
                ("Edge families", "horizontal and vertical"),
            ],
        },
        {
            "title": "Hough Line Extraction Search Space",
            "rows": [
                ("Canny low thresholds", ", ".join(map(str, values("canny_low_threshold")))),
                ("Vote threshold fractions", ", ".join(map(str, values("hough_threshold_fraction")))),
                ("Minimum length fractions", ", ".join(map(str, values("minimum_length_fraction")))),
                ("Maximum gap fractions", ", ".join(map(str, values("maximum_gap_fraction")))),
                ("Extraction variants", variants(extraction_names)),
            ],
        },
        {
            "title": "Hough Axis Classification Search Space",
            "rows": [
                ("Axis-angle tolerances", ", ".join(map(str, values("axis_angle_tolerance_degrees")))),
                ("Classification variants", variants(classification_names)),
            ],
        },
        {
            "title": "Hough Line Detector Configuration",
            "rows": [
                ("Outer percentiles", ", ".join(map(str, values("outer_percentile")))),
                ("Bounding-box padding fractions", ", ".join(map(str, values("bbox_padding_fraction")))),
                ("Minimum envelope fraction", config.get("profiles", {}).get("baseline", {}).get("minimum_bbox_area_fraction", "--")),
                ("Envelope variants", variants(envelope_names)),
            ],
        },
    ]


__all__ = ["detect", "pre_regression_report_sections"]
