from __future__ import annotations

import unittest

import cv2
import numpy as np

from hth.geometry.detector_lsd import BASELINE_PARAMETERS, detect
from hth.regression.adapters.lsd import pre_regression_report_sections


def _page_image() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((600, 800, 3), 24, dtype=np.uint8)
    cv2.rectangle(image, (90, 55), (710, 545), (235, 235, 235), -1)
    cv2.rectangle(image, (90, 55), (710, 545), (255, 255, 255), 4)
    mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.rectangle(mask, (90, 55), (710, 545), 255, -1)
    return image, mask


class LSDDetectorTests(unittest.TestCase):
    def test_lsd_returns_candidate_for_rectangular_page(self) -> None:
        image, mask = _page_image()
        candidate = detect(image_bgr=image, mask=mask)
        self.assertEqual(candidate.method, "lsd")
        self.assertIsNotNone(candidate.bbox)
        self.assertGreater(candidate.confidence, 0.5)
        self.assertGreaterEqual(candidate.diagnostics["vertical_segments"], 2)
        self.assertGreaterEqual(candidate.diagnostics["horizontal_segments"], 2)
        self.assertEqual(candidate.diagnostics["parameters"], BASELINE_PARAMETERS)

    def test_lsd_accepts_black_box_parameter_overrides(self) -> None:
        image, mask = _page_image()
        candidate = detect(
            image_bgr=image,
            mask=mask,
            parameters={
                "refine_mode": "none",
                "scale": 1.0,
                "minimum_length_fraction": 0.08,
                "axis_angle_tolerance_degrees": 28.0,
                "outer_percentile": 5.0,
                "bbox_padding_fraction": 0.005,
            },
        )
        self.assertIsNotNone(candidate.bbox)
        self.assertEqual(candidate.diagnostics["parameters"]["refine_mode"], "none")
        self.assertEqual(candidate.diagnostics["parameters"]["scale"], 1.0)
        self.assertEqual(
            candidate.diagnostics["parameters"]["bbox_padding_fraction"], 0.005
        )

    def test_lsd_rejects_unknown_parameter(self) -> None:
        image, mask = _page_image()
        with self.assertRaisesRegex(ValueError, "Unknown LSD parameters"):
            detect(image_bgr=image, mask=mask, parameters={"unknown": 1})

    def test_lsd_cleanly_returns_no_candidate_for_blank_image(self) -> None:
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        mask = np.zeros((300, 400), dtype=np.uint8)
        candidate = detect(image_bgr=image, mask=mask)
        self.assertIsNone(candidate.bbox)
        self.assertEqual(candidate.status, "ok")
        self.assertIn(
            candidate.diagnostics["reason"],
            {"no_line_segments", "insufficient_axis_segments"},
        )

    def test_lsd_pre_regression_sections_describe_search_stages(self) -> None:
        config = {
            "parameters": {
                "refine_mode": {"values": ["none", "std", "adv"]},
                "scale": {"values": [0.6, 0.8, 1.0]},
                "minimum_length_fraction": {"values": [0.08, 0.14, 0.22]},
                "axis_angle_tolerance_degrees": {"values": [10.0, 18.0, 28.0]},
                "outer_percentile": {"values": [5.0, 10.0, 20.0]},
                "minimum_bbox_area_fraction": {"values": [0.08, 0.10, 0.15]},
                "bbox_padding_fraction": {"values": [0.0, 0.005, 0.015]},
            }
        }
        sections = pre_regression_report_sections(config)
        self.assertEqual(
            [section["title"] for section in sections],
            [
                "Line Segment Detection Algorithm",
                "Line Segment Extraction Search Space",
                "Axis Segment Classification Search Space",
                "Line Segment Detector Configuration",
            ],
        )
        self.assertEqual(dict(sections[1]["rows"])["Extraction variants"], 9)
        self.assertEqual(dict(sections[2]["rows"])["Classification variants"], 9)
        self.assertEqual(dict(sections[3]["rows"])["Envelope variants"], 27)


if __name__ == "__main__":
    unittest.main()
