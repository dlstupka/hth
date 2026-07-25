from __future__ import annotations

import unittest

import cv2
import numpy as np

from hth.geometry.detector_hough import BASELINE_PARAMETERS, detect
from hth.regression.adapters.hough import pre_regression_report_sections


def _page_image() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((600, 800, 3), 24, dtype=np.uint8)
    cv2.rectangle(image, (90, 55), (710, 545), (235, 235, 235), -1)
    cv2.rectangle(image, (90, 55), (710, 545), (255, 255, 255), 5)
    mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.rectangle(mask, (90, 55), (710, 545), 255, -1)
    return image, mask


class HoughDetectorTests(unittest.TestCase):
    def test_hough_returns_candidate_for_rectangular_page(self) -> None:
        image, mask = _page_image()
        candidate = detect(image_bgr=image, mask=mask)
        self.assertEqual(candidate.method, "hough")
        self.assertIsNotNone(candidate.bbox)
        self.assertGreater(candidate.confidence, 0.5)
        self.assertGreaterEqual(candidate.diagnostics["vertical_lines"], 2)
        self.assertGreaterEqual(candidate.diagnostics["horizontal_lines"], 2)
        self.assertEqual(candidate.diagnostics["parameters"], BASELINE_PARAMETERS)

    def test_hough_accepts_black_box_parameter_overrides(self) -> None:
        image, mask = _page_image()
        candidate = detect(
            image_bgr=image,
            mask=mask,
            parameters={
                "canny_low_threshold": 25,
                "hough_threshold_fraction": 0.035,
                "minimum_length_fraction": 0.12,
                "maximum_gap_fraction": 0.09,
                "axis_angle_tolerance_degrees": 32.0,
                "outer_percentile": 5.0,
                "bbox_padding_fraction": 0.005,
            },
        )
        self.assertIsNotNone(candidate.bbox)
        self.assertEqual(candidate.diagnostics["parameters"]["canny_low_threshold"], 25)
        self.assertEqual(candidate.diagnostics["parameters"]["bbox_padding_fraction"], 0.005)

    def test_hough_rejects_unknown_parameter(self) -> None:
        image, mask = _page_image()
        with self.assertRaisesRegex(ValueError, "Unknown Hough parameters"):
            detect(image_bgr=image, mask=mask, parameters={"unknown": 1})

    def test_hough_cleanly_returns_no_candidate_for_blank_image(self) -> None:
        image = np.zeros((300, 400, 3), dtype=np.uint8)
        mask = np.zeros((300, 400), dtype=np.uint8)
        candidate = detect(image_bgr=image, mask=mask)
        self.assertIsNone(candidate.bbox)
        self.assertEqual(candidate.diagnostics["reason"], "no_hough_lines")

    def test_hough_pre_regression_sections_describe_search_stages(self) -> None:
        config = {
            "profiles": {"baseline": {"minimum_bbox_area_fraction": 0.10}},
            "parameters": {
                "canny_low_threshold": {"values": [25, 40, 65]},
                "hough_threshold_fraction": {"values": [0.035, 0.055, 0.08]},
                "minimum_length_fraction": {"values": [0.12, 0.20, 0.30]},
                "maximum_gap_fraction": {"values": [0.025, 0.055, 0.09]},
                "axis_angle_tolerance_degrees": {"values": [12.0, 22.0, 32.0]},
                "outer_percentile": {"values": [5.0, 10.0, 20.0]},
                "bbox_padding_fraction": {"values": [0.0, 0.005, 0.015]},
            },
        }
        sections = pre_regression_report_sections(config)
        self.assertEqual(
            [section["title"] for section in sections],
            [
                "Probabilistic Hough Transform Algorithm",
                "Hough Line Extraction Search Space",
                "Hough Axis Classification Search Space",
                "Hough Line Detector Configuration",
            ],
        )
        self.assertEqual(dict(sections[1]["rows"])["Extraction variants"], 81)
        self.assertEqual(dict(sections[2]["rows"])["Classification variants"], 3)
        self.assertEqual(dict(sections[3]["rows"])["Envelope variants"], 9)


if __name__ == "__main__":
    unittest.main()
