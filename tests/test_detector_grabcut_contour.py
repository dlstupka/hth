from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hth.geometry.detector_grabcut_contour import BASELINE_PARAMETERS, detect
from hth.geometry.model import Candidate


class GrabCutContourDetectorTests(unittest.TestCase):
    def test_returns_grabcut_geometry_after_contour_validation(self) -> None:
        grabcut = Candidate("grabcut", [12, 12, 88, 88], [[12, 12], [88, 12], [88, 88], [12, 88]], 0.8, 0.8, {})
        contour = Candidate("contour_quad", [10, 10, 90, 90], [[10, 10], [90, 10], [90, 90], [10, 90]], 0.7, 0.7, {})
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        with patch("hth.geometry.detector_grabcut_contour.detector_grabcut.detect", return_value=grabcut), patch("hth.geometry.detector_grabcut_contour.detector_contour_quad.detect", return_value=contour):
            candidate = detect(image_bgr=image, mask=mask)
        self.assertEqual(candidate.method, "grabcut_contour")
        self.assertEqual(candidate.status, "ok")
        self.assertEqual(candidate.corners, grabcut.corners)
        self.assertEqual(candidate.diagnostics["selected_geometry"], "grabcut")
        self.assertEqual(candidate.diagnostics["fusion_mode"], "grabcut_generated_contour_validated")
        self.assertGreater(candidate.diagnostics["agreement_iou"], 0.8)

    def test_rejects_low_agreement(self) -> None:
        grabcut = Candidate("grabcut", [0, 0, 30, 30], [[0, 0], [30, 0], [30, 30], [0, 30]], 0.8, 0.8, {})
        contour = Candidate("contour_quad", [70, 70, 99, 99], [[70, 70], [99, 70], [99, 99], [70, 99]], 0.8, 0.8, {})
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        with patch("hth.geometry.detector_grabcut_contour.detector_grabcut.detect", return_value=grabcut), patch("hth.geometry.detector_grabcut_contour.detector_contour_quad.detect", return_value=contour):
            candidate = detect(image_bgr=image, mask=mask)
        self.assertIsNone(candidate.bbox)
        self.assertEqual(candidate.diagnostics["reason"], "insufficient_grabcut_contour_agreement")

    def test_accepts_grabcut_without_contour_when_optional(self) -> None:
        grabcut = Candidate("grabcut", [12, 12, 88, 88], [[12, 12], [88, 12], [88, 88], [12, 88]], 0.8, 0.8, {})
        contour = Candidate("contour_quad", None, None, 0.0, 0.0, {"reason": "none"})
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        with patch("hth.geometry.detector_grabcut_contour.detector_grabcut.detect", return_value=grabcut), patch("hth.geometry.detector_grabcut_contour.detector_contour_quad.detect", return_value=contour):
            candidate = detect(image_bgr=image, mask=mask)
        self.assertEqual(candidate.corners, grabcut.corners)
        self.assertEqual(candidate.diagnostics["fusion_mode"], "grabcut_without_contour")

    def test_rejects_unknown_parameter(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "Unknown GrabCut \\+ Contour parameters"):
            detect(image_bgr=image, mask=mask, parameters={"mystery": 1})

    def test_regression_config_defines_baseline_profile(self) -> None:
        path = Path(__file__).resolve().parents[1] / "config" / "detectors" / "grabcut_contour.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["detector"], "grabcut_contour")
        self.assertEqual(config["profiles"]["baseline"], BASELINE_PARAMETERS)


if __name__ == "__main__":
    unittest.main()
