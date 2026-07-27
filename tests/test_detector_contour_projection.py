from __future__ import annotations

import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.geometry.detector_contour_projection import BASELINE_PARAMETERS, detect


class ContourProjectionDetectorTests(unittest.TestCase):
    def test_detects_text_banded_contour_quadrilateral(self) -> None:
        image = np.full((420, 620, 3), 30, dtype=np.uint8)
        mask = np.zeros((420, 620), dtype=np.uint8)
        polygon = np.array([[80, 55], [545, 75], [520, 375], [65, 350]], dtype=np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
        cv2.fillConvexPoly(image, polygon, (235, 235, 235))
        for y in range(100, 335, 24):
            cv2.line(image, (115, y), (485, y + 8), (35, 35, 35), 5)

        candidate = detect(image_bgr=image, mask=mask)

        self.assertEqual(candidate.method, "contour_projection")
        self.assertEqual(candidate.status, "ok")
        self.assertIsNotNone(candidate.corners)
        self.assertEqual(candidate.diagnostics["parameters"], BASELINE_PARAMETERS)
        self.assertGreater(candidate.diagnostics["projection_score"], 0.08)

    def test_rejects_unknown_parameter(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "Unknown Contour \\+ Projection parameters"):
            detect(image_bgr=image, mask=mask, parameters={"mystery": 1})

    def test_regression_config_defines_baseline_profile(self) -> None:
        path = Path(__file__).resolve().parents[1] / "config" / "detectors" / "contour_projection.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(config["detector"], "contour_projection")
        self.assertEqual(config["profiles"]["baseline"], BASELINE_PARAMETERS)


if __name__ == "__main__":
    unittest.main()
