import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.geometry import detector_page_background
from hth.regression.parameter_space import exhaustive_parameter_sets


class PageBackgroundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((260, 360, 3), (35, 48, 62), dtype=np.uint8)
        page = np.array([[55, 35], [315, 48], [300, 225], [42, 215]], dtype=np.int32)
        cv2.fillConvexPoly(self.image, page, (225, 220, 205))
        cv2.line(self.image, (80, 80), (280, 90), (80, 80, 80), 2)
        self.mask = np.ones((260, 360), dtype=np.uint8) * 255

    def test_detects_page_from_surrounding_background(self) -> None:
        candidate = detector_page_background.detect(image_bgr=self.image, mask=self.mask)
        self.assertEqual(candidate.method, "page_background")
        self.assertEqual(candidate.status, "ok")
        self.assertEqual(len(candidate.corners), 4)
        self.assertGreater(candidate.diagnostics["border_background_fraction"], 0.8)
        left, top, right, bottom = candidate.bbox
        self.assertLess(left, 75)
        self.assertGreater(right, 285)
        self.assertLess(top, 65)
        self.assertGreater(bottom, 200)

    def test_rejects_unknown_parameter(self) -> None:
        with self.assertRaises(ValueError):
            detector_page_background.detect(image_bgr=self.image, mask=self.mask, parameters={"mystery": 1})

    def test_debug_artifacts_include_background_evidence(self) -> None:
        images = detector_page_background.debug_images(image_bgr=self.image, mask=self.mask, verbose=True)
        self.assertIn("page-background-distance.png", images)
        self.assertIn("page-background-mask.png", images)
        self.assertIn("page-background-candidate.png", images)
        self.assertIn("page-background-border-samples.png", images)

    def test_initial_calibration_domain_is_2187_sets_and_retains_baseline(self) -> None:
        config = json.loads(Path("config/detectors/page_background.json").read_text(encoding="utf-8"))
        self.assertEqual(len(exhaustive_parameter_sets(config)), 2187)
        baseline = config["profiles"]["baseline"]
        self.assertEqual(baseline["border_band_fraction"], detector_page_background.BASELINE_PARAMETERS["border_band_fraction"])
        self.assertIn(baseline["color_distance_threshold"], config["parameters"]["color_distance_threshold"]["values"])


if __name__ == "__main__":
    unittest.main()
