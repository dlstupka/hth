import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.geometry.detector_border_fusion_quad import BASELINE_PARAMETERS, debug_images, detect
from hth.regression.parameter_space import exhaustive_parameter_sets


class BorderFusionQuadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(self.image, (45, 30), (275, 210), (235, 235, 235), -1)
        self.mask = np.ones((240, 320), dtype=np.uint8) * 255

    def test_fuses_sides_from_multiple_child_detectors(self) -> None:
        candidate = detect(image_bgr=self.image, mask=self.mask)
        self.assertEqual(candidate.method, "border_fusion_quad")
        self.assertEqual(candidate.status, "ok")
        sources = candidate.diagnostics["selected_side_sources"]
        self.assertGreaterEqual(len(set(sources)), BASELINE_PARAMETERS["minimum_distinct_sources"])
        left, top, right, bottom = candidate.bbox
        self.assertLess(abs(left - 45), 20)
        self.assertLess(abs(right - 275), 20)
        self.assertLess(abs(top - 30), 20)
        self.assertLess(abs(bottom - 210), 20)

    def test_rejects_unknown_parameter(self) -> None:
        with self.assertRaises(ValueError):
            detect(image_bgr=self.image, mask=self.mask, parameters={"bogus": 1})

    def test_debug_package_shows_child_and_fused_geometry(self) -> None:
        candidate = detect(image_bgr=self.image, mask=self.mask)
        images = debug_images(
            image_bgr=self.image,
            mask=self.mask,
            candidate_corners=candidate.corners,
            verbose=True,
        )
        self.assertIn("fusion-gradient.png", images)
        self.assertIn("fusion-child-quads.png", images)
        self.assertIn("fusion-selected-quad.png", images)
        self.assertIn("fusion-side-sources.png", images)

    def test_initial_calibration_domain_has_243_sets(self) -> None:
        config = json.loads(Path("config/detectors/border_fusion_quad.json").read_text(encoding="utf-8"))
        self.assertEqual(len(exhaustive_parameter_sets(config)), 243)
        self.assertEqual(config["profiles"]["baseline"]["minimum_distinct_sources"], BASELINE_PARAMETERS["minimum_distinct_sources"])


if __name__ == "__main__":
    unittest.main()
