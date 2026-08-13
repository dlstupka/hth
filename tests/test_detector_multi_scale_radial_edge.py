import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.geometry.detector_multi_scale_radial_edge import BASELINE_PARAMETERS, debug_images, detect
from hth.regression.parameter_space import exhaustive_parameter_sets


class MultiScaleRadialEdgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(self.image, (45, 30), (275, 210), (235, 235, 235), -1)
        self.mask = np.ones((240, 320), dtype=np.uint8) * 255

    def test_detects_document_boundary_from_scale_space(self) -> None:
        candidate = detect(image_bgr=self.image, mask=self.mask)
        self.assertEqual(candidate.method, "multi_scale_radial_edge")
        self.assertEqual(candidate.status, "ok")
        self.assertGreaterEqual(len(candidate.diagnostics["scale_sigmas"]), 2)
        left, top, right, bottom = candidate.bbox
        self.assertLess(abs(left - 45), 20)
        self.assertLess(abs(right - 275), 20)
        self.assertLess(abs(top - 30), 20)
        self.assertLess(abs(bottom - 210), 20)

    def test_rejects_unknown_parameter(self) -> None:
        with self.assertRaises(ValueError):
            detect(image_bgr=self.image, mask=self.mask, parameters={"bogus": 1})

    def test_debug_package_has_comparable_scale_space_evidence(self) -> None:
        images = debug_images(image_bgr=self.image, mask=self.mask, verbose=True)
        self.assertIn("multi-scale-gradient.png", images)
        self.assertIn("multi-scale-radial-points.png", images)
        self.assertIn("scale-space.png", images)

    def test_generation_3_calibration_domain_expands_refined_basin_and_retains_anchors(self) -> None:
        config = json.loads(Path("config/detectors/multi_scale_radial_edge.json").read_text(encoding="utf-8"))
        self.assertEqual(len(exhaustive_parameter_sets(config)), 384076)
        self.assertEqual(config["profiles"]["baseline"]["scale_count"], BASELINE_PARAMETERS["scale_count"])
        self.assertIn(0.8, config["parameters"]["base_sigma"]["values"])
        self.assertIn(1.2, config["parameters"]["base_sigma"]["values"])
        self.assertIn(2.0, config["parameters"]["scale_ratio"]["values"])
        self.assertIn(2.5, config["parameters"]["scale_ratio"]["values"])
        self.assertIn(96, config["parameters"]["ray_count"]["values"])
        self.assertIn(144, config["parameters"]["ray_count"]["values"])
        self.assertIn(82.0, config["parameters"]["gradient_percentile"]["values"])
        self.assertIn(90.0, config["parameters"]["gradient_percentile"]["values"])


if __name__ == "__main__":
    unittest.main()
