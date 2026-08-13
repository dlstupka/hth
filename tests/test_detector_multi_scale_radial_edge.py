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

    def test_initial_calibration_domain_has_729_sets(self) -> None:
        config = json.loads(Path("config/detectors/multi_scale_radial_edge.json").read_text(encoding="utf-8"))
        self.assertEqual(len(exhaustive_parameter_sets(config)), 729)
        self.assertEqual(config["profiles"]["baseline"]["scale_count"], BASELINE_PARAMETERS["scale_count"])


if __name__ == "__main__":
    unittest.main()
