import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.geometry.detector_projective_gradient_vote import BASELINE_PARAMETERS, debug_images, detect
from hth.regression.parameter_space import exhaustive_parameter_sets


class ProjectiveGradientVoteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = np.ones((240, 320), dtype=np.uint8) * 255

    def test_detects_perspective_quadrilateral(self) -> None:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        expected = np.array([[55, 35], [270, 22], [288, 208], [40, 218]], dtype=np.int32)
        cv2.fillConvexPoly(image, expected, (235, 235, 235))
        candidate = detect(
            image_bgr=image,
            mask=self.mask,
            parameters={"minimum_side_support": 0.05, "minimum_segment_fraction": 0.08},
        )
        self.assertEqual(candidate.method, "projective_gradient_vote")
        self.assertEqual(candidate.status, "ok")
        actual = np.asarray(candidate.corners, dtype=np.float32)
        self.assertLess(float(np.mean(np.linalg.norm(actual - expected.astype(np.float32), axis=1))), 8.0)
        self.assertEqual(len(candidate.diagnostics["side_support"]), 4)

    def test_rejects_unknown_parameter(self) -> None:
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            detect(image_bgr=image, mask=np.zeros((40, 40), dtype=np.uint8), parameters={"bogus": 1})

    def test_debug_package_exposes_projective_votes(self) -> None:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(image, (45, 30), (275, 210), (235, 235, 235), -1)
        images = debug_images(image_bgr=image, mask=self.mask, verbose=True)
        self.assertIn("projective-gradient.png", images)
        self.assertIn("projective-line-votes.png", images)
        self.assertIn("projective-segments.png", images)

    def test_initial_calibration_domain_has_729_sets(self) -> None:
        config = json.loads(Path("config/detectors/projective_gradient_vote.json").read_text(encoding="utf-8"))
        self.assertEqual(len(exhaustive_parameter_sets(config)), 729)
        self.assertEqual(config["profiles"]["baseline"]["family_tolerance_degrees"], BASELINE_PARAMETERS["family_tolerance_degrees"])


if __name__ == "__main__":
    unittest.main()
