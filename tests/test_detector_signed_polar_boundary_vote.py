import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.geometry import detector_signed_polar_boundary_vote
from hth.regression.parameter_space import exhaustive_parameter_sets


class SignedPolarBoundaryVoteTests(unittest.TestCase):
    def test_detector_contract(self) -> None:
        image = np.zeros((320, 520, 3), np.uint8)
        mask = np.zeros((320, 520), np.uint8)
        cv2.rectangle(mask, (70, 45), (450, 275), 255, -1)
        cv2.rectangle(image, (70, 45), (450, 275), (220, 220, 220), -1)
        candidate = detector_signed_polar_boundary_vote.detect(image_bgr=image, mask=mask)
        self.assertEqual(candidate.method, "signed_polar_boundary_vote")
        self.assertIn(candidate.status, ("ok", "no_candidate"))

    def test_unknown_parameter_rejected(self) -> None:
        image = np.zeros((100, 100, 3), np.uint8)
        mask = np.zeros((100, 100), np.uint8)
        with self.assertRaises(ValueError):
            detector_signed_polar_boundary_vote.detect(image_bgr=image, mask=mask, parameters={"mystery": 1})

    def test_generation_2_calibration_domain_expands_winner_boundaries(self) -> None:
        config = json.loads(Path("config/detectors/signed_polar_boundary_vote.json").read_text(encoding="utf-8"))
        self.assertEqual(len(exhaustive_parameter_sets(config)), 453600)
        self.assertIn(0.45, config["parameters"]["outer_radius_fraction"]["values"])
        self.assertIn(0.22, config["parameters"]["inner_radius_fraction"]["values"])
        self.assertIn(95.0, config["parameters"]["gradient_percentile"]["values"])
        self.assertIn(0.0, config["parameters"]["bbox_padding_fraction"]["values"])
        self.assertIn("absolute", config["parameters"]["polarity"]["values"])


if __name__ == "__main__":
    unittest.main()
