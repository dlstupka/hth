import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.geometry import detector_segment_supported_polar_vote
from hth.regression.parameter_space import exhaustive_parameter_sets


class SegmentSupportedPolarVoteTests(unittest.TestCase):
    def test_detector_contract(self) -> None:
        image = np.zeros((320, 520, 3), np.uint8)
        mask = np.zeros((320, 520), np.uint8)
        cv2.rectangle(mask, (70, 45), (450, 275), 255, -1)
        cv2.rectangle(image, (70, 45), (450, 275), (220, 220, 220), -1)
        candidate = detector_segment_supported_polar_vote.detect(image_bgr=image, mask=mask)
        self.assertEqual(candidate.method, "segment_supported_polar_vote")
        self.assertIn(candidate.status, ("ok", "no_candidate"))

    def test_lsd_line_shape_is_platform_independent(self) -> None:
        image = np.zeros((320, 520, 3), np.uint8)
        cv2.rectangle(image, (70, 45), (450, 275), (220, 220, 220), -1)
        values = detector_segment_supported_polar_vote._parameters(None)
        # This exercises the locally installed OpenCV representation.  The
        # detector must accept either common LSD shape without indexing it as
        # a fixed three-dimensional tensor.
        _mag, _raw, _supported, segments = detector_segment_supported_polar_vote._evidence(image, values)
        self.assertIsInstance(segments, list)
        for a, b in segments:
            self.assertEqual(a.shape, (2,))
            self.assertEqual(b.shape, (2,))

    def test_unknown_parameter_rejected(self) -> None:
        image = np.zeros((100, 100, 3), np.uint8)
        mask = np.zeros((100, 100), np.uint8)
        with self.assertRaises(ValueError):
            detector_segment_supported_polar_vote.detect(image_bgr=image, mask=mask, parameters={"mystery": 1})

    def test_generation_3_calibration_domain_collapses_dormant_support_and_refines_winner(self) -> None:
        config = json.loads(Path("config/detectors/segment_supported_polar_vote.json").read_text(encoding="utf-8"))
        self.assertEqual(len(exhaustive_parameter_sets(config)), 45360)
        self.assertEqual(config["parameters"]["minimum_support_fraction"]["values"], [0.20])
        self.assertEqual(config["parameters"]["minimum_segment_support_fraction"]["values"], [0.10])
        self.assertEqual(config["parameters"]["bbox_padding_fraction"]["values"], [0.0])
        self.assertIn(0.03, config["parameters"]["minimum_segment_length_fraction"]["values"])
        self.assertIn(0.005, config["parameters"]["segment_distance_fraction"]["values"])
        self.assertIn(0.72, config["parameters"]["outer_radius_fraction"]["values"])
        self.assertIn(60, config["parameters"]["ray_count"]["values"])
        self.assertIn(84.0, config["parameters"]["gradient_percentile"]["values"])
        self.assertIn(0.14, config["parameters"]["inner_radius_fraction"]["values"])
        self.assertLess(min(config["parameters"]["minimum_segment_length_fraction"]["values"]), 0.03)
        self.assertLess(min(config["parameters"]["segment_distance_fraction"]["values"]), 0.005)



if __name__ == "__main__":
    unittest.main()
