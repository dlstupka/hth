from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from hth.geometry.detector_consensus_quad import detect
from hth.geometry.model import Candidate


class ConsensusQuadDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((240, 360, 3), dtype=np.uint8)
        self.mask = np.zeros((240, 360), dtype=np.uint8)
        cv2.rectangle(self.mask, (40, 30), (320, 210), 255, -1)

    def test_regression_config_defines_profiles_baseline(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "detectors" / "consensus_quad.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertIsInstance(config.get("profiles", {}).get("baseline"), dict)
        self.assertTrue(config["profiles"]["baseline"])

    def test_returns_consensus_for_agreeing_quad_voters(self) -> None:
        first = Candidate("contour_quad", [40, 30, 320, 210], [[40,30],[320,30],[320,210],[40,210]], .9, .9, {})
        second = Candidate("edge_contour", [42, 31, 318, 209], [[42,31],[318,31],[318,209],[42,209]], .8, .8, {})
        with patch("hth.geometry.detector_consensus_quad.detector_contour_quad.detect", return_value=first), patch("hth.geometry.detector_consensus_quad.detector_edge_contour.detect", return_value=second):
            candidate = detect(image_bgr=self.image, mask=self.mask)
        self.assertEqual(candidate.method, "consensus_quad")
        self.assertEqual(candidate.status, "ok")
        self.assertIsNotNone(candidate.corners)
        self.assertGreater(candidate.diagnostics["polygon_iou"], 0.9)
        self.assertEqual(candidate.diagnostics["reason"], "consensus")

    def test_rejects_disagreeing_quad_voters(self) -> None:
        first = Candidate("contour_quad", [20, 20, 160, 120], [[20,20],[160,20],[160,120],[20,120]], .9, .9, {})
        second = Candidate("edge_contour", [200, 100, 340, 220], [[200,100],[340,100],[340,220],[200,220]], .8, .8, {})
        with patch("hth.geometry.detector_consensus_quad.detector_contour_quad.detect", return_value=first), patch("hth.geometry.detector_consensus_quad.detector_edge_contour.detect", return_value=second):
            candidate = detect(image_bgr=self.image, mask=self.mask)
        self.assertEqual(candidate.status, "no_candidate")
        self.assertEqual(candidate.diagnostics["reason"], "polygon_iou_below_minimum")

    def test_requires_both_quad_votes(self) -> None:
        first = Candidate("contour_quad", [40, 30, 320, 210], [[40,30],[320,30],[320,210],[40,210]], .9, .9, {})
        second = Candidate("edge_contour", None, None, 0.0, 0.0, {"reason":"none"}, status="no_candidate")
        with patch("hth.geometry.detector_consensus_quad.detector_contour_quad.detect", return_value=first), patch("hth.geometry.detector_consensus_quad.detector_edge_contour.detect", return_value=second):
            candidate = detect(image_bgr=self.image, mask=self.mask)
        self.assertEqual(candidate.status, "no_candidate")
        self.assertEqual(candidate.diagnostics["available_votes"], 1)


if __name__ == "__main__":
    unittest.main()
