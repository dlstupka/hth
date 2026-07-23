from __future__ import annotations

import cv2
import numpy as np
import unittest

from hth.geometry.detector_grabcut import BASELINE_PARAMETERS, detect

class GrabCutDetectorTests(unittest.TestCase):

    def test_grabcut_refines_seeded_page_region(self) -> None:
        image = np.full((420, 620, 3), (28, 35, 42), dtype=np.uint8)
        cv2.rectangle(image, (70, 45), (550, 380), (220, 225, 230), -1)
        cv2.rectangle(image, (82, 58), (538, 367), (238, 238, 238), 3)
        mask = np.zeros((420, 620), dtype=np.uint8)
        cv2.rectangle(mask, (62, 38), (558, 388), 255, -1)

        candidate = detect(image_bgr=image, mask=mask)
        assert candidate.method == "grabcut"
        assert candidate.bbox is not None
        assert candidate.confidence > 0.45
        assert candidate.diagnostics["iterations"] == 3
        assert candidate.diagnostics["parameters"] == BASELINE_PARAMETERS
        assert candidate.diagnostics["refined_foreground_fraction"] > 0.1


    def test_grabcut_accepts_black_box_parameter_overrides(self) -> None:
        image = np.full((320, 460, 3), (20, 25, 30), dtype=np.uint8)
        cv2.rectangle(image, (50, 35), (410, 290), (225, 225, 225), -1)
        mask = np.zeros((320, 460), dtype=np.uint8)
        cv2.rectangle(mask, (44, 30), (416, 296), 255, -1)

        candidate = detect(
            image_bgr=image,
            mask=mask,
            parameters={"grabcut_iterations": 1, "close_iterations": 2},
        )
        assert candidate.bbox is not None
        assert candidate.diagnostics["iterations"] == 1
        assert candidate.diagnostics["parameters"]["close_iterations"] == 2


    def test_grabcut_rejects_unknown_parameter(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "Unknown GrabCut parameters"):
            detect(image_bgr=image, mask=mask, parameters={"mystery": 1})


    def test_grabcut_cleanly_rejects_empty_seed(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=np.uint8)
        candidate = detect(image_bgr=image, mask=mask)
        assert candidate.bbox is None
        assert candidate.diagnostics["reason"] == "insufficient_initial_foreground"
