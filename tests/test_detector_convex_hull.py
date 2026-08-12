from __future__ import annotations
import unittest
import cv2
import numpy as np
from hth.geometry import detector_convex_hull

class ConvexHullDetectorTests(unittest.TestCase):
    def test_detects_fragmented_rectangular_document(self):
        image = np.zeros((300, 500, 3), dtype=np.uint8)
        mask = np.zeros((300, 500), dtype=np.uint8)
        cv2.rectangle(mask, (60, 45), (220, 250), 255, -1)
        cv2.rectangle(mask, (235, 45), (440, 250), 255, -1)
        candidate = detector_convex_hull.detect(
            image_bgr=image, mask=mask,
            parameters={"minimum_solidity": 0.45},
        )
        self.assertEqual(candidate.status, "ok")
        self.assertEqual(candidate.method, "convex_hull")
        self.assertLessEqual(candidate.bbox[0], 60)
        self.assertGreaterEqual(candidate.bbox[2], 441)
        self.assertIsNotNone(candidate.corners)
        self.assertGreater(candidate.diagnostics["hull_area_fraction"], 0.4)

    def test_rejects_sparse_low_solidity_fragments(self):
        image = np.zeros((300, 500, 3), dtype=np.uint8)
        mask = np.zeros((300, 500), dtype=np.uint8)
        for x, y in ((30,30),(450,30),(30,260),(450,260)):
            cv2.rectangle(mask, (x,y), (x+5,y+5), 255, -1)
        candidate = detector_convex_hull.detect(
            image_bgr=image, mask=mask,
            parameters={"minimum_fragment_area_fraction": 0.0, "minimum_solidity": 0.5},
        )
        self.assertEqual(candidate.status, "no_candidate")
        self.assertEqual(candidate.diagnostics["reason"], "insufficient_solidity")

    def test_unknown_parameter_is_rejected(self):
        image=np.zeros((100,100,3),np.uint8); mask=np.zeros((100,100),np.uint8)
        with self.assertRaisesRegex(ValueError, "Unknown Convex Hull parameters"):
            detector_convex_hull.detect(image_bgr=image, mask=mask, parameters={"mystery":1})

if __name__ == "__main__":
    unittest.main()
