from __future__ import annotations

import unittest
import cv2
import numpy as np

from hth.geometry.detector_edge_contour import BASELINE_PARAMETERS, detect


class EdgeContourDetectorTests(unittest.TestCase):
    def test_detects_contour_quad_verified_by_lsd_segments(self) -> None:
        image = np.zeros((420, 620, 3), dtype=np.uint8)
        mask = np.zeros((420, 620), dtype=np.uint8)
        polygon = np.array([[80, 55], [545, 75], [520, 375], [65, 350]], dtype=np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
        cv2.polylines(image, [polygon], True, (255, 255, 255), 5)

        candidate = detect(image_bgr=image, mask=mask)

        self.assertEqual(candidate.method, "edge_contour")
        self.assertEqual(candidate.status, "ok")
        self.assertIsNotNone(candidate.corners)
        self.assertEqual(candidate.diagnostics["parameters"], BASELINE_PARAMETERS)
        self.assertGreater(candidate.diagnostics["retained_lsd_segment_count"], 0)
        self.assertGreater(candidate.diagnostics["edge_support"], 0.1)

    def test_rejects_contour_without_independent_line_evidence(self) -> None:
        image = np.zeros((300, 500, 3), dtype=np.uint8)
        mask = np.zeros((300, 500), dtype=np.uint8)
        cv2.rectangle(mask, (60, 40), (440, 260), 255, -1)

        candidate = detect(image_bgr=image, mask=mask)

        self.assertEqual(candidate.status, "no_candidate")
        self.assertEqual(candidate.diagnostics["reason"], "no_edge_verified_quadrilateral")
        self.assertEqual(candidate.diagnostics["retained_lsd_segment_count"], 0)

    def test_rejects_unknown_parameter(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "Unknown Edge-Contour parameters"):
            detect(image_bgr=image, mask=mask, parameters={"mystery": 1})


if __name__ == "__main__":
    unittest.main()
