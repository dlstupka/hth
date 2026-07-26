from __future__ import annotations

import cv2
import numpy as np
import unittest

from hth.geometry.detector_contour_quad import BASELINE_PARAMETERS, detect

class ContourQuadDetectorTests(unittest.TestCase):

    def test_detects_perspective_quadrilateral(self) -> None:
        image = np.zeros((420, 620, 3), dtype=np.uint8)
        mask = np.zeros((420, 620), dtype=np.uint8)
        polygon = np.array([[80, 55], [545, 75], [520, 375], [65, 350]], dtype=np.int32)
        cv2.fillConvexPoly(mask, polygon, 255)
        cv2.polylines(image, [polygon], True, (255, 255, 255), 4)

        candidate = detect(image_bgr=image, mask=mask)

        assert candidate.method == "contour_quad"
        assert candidate.status == "ok"
        assert candidate.corners is not None
        assert len(candidate.corners) == 4
        assert candidate.bbox is not None
        assert candidate.diagnostics["parameters"] == BASELINE_PARAMETERS
        assert candidate.diagnostics["quadrilateral_count"] >= 1
        assert candidate.diagnostics["angle_score"] > 0.8


    def test_uses_edge_evidence_as_hybrid_score_component(self) -> None:
        image = np.zeros((300, 500, 3), dtype=np.uint8)
        mask = np.zeros((300, 500), dtype=np.uint8)
        cv2.rectangle(mask, (60, 40), (440, 260), 255, -1)
        cv2.rectangle(image, (60, 40), (440, 260), (255, 255, 255), 3)

        candidate = detect(image_bgr=image, mask=mask)

        assert candidate.status == "ok"
        assert candidate.diagnostics["edge_support"] > 0.5
        assert candidate.diagnostics["rectangularity"] > 0.95


    def test_rejects_shape_that_never_forms_a_quadrilateral(self) -> None:
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        mask = np.zeros((300, 300), dtype=np.uint8)
        cv2.circle(mask, (150, 150), 100, 255, -1)

        candidate = detect(
            image_bgr=image,
            mask=mask,
            parameters={"epsilon_min_fraction": 0.004, "epsilon_max_fraction": 0.01},
        )

        assert candidate.status == "no_candidate"
        assert candidate.bbox is None
        assert candidate.diagnostics["reason"] == "no_plausible_quadrilateral"


    def test_rejects_unknown_parameter(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "Unknown Contour Quadrilateral parameters"):
            detect(image_bgr=image, mask=mask, parameters={"mystery": 1})


    def test_rejects_invalid_weight_set(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "at least one quadrilateral score weight"):
            detect(
                image_bgr=image,
                mask=mask,
                parameters={
                    "area_weight": 0,
                    "rectangularity_weight": 0,
                    "angle_weight": 0,
                    "edge_support_weight": 0,
                },
            )
