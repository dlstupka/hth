from __future__ import annotations

import cv2
import numpy as np
import unittest

from hth.geometry.detector_contour import BASELINE_PARAMETERS, detect

class ContourDetectorTests(unittest.TestCase):

    def test_contour_detects_large_document_region(self) -> None:
        image = np.zeros((420, 620, 3), dtype=np.uint8)
        mask = np.zeros((420, 620), dtype=np.uint8)
        cv2.rectangle(mask, (70, 45), (550, 380), 255, -1)

        candidate = detect(image_bgr=image, mask=mask)

        assert candidate.method == "contour"
        assert candidate.bbox == [70, 45, 551, 381]
        assert candidate.corners is not None
        assert candidate.confidence > 0.9
        assert candidate.diagnostics["parameters"] == BASELINE_PARAMETERS
        assert candidate.diagnostics["corner_source"] == "approx_poly_dp"


    def test_contour_accepts_black_box_parameter_overrides(self) -> None:
        image = np.zeros((300, 500, 3), dtype=np.uint8)
        mask = np.zeros((300, 500), dtype=np.uint8)
        cv2.rectangle(mask, (60, 40), (440, 260), 255, -1)

        candidate = detect(
            image_bgr=image,
            mask=mask,
            parameters={
                "minimum_contour_area_fraction": 0.05,
                "bbox_padding_fraction": 0.01,
                "rectangularity_weight": 0.4,
            },
        )

        assert candidate.bbox == [57, 37, 444, 264]
        assert candidate.diagnostics["parameters"]["bbox_padding_fraction"] == 0.01
        assert candidate.diagnostics["parameters"]["rectangularity_weight"] == 0.4


    def test_contour_closing_can_join_fragmented_document_mask(self) -> None:
        image = np.zeros((240, 360, 3), dtype=np.uint8)
        mask = np.zeros((240, 360), dtype=np.uint8)
        cv2.rectangle(mask, (40, 35), (172, 205), 255, -1)
        cv2.rectangle(mask, (180, 35), (320, 205), 255, -1)

        candidate = detect(
            image_bgr=image,
            mask=mask,
            parameters={
                "minimum_contour_area_fraction": 0.10,
                "close_kernel_fraction": 0.04,
                "close_iterations": 1,
            },
        )

        assert candidate.bbox is not None
        assert candidate.bbox[0] <= 40
        assert candidate.bbox[2] >= 321
        assert candidate.diagnostics["close_kernel_size"] > 0



    def test_contour_can_merge_sparse_fragments_into_document_hull(self) -> None:
        image = np.zeros((300, 500, 3), dtype=np.uint8)
        mask = np.zeros((300, 500), dtype=np.uint8)
        for x, y in ((60, 40), (420, 40), (60, 250), (420, 250)):
            cv2.rectangle(mask, (x, y), (x + 12, y + 12), 255, -1)

        candidate = detect(
            image_bgr=image,
            mask=mask,
            parameters={
                "minimum_contour_area_fraction": 0.20,
                "merge_fragmented_contours": True,
            },
        )

        self.assertEqual(candidate.status, "ok")
        self.assertIsNotNone(candidate.bbox)
        self.assertEqual(candidate.diagnostics["contour_source"], "merged_convex_hull")

    def test_contour_rejects_unknown_parameter(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "Unknown Contour parameters"):
            detect(image_bgr=image, mask=mask, parameters={"mystery": 1})


    def test_contour_cleanly_rejects_small_regions(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (30, 30), 255, -1)

        candidate = detect(image_bgr=image, mask=mask)

        assert candidate.bbox is None
        assert candidate.status == "no_candidate"
        assert candidate.diagnostics["reason"] == "no_plausible_contour"
