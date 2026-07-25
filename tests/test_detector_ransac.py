from __future__ import annotations

import unittest

import cv2
import numpy as np

from hth.geometry import detector_ransac
from hth.regression.adapters.ransac import pre_regression_report_sections


class RansacDetectorTests(unittest.TestCase):
    @staticmethod
    def rectangular_page() -> tuple[np.ndarray, np.ndarray]:
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        mask = np.zeros((400, 600), dtype=np.uint8)
        cv2.rectangle(mask, (70, 45), (530, 355), 255, -1)
        return image, mask

    def test_returns_candidate_for_rectangular_page(self) -> None:
        image, mask = self.rectangular_page()
        candidate = detector_ransac.detect(image_bgr=image, mask=mask)

        self.assertEqual(candidate.method, "ransac")
        self.assertIsNotNone(candidate.bbox)
        self.assertEqual(candidate.status, "ok")
        self.assertEqual(
            candidate.diagnostics["fitted_edges"],
            ["bottom", "left", "right", "top"],
        )
        self.assertGreaterEqual(candidate.diagnostics["mean_inlier_ratio"], 0.9)
        self.assertEqual(
            candidate.diagnostics["parameters"], detector_ransac.BASELINE_PARAMETERS
        )

    def test_parameter_override_changes_padding(self) -> None:
        image, mask = self.rectangular_page()
        baseline = detector_ransac.detect(image_bgr=image, mask=mask)
        padded = detector_ransac.detect(
            image_bgr=image,
            mask=mask,
            parameters={"bbox_padding_fraction": 0.02},
        )

        self.assertIsNotNone(baseline.bbox)
        self.assertIsNotNone(padded.bbox)
        assert baseline.bbox is not None and padded.bbox is not None
        self.assertLess(padded.bbox[0], baseline.bbox[0])
        self.assertLess(padded.bbox[1], baseline.bbox[1])
        self.assertEqual(
            padded.diagnostics["parameters"]["bbox_padding_fraction"], 0.02
        )

    def test_blank_mask_is_a_normal_miss(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        mask = np.zeros((200, 300), dtype=np.uint8)
        candidate = detector_ransac.detect(image_bgr=image, mask=mask)

        self.assertIsNone(candidate.bbox)
        self.assertEqual(
            candidate.diagnostics["reason"], "insufficient_edge_models"
        )

    def test_debug_images_show_sampling_fitting_and_candidate(self) -> None:
        _, mask = self.rectangular_page()
        images = detector_ransac.debug_images(mask=mask)

        self.assertEqual(
            set(images),
            {
                "boundary-samples.png",
                "fitted-edge-models.png",
                "ransac-inliers.png",
                "candidate-quadrilateral.png",
            },
        )
        for image in images.values():
            self.assertEqual(image.shape, (*mask.shape, 3))

    def test_report_sections_describe_ordered_search_stages(self) -> None:
        config = {
            "parameters": {
                "scan_samples": {"values": [140, 220, 320]},
                "minimum_scan_foreground_fraction": {
                    "values": [0.008, 0.0125, 0.02]
                },
                "residual_threshold_fraction": {
                    "values": [0.004, 0.008, 0.014]
                },
                "max_trials": {"values": [200, 400]},
                "minimum_mean_inlier_ratio": {"values": [0.25, 0.45, 0.65]},
                "minimum_bbox_area_fraction": {"values": [0.10, 0.18, 0.28]},
                "bbox_padding_fraction": {"values": [0.0, 0.008, 0.016]},
            }
        }
        sections = pre_regression_report_sections(config)

        self.assertEqual(
            [section["title"] for section in sections],
            [
                "RANSAC Boundary Sampling Algorithm",
                "RANSAC Boundary Sampling Search Space",
                "RANSAC Line Fitting Search Space",
                "RANSAC Detector Configuration",
            ],
        )

    def test_rejects_unknown_parameter(self) -> None:
        image, mask = self.rectangular_page()
        with self.assertRaisesRegex(ValueError, "Unknown RANSAC parameters"):
            detector_ransac.detect(
                image_bgr=image, mask=mask, parameters={"mystery": 1}
            )


if __name__ == "__main__":
    unittest.main()
