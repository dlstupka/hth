import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.geometry.detector_adaptive_multi_scale_radial_edge import (
    BASELINE_PARAMETERS,
    PARENT_PARAMETER_SET_ID,
    debug_images,
    detect,
)
from hth.regression.parameter_space import exhaustive_parameter_sets


class AdaptiveMultiScaleRadialEdgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(self.image, (45, 30), (275, 210), (235, 235, 235), -1)
        self.mask = np.ones((240, 320), dtype=np.uint8) * 255

    def test_detects_document_with_fixed_msre_scale_space(self) -> None:
        candidate = detect(image_bgr=self.image, mask=self.mask)
        self.assertEqual(candidate.method, "adaptive_multi_scale_radial_edge")
        self.assertEqual(candidate.status, "ok")
        self.assertEqual(candidate.diagnostics["parent_parameter_set_id"], PARENT_PARAMETER_SET_ID)
        self.assertEqual(candidate.diagnostics["scale_sigmas"], [1.0, 3.5, 12.25, 42.875])
        left, top, right, bottom = candidate.bbox
        self.assertLess(abs(left - 45), 20)
        self.assertLess(abs(right - 275), 20)
        self.assertLess(abs(top - 30), 20)
        self.assertLess(abs(bottom - 210), 20)

    def test_no_refinement_control_preserves_coarse_only_sampling(self) -> None:
        candidate = detect(
            image_bgr=self.image,
            mask=self.mask,
            parameters={"maximum_refined_sides": 0},
        )
        self.assertEqual(candidate.status, "ok")
        self.assertFalse(candidate.diagnostics["refinement_triggered"])
        self.assertEqual(candidate.diagnostics["refined_supported_rays"], 0)
        self.assertEqual(candidate.diagnostics["requested_coarse_rays"], 176)

    def test_debug_package_exposes_multiscale_and_adaptive_evidence(self) -> None:
        images = debug_images(image_bgr=self.image, mask=self.mask, verbose=True)
        for name in (
            "adaptive-multi-scale-gradient.png",
            "adaptive-multi-scale-radial-points.png",
            "adaptive-multi-scale-space.png",
            "adaptive-multi-scale-pass1-fit.png",
            "adaptive-multi-scale-side-support.png",
            "adaptive-multi-scale-pass2-fit.png",
        ):
            self.assertIn(name, images)

    def test_generation_2_refinement_is_exactly_50000_angular_only_sets(self) -> None:
        config = json.loads(
            Path("config/detectors/adaptive_multi_scale_radial_edge.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(exhaustive_parameter_sets(config)), 50000)
        self.assertEqual(set(config["parameters"]), {
            "coarse_angle_step_degrees",
            "refined_angle_step_degrees",
            "weak_side_support_fraction",
            "side_assignment_tolerance_fraction",
            "maximum_refined_sides",
        })
        self.assertIn(360.0 / 176.0, config["parameters"]["coarse_angle_step_degrees"]["values"])
        self.assertIn(0, config["parameters"]["maximum_refined_sides"]["values"])
        self.assertEqual(config["profiles"]["baseline"]["base_sigma"], BASELINE_PARAMETERS["base_sigma"])
        self.assertEqual(config["profiles"]["baseline"]["gradient_percentile"], 96.875)

    def test_rejects_unknown_parameter(self) -> None:
        with self.assertRaises(ValueError):
            detect(image_bgr=self.image, mask=self.mask, parameters={"bogus": 1})


if __name__ == "__main__":
    unittest.main()
