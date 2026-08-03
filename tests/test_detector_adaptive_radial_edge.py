import unittest

import cv2
import numpy as np

from hth.geometry.detector_adaptive_radial_edge import (
    BASELINE_PARAMETERS,
    _side_support,
    debug_images,
    detect,
)


class AdaptiveRadialEdgeTests(unittest.TestCase):
    def test_detects_high_contrast_document_boundary(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(image, (45, 30), (275, 210), (235, 235, 235), -1)
        mask = np.ones((240, 320), dtype=np.uint8) * 255
        candidate = detect(
            image_bgr=image,
            mask=mask,
            parameters={
                "minimum_ray_support": 0.20,
                "gradient_percentile": 70.0,
                "maximum_radius_fraction": 0.90,
            },
        )
        self.assertEqual(candidate.method, "adaptive_radial_edge")
        self.assertEqual(candidate.status, "ok")
        self.assertIsNotNone(candidate.corners)
        self.assertIn("refinement_triggered", candidate.diagnostics)
        self.assertIn("side_eligible_rays", candidate.diagnostics)

    def test_rejects_unknown_parameter(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        with self.assertRaises(ValueError):
            detect(image_bgr=image, mask=mask, parameters={"bogus": 1})

    def test_baseline_has_refinement_controls(self):
        self.assertEqual(BASELINE_PARAMETERS["coarse_angle_step_degrees"], 3.0)
        self.assertEqual(BASELINE_PARAMETERS["refined_angle_step_degrees"], 1.0)

    def test_side_support_is_normalized_by_each_sides_eligible_rays(self):
        center = np.array([50.0, 50.0], dtype=np.float32)
        corners = np.array(
            [[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]],
            dtype=np.float32,
        )
        requested_angles = np.array(
            [0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0], dtype=np.float32
        )
        # Right, bottom, and top are confirmed; the left-eligible ray has no
        # accepted endpoint.  Every side still has its own denominator of one.
        point_angles = np.array(
            [0.0, np.pi / 2.0, 3.0 * np.pi / 2.0], dtype=np.float32
        )
        points = np.array(
            [[80.0, 50.0], [50.0, 80.0], [50.0, 20.0]], dtype=np.float32
        )
        eligible, accepted, fractions = _side_support(
            points,
            point_angles,
            requested_angles,
            center,
            corners,
            diagonal=100.0,
            tolerance_fraction=0.03,
        )
        self.assertEqual(eligible, [1, 1, 1, 1])
        self.assertEqual(accepted, [1, 1, 1, 0])
        self.assertEqual(fractions, [1.0, 1.0, 1.0, 0.0])

    def test_verbose_debug_tells_pass_one_and_pass_two_story(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(image, (45, 30), (275, 210), (235, 235, 235), -1)
        mask = np.ones((240, 320), dtype=np.uint8) * 255
        images = debug_images(
            image_bgr=image,
            mask=mask,
            parameters={
                "minimum_ray_support": 0.20,
                "gradient_percentile": 70.0,
                "maximum_radius_fraction": 0.90,
            },
            verbose=True,
        )
        self.assertEqual(
            list(images),
            [
                "adaptive-radial-gradient.png",
                "adaptive-radial-edge-points.png",
                "pass1-rays.png",
                "pass1-fit-overlay.jpg",
                "side-support.png",
                "pass2-rays.png",
                "pass2-fit-overlay.jpg",
            ],
        )


if __name__ == "__main__":
    unittest.main()
