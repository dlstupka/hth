import unittest
import cv2
import numpy as np
from hth.geometry.detector_adaptive_radial_edge import BASELINE_PARAMETERS, detect

class AdaptiveRadialEdgeTests(unittest.TestCase):
    def test_detects_high_contrast_document_boundary(self):
        image=np.zeros((240,320,3),dtype=np.uint8)
        cv2.rectangle(image,(45,30),(275,210),(235,235,235),-1)
        mask=np.ones((240,320),dtype=np.uint8)*255
        candidate=detect(image_bgr=image,mask=mask,parameters={"minimum_ray_support":0.20,"gradient_percentile":70.0,"maximum_radius_fraction":0.90})
        self.assertEqual(candidate.method,"adaptive_radial_edge")
        self.assertEqual(candidate.status,"ok")
        self.assertIsNotNone(candidate.corners)
        self.assertIn("refinement_triggered", candidate.diagnostics)

    def test_rejects_unknown_parameter(self):
        image=np.zeros((40,40,3),dtype=np.uint8); mask=np.zeros((40,40),dtype=np.uint8)
        with self.assertRaises(ValueError): detect(image_bgr=image,mask=mask,parameters={"bogus":1})

    def test_baseline_has_refinement_controls(self):
        self.assertEqual(BASELINE_PARAMETERS["coarse_angle_step_degrees"],3.0)
        self.assertEqual(BASELINE_PARAMETERS["refined_angle_step_degrees"],1.0)

if __name__ == "__main__": unittest.main()
