import json
import unittest
from pathlib import Path

import cv2
import numpy as np
from hth.geometry.detector_radial_edge import BASELINE_PARAMETERS, detect
from hth.regression.parameter_space import exhaustive_parameter_sets

class RadialEdgeTests(unittest.TestCase):
    def test_detects_high_contrast_document_boundary(self):
        image=np.zeros((240,320,3),dtype=np.uint8)
        cv2.rectangle(image,(45,30),(275,210),(235,235,235),-1)
        mask=np.ones((240,320),dtype=np.uint8)*255
        candidate=detect(image_bgr=image,mask=mask,parameters={"minimum_ray_support":0.20,"gradient_percentile":70.0,"maximum_radius_fraction":0.90})
        self.assertEqual(candidate.method,"radial_edge")
        self.assertEqual(candidate.status,"ok")
        self.assertIsNotNone(candidate.corners)
        left,top,right,bottom=candidate.bbox
        self.assertLess(abs(left-45),18); self.assertLess(abs(right-275),18)
        self.assertLess(abs(top-30),18); self.assertLess(abs(bottom-210),18)

    def test_rejects_unknown_parameter(self):
        image=np.zeros((40,40,3),dtype=np.uint8); mask=np.zeros((40,40),dtype=np.uint8)
        with self.assertRaises(ValueError): detect(image_bgr=image,mask=mask,parameters={"bogus":1})

    def test_baseline_is_nonempty(self):
        self.assertIn("ray_count",BASELINE_PARAMETERS)

    def test_generation_2_calibration_domain_is_broad_and_retains_baseline(self):
        config = json.loads(Path("config/detectors/radial_edge.json").read_text(encoding="utf-8"))
        self.assertEqual(len(exhaustive_parameter_sets(config)), 50000)
        self.assertIn(BASELINE_PARAMETERS["gaussian_sigma"], config["parameters"]["gaussian_sigma"]["values"])
        self.assertIn(BASELINE_PARAMETERS["ray_count"], config["parameters"]["ray_count"]["values"])
        self.assertIn(BASELINE_PARAMETERS["gradient_percentile"], config["parameters"]["gradient_percentile"]["values"])
        self.assertIn(BASELINE_PARAMETERS["minimum_ray_support"], config["parameters"]["minimum_ray_support"]["values"])
