import unittest
from unittest.mock import patch
import cv2
import numpy as np
from hth.geometry.model import Candidate
from hth.geometry.detector_border_energy import BASELINE_PARAMETERS, detect

class BorderEnergyTests(unittest.TestCase):
    def test_accepts_energy_supported_contour(self):
        image=np.zeros((240,320,3),dtype=np.uint8)
        cv2.rectangle(image,(45,30),(275,210),(235,235,235),-1)
        mask=np.ones((240,320),dtype=np.uint8)*255
        contour=Candidate("contour_quad",[45,30,276,211],[[45,30],[275,30],[275,210],[45,210]],0.8,0.8,{})
        with patch("hth.geometry.detector_border_energy.detector_contour_quad.detect",return_value=contour):
            candidate=detect(image_bgr=image,mask=mask,parameters={"minimum_border_energy":0.01,"minimum_side_consistency":0.10})
        self.assertEqual(candidate.method,"border_energy")
        self.assertEqual(candidate.status,"ok")
        self.assertGreater(candidate.diagnostics["border_energy"],0.0)

    def test_rejects_unknown_parameter(self):
        image=np.zeros((40,40,3),dtype=np.uint8); mask=np.zeros((40,40),dtype=np.uint8)
        with self.assertRaises(ValueError): detect(image_bgr=image,mask=mask,parameters={"bogus":1})

    def test_baseline_is_nonempty(self):
        self.assertIn("minimum_border_energy",BASELINE_PARAMETERS)
