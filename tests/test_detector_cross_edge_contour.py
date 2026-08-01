import unittest
from unittest.mock import patch
import numpy as np
from hth.geometry.model import Candidate
from hth.geometry.detector_cross_edge_contour import detect

class CrossEdgeContourTests(unittest.TestCase):
    def test_accepts_contour_with_strong_cross_edge_transition(self):
        image=np.zeros((240,320,3),dtype=np.uint8); image[30:211,45:276]=230
        mask=np.zeros((240,320),dtype=np.uint8); mask[30:211,45:276]=255
        contour=Candidate("contour_quad",[45,30,276,211],[[45,30],[276,30],[276,211],[45,211]],0.8,0.8,{})
        with patch("hth.geometry.detector_cross_edge_contour.detector_contour_quad.detect",return_value=contour):
            candidate=detect(image_bgr=image,mask=mask,parameters={"minimum_cross_edge_contrast":0.01})
        self.assertEqual(candidate.method,"cross_edge_contour")
        self.assertEqual(candidate.status,"ok")
        self.assertGreater(candidate.diagnostics["cross_edge_contrast"],0.1)

    def test_rejects_flat_image(self):
        image=np.ones((120,160,3),dtype=np.uint8)*128; mask=np.ones((120,160),dtype=np.uint8)*255
        contour=Candidate("contour_quad",[20,20,140,100],[[20,20],[140,20],[140,100],[20,100]],0.8,0.8,{})
        with patch("hth.geometry.detector_cross_edge_contour.detector_contour_quad.detect",return_value=contour):
            candidate=detect(image_bgr=image,mask=mask)
        self.assertEqual(candidate.status,"no_candidate")
        self.assertEqual(candidate.diagnostics["reason"],"insufficient_cross_edge_contrast")
