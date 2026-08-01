import unittest
import cv2
import numpy as np
from hth.geometry.detector_gradient_vote import BASELINE_PARAMETERS, detect

class GradientVoteTests(unittest.TestCase):
    def test_detects_high_contrast_document_boundary(self):
        image=np.zeros((240,320,3),dtype=np.uint8)
        cv2.rectangle(image,(45,30),(275,210),(235,235,235),-1)
        mask=np.ones((240,320),dtype=np.uint8)*255
        candidate=detect(image_bgr=image,mask=mask,parameters={"minimum_vote_support":0.02})
        self.assertEqual(candidate.method,"gradient_vote")
        self.assertEqual(candidate.status,"ok")
        self.assertIsNotNone(candidate.corners)
        left,top,right,bottom=candidate.bbox
        self.assertLess(abs(left-45),12); self.assertLess(abs(right-275),12)
        self.assertLess(abs(top-30),12); self.assertLess(abs(bottom-210),12)

    def test_rejects_unknown_parameter(self):
        image=np.zeros((40,40,3),dtype=np.uint8); mask=np.zeros((40,40),dtype=np.uint8)
        with self.assertRaises(ValueError): detect(image_bgr=image,mask=mask,parameters={"bogus":1})

    def test_baseline_is_nonempty(self):
        self.assertIn("gradient_percentile",BASELINE_PARAMETERS)
