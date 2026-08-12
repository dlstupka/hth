from __future__ import annotations
import unittest
import cv2
import numpy as np
from hth.geometry import detector_distance_transform

class DistanceTransformDetectorTests(unittest.TestCase):
    def test_detects_dominant_page_region_despite_small_noise(self):
        image = np.zeros((320, 520, 3), dtype=np.uint8)
        mask = np.zeros((320, 520), dtype=np.uint8)
        cv2.rectangle(mask, (70, 45), (450, 275), 255, -1)
        cv2.circle(mask, (15, 15), 5, 255, -1)
        cv2.circle(mask, (500, 300), 4, 255, -1)
        candidate = detector_distance_transform.detect(image_bgr=image, mask=mask)
        self.assertEqual(candidate.status, "ok")
        self.assertEqual(candidate.method, "distance_transform")
        self.assertLess(abs(candidate.bbox[0] - 70), 3)
        self.assertLess(abs(candidate.bbox[1] - 45), 3)
        self.assertGreater(candidate.diagnostics["core_area_fraction"], 0.01)
        self.assertIsNotNone(candidate.corners)

    def test_blank_mask_is_normal_miss(self):
        image=np.zeros((200,300,3),np.uint8); mask=np.zeros((200,300),np.uint8)
        candidate=detector_distance_transform.detect(image_bgr=image,mask=mask)
        self.assertEqual(candidate.status,"no_candidate")
        self.assertEqual(candidate.diagnostics["reason"],"empty_foreground")

    def test_unknown_parameter_is_rejected(self):
        image=np.zeros((100,100,3),np.uint8); mask=np.zeros((100,100),np.uint8)
        with self.assertRaisesRegex(ValueError, "Unknown Distance Transform parameters"):
            detector_distance_transform.detect(image_bgr=image, mask=mask, parameters={"mystery":1})

if __name__ == "__main__":
    unittest.main()
