from __future__ import annotations
import json
from pathlib import Path
import unittest
import numpy as np
from hth.geometry import detector_mask_rcnn_page_mask as detector
from hth.geometry.registry import detector_names
from hth.regression.runner import PRECOMPUTED_EVIDENCE_LOADERS, PRECOMPUTED_EVIDENCE_PREPARERS

ROOT = Path(__file__).resolve().parents[1]

class MaskRCNNPageMaskIntegrationTests(unittest.TestCase):
    def test_registered_and_shared_evidence_enabled(self):
        self.assertIn("mask_rcnn_page_mask", detector_names())
        config=json.loads((ROOT/"config/detectors/mask_rcnn_page_mask.json").read_text(encoding="utf-8"))
        self.assertEqual(config["lifecycle"]["prepare"], "mask_rcnn_page_mask")
        self.assertIn("mask_rcnn_page_mask", PRECOMPUTED_EVIDENCE_PREPARERS)
        self.assertIn("mask_rcnn_page_mask", PRECOMPUTED_EVIDENCE_LOADERS)

    def test_targeted_refinement_grid_focuses_confidence_and_padding(self):
        config=json.loads((ROOT/"config/detectors/mask_rcnn_page_mask.json").read_text(encoding="utf-8"))
        params=config["parameters"]
        self.assertEqual(params["minimum_confidence"]["values"], [0.0,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25])
        self.assertEqual(params["minimum_instance_area_fraction"]["values"], [0.0])
        self.assertEqual(params["minimum_page_area_fraction"]["values"], [0.25])
        self.assertEqual(params["page_padding_fraction"]["values"], [0.0,0.005,0.01,0.015,0.02,0.025,0.03,0.035,0.04])
        total=1
        for spec in params.values():
            total *= len(spec["values"])
        self.assertEqual(total, 90)

    def test_detect_uses_precomputed_instance_without_runtime(self):
        image=np.zeros((100,120,3),dtype=np.uint8)
        key=detector._image_key(image)
        record={"confidence":0.9,"class_id":0,"area":7200.0,"bounds":[10,10,110,90],"polygon":[[10,10],[110,10],[110,90],[10,90]]}
        with detector._EVIDENCE_CACHE_LOCK:
            detector._EVIDENCE_CACHE[key]=(record,)
        candidate=detector.detect(image_bgr=image,mask=None,parameters={"minimum_page_area_fraction":0.1,"page_padding_fraction":0.0})
        self.assertEqual(candidate.status,"ok")
        self.assertEqual(candidate.method,"mask_rcnn_page_mask")
        self.assertGreater(candidate.score,0.5)
        self.assertEqual(candidate.diagnostics["selection"],"largest-instance")


    def test_detect_builds_coherent_two_leaf_envelope(self):
        image=np.zeros((100,200,3),dtype=np.uint8)
        key=detector._image_key(image)
        left={"confidence":0.95,"class_id":0,"area":7200.0,"bounds":[5,10,95,90],"polygon":[[5,10],[95,10],[95,90],[5,90]]}
        right={"confidence":0.90,"class_id":1,"area":7200.0,"bounds":[100,10,190,90],"polygon":[[100,10],[190,10],[190,90],[100,90]]}
        with detector._EVIDENCE_CACHE_LOCK:
            detector._EVIDENCE_CACHE[key]=(left,right)
        candidate=detector.detect(image_bgr=image,mask=None,parameters={"minimum_confidence":0.0,"minimum_instance_area_fraction":0.0,"minimum_page_area_fraction":0.08,"page_padding_fraction":0.0})
        self.assertEqual(candidate.status,"ok")
        self.assertEqual(candidate.diagnostics["selection"],"coherent-multi-instance-envelope")
        self.assertEqual(candidate.diagnostics["selected_instance_count"],2)
        self.assertLessEqual(candidate.bbox[0],6)
        self.assertGreaterEqual(candidate.bbox[2],189)

    def test_detect_rejects_detached_fragment_from_envelope(self):
        image=np.zeros((100,200,3),dtype=np.uint8)
        key=detector._image_key(image)
        primary={"confidence":0.95,"class_id":0,"area":11200.0,"bounds":[10,10,150,90],"polygon":[[10,10],[150,10],[150,90],[10,90]]}
        detached={"confidence":0.90,"class_id":2,"area":1000.0,"bounds":[185,0,199,70],"polygon":[[185,0],[199,0],[199,70],[185,70]]}
        with detector._EVIDENCE_CACHE_LOCK:
            detector._EVIDENCE_CACHE[key]=(primary,detached)
        candidate=detector.detect(image_bgr=image,mask=None,parameters={"minimum_confidence":0.0,"minimum_instance_area_fraction":0.0,"minimum_page_area_fraction":0.08,"page_padding_fraction":0.0})
        self.assertEqual(candidate.status,"ok")
        self.assertEqual(candidate.diagnostics["selection"],"largest-instance")
        self.assertLess(candidate.bbox[2],170)

    def test_debug_images_uses_cached_evidence_without_runtime(self):
        image=np.zeros((100,120,3),dtype=np.uint8)
        key=detector._image_key(image)
        record={"confidence":0.9,"class_id":0,"area":7200.0,"bounds":[10,10,110,90],"polygon":[[10,10],[110,10],[110,90],[10,90]]}
        with detector._EVIDENCE_CACHE_LOCK:
            detector._EVIDENCE_CACHE[key]=(record,)
        images=detector.debug_images(
            image_bgr=image,
            mask=None,
            parameters={"minimum_page_area_fraction":0.1,"page_padding_fraction":0.0},
            candidate_corners=[[10,10],[110,10],[110,90],[10,90]],
        )
        self.assertEqual(set(images), {"mask-rcnn-page-instances.png"})
        self.assertEqual(images["mask-rcnn-page-instances.png"].shape, image.shape)

if __name__ == "__main__": unittest.main()
