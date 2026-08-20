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

if __name__ == "__main__": unittest.main()
