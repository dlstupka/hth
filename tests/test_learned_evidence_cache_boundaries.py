import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hth.geometry import detector_dhsegment_page_mask as dhsegment
from hth.geometry import detector_doc_ufcn_page_mask as doc_ufcn
from hth.geometry import detector_kraken_page_mask as kraken
from hth.geometry import detector_mask_rcnn_page_mask as mask_rcnn
from hth.geometry import detector_docextractor_page_mask as docextractor
from hth.geometry import detector_eynollah_page_mask as eynollah
from hth.geometry import detector_pagenet_page_mask as pagenet


class LearnedEvidenceCacheBoundaryTests(unittest.TestCase):
    MODULES = (kraken, doc_ufcn, mask_rcnn, dhsegment, docextractor, eynollah, pagenet)

    @staticmethod
    def _cache_state(module):
        if hasattr(module, "_EVIDENCE_CACHE"):
            return module._EVIDENCE_CACHE_LOCK, module._EVIDENCE_CACHE
        return module._CACHE_LOCK, module._CACHE

    def setUp(self):
        for module in self.MODULES:
            lock, cache = self._cache_state(module)
            with lock:
                cache.clear()
                module._PRECOMPUTED_EVIDENCE.clear()
        with kraken._RUNTIME_DIAGNOSTICS_LOCK:
            kraken._RUNTIME_DIAGNOSTICS.clear()

    def tearDown(self):
        self.setUp()

    @staticmethod
    def _images(module):
        limit = getattr(module, "_EVIDENCE_CACHE_LIMIT", None)
        if limit is None:
            limit = module._CACHE_LIMIT
        return [
            np.full((3, 3, 3), value, dtype=np.uint8)
            for value in range(limit + 2)
        ]

    def test_every_bounded_exporter_supports_more_pages_than_its_lru(self):
        frozen = kraken._freeze_evidence({
            "regions": [], "lines": [], "baselines": [],
            "text_direction": "horizontal-lr",
        })
        cases = (
            (kraken, frozen),
            (doc_ufcn, ()),
            (mask_rcnn, ()),
            (dhsegment, (np.zeros((2, 2), dtype=np.float32), (3, 3))),
            (docextractor, np.zeros((2, 2), dtype=np.float32)),
            (eynollah, np.zeros((2, 2), dtype=np.float32)),
            (pagenet, (np.zeros((2, 2), dtype=np.float32), {"model_id": "test"})),
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for module, evidence in cases:
                with self.subTest(detector=module.METHOD), \
                     patch.object(module, "_infer_evidence" if hasattr(module, "_infer_evidence") else ("_probability" if module is pagenet else "_infer"), return_value=evidence):
                    images = self._images(module)
                    manifest = module.export_precomputed_golden_set_evidence(
                        images, root / module.METHOD,
                    )
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    self.assertEqual(payload["page_count"], len(images))
                    self.assertEqual(len(payload["records"]), len(images))

    def test_loaded_collection_larger_than_lru_never_falls_back_to_inference(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for module in self.MODULES:
                with self.subTest(detector=module.METHOD):
                    images = self._images(module)
                    output = root / module.METHOD
                    output.mkdir()
                    records = []
                    for index, image in enumerate(images):
                        key = module._image_key(image)
                        if module is kraken:
                            records.append({
                                "image_key": key,
                                "evidence": {
                                    "regions": [], "lines": [], "baselines": [],
                                    "text_direction": "horizontal-lr",
                                },
                            })
                        elif module is doc_ufcn:
                            records.append({"image_key": key, "polygons": []})
                        elif module is mask_rcnn:
                            records.append({"image_key": key, "instances": []})
                        elif module is dhsegment:
                            filename = f"page-{index:04d}.npy"
                            np.save(output / filename, np.zeros((2, 2), dtype=np.float32))
                            records.append({
                                "image_key": key,
                                "probability_file": filename,
                                "original_shape": [3, 3],
                            })
                        else:
                            filename = f"page-{index:04d}.npy"
                            np.save(output / filename, np.zeros((2, 2), dtype=np.float32))
                            records.append({"image_key": key, "file": filename})
                    (output / "manifest.json").write_text(json.dumps({
                        "detector": module.METHOD,
                        "records": records,
                    }), encoding="utf-8")
                    module.load_precomputed_golden_set_evidence(output, images)
                    model_name = "_model" if module is dhsegment else ("_assets" if module is pagenet else "_load_model")
                    model_owner = module._impl if module is pagenet else module
                    lookup_name = "_infer_evidence" if hasattr(module, "_infer_evidence") else ("_probability" if module is pagenet else "_infer")
                    with patch.object(model_owner, model_name, side_effect=AssertionError("must not infer")):
                        for _ in range(2):
                            for image in images:
                                getattr(module, lookup_name)(image)
                    self.assertEqual(len(module._PRECOMPUTED_EVIDENCE), len(images))
                    lock, _ = self._cache_state(module)
                    with lock:
                        module._PRECOMPUTED_EVIDENCE.clear()


if __name__ == "__main__":
    unittest.main()
