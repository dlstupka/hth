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


class LearnedEvidenceCacheBoundaryTests(unittest.TestCase):
    MODULES = (kraken, doc_ufcn, mask_rcnn, dhsegment)

    def setUp(self):
        for module in self.MODULES:
            with module._EVIDENCE_CACHE_LOCK:
                module._EVIDENCE_CACHE.clear()
                module._PRECOMPUTED_EVIDENCE.clear()
        with kraken._RUNTIME_DIAGNOSTICS_LOCK:
            kraken._RUNTIME_DIAGNOSTICS.clear()

    def tearDown(self):
        self.setUp()

    @staticmethod
    def _images(module):
        return [
            np.full((3, 3, 3), value, dtype=np.uint8)
            for value in range(module._EVIDENCE_CACHE_LIMIT + 2)
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
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for module, evidence in cases:
                with self.subTest(detector=module.METHOD), \
                     patch.object(module, "_infer_evidence", return_value=evidence):
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
                        else:
                            filename = f"page-{index:04d}.npy"
                            np.save(output / filename, np.zeros((2, 2), dtype=np.float32))
                            records.append({
                                "image_key": key,
                                "probability_file": filename,
                                "original_shape": [3, 3],
                            })
                    (output / "manifest.json").write_text(json.dumps({
                        "detector": module.METHOD,
                        "records": records,
                    }), encoding="utf-8")
                    module.load_precomputed_golden_set_evidence(output, images)
                    model_name = "_model" if module is dhsegment else "_load_model"
                    with patch.object(module, model_name, side_effect=AssertionError("must not infer")):
                        for _ in range(2):
                            for image in images:
                                module._infer_evidence(image)
                    self.assertEqual(len(module._PRECOMPUTED_EVIDENCE), len(images))
                    with module._EVIDENCE_CACHE_LOCK:
                        module._PRECOMPUTED_EVIDENCE.clear()


if __name__ == "__main__":
    unittest.main()
