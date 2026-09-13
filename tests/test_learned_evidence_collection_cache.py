from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hth.regression import learned_evidence


class LearnedEvidenceCollectionCacheTests(unittest.TestCase):
    def test_generic_sidecar_evidence_round_trips_through_runner_local_cache(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            golden = root / "golden.json"
            golden.write_text('{"pages": []}\n', encoding="utf-8")
            provenance = root / "model-provenance.json"
            provenance.write_text(json.dumps({
                "model_id": "pagenet-ohio",
                "weights_sha256": "a" * 64,
                "deploy_prototxt_sha256": "b" * 64,
                "prepared_at_utc": "changes-every-run",
                "inference_backend": "opencv-dnn-caffe",
                "input_contract": "BGR 256x256",
            }), encoding="utf-8")
            calls = []

            def exporter(images, output, *, progress=None):
                calls.append(len(images))
                output.mkdir(parents=True, exist_ok=True)
                key = learned_evidence.detector_pagenet_page_mask._image_key(images[0])
                np.save(output / f"{key}.npy", np.zeros((2, 2), dtype=np.float32))
                manifest = output / "manifest.json"
                manifest.write_text(json.dumps({
                    "schema_version": "0.1",
                    "detector": "pagenet_page_mask",
                    "representation": "pagenet-ohio-page-probability-256",
                    "page_count": 1,
                    "records": [{"image_key": key, "file": f"{key}.npy"}],
                }), encoding="utf-8")
                return manifest

            env = {
                "HTH_LEARNED_PAGE_MASK_PROVENANCE": str(provenance),
                "HTH_EVIDENCE_LOCAL_CACHE_ROOT": str(root / "local-cache"),
            }
            with patch.dict(os.environ, env, clear=False), \
                 patch.dict(learned_evidence.EXPORTERS, {"pagenet_page_mask": exporter}):
                first, first_source = learned_evidence.prepare_images(
                    detector="pagenet_page_mask",
                    golden_set=golden,
                    maximum_dimension=1800,
                    images=[image],
                    output=root / "first",
                    cache_repository="",
                )
                second, second_source = learned_evidence.prepare_images(
                    detector="pagenet_page_mask",
                    golden_set=golden,
                    maximum_dimension=1800,
                    images=[image],
                    output=root / "second",
                    cache_repository="",
                )

            self.assertEqual(calls, [1])
            self.assertEqual(first_source, "inference")
            self.assertEqual(second_source, "runner-local-cache")
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())
            second_payload = json.loads(second.read_text(encoding="utf-8"))
            self.assertTrue((second.parent / second_payload["records"][0]["file"]).is_file())

    def test_doc_ufcn_fusion_reuses_the_canonical_doc_ufcn_identity(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            golden = root / "golden.json"
            golden.write_text("{}\n", encoding="utf-8")
            provenance = root / "model-provenance.json"
            provenance.write_text(json.dumps({
                "model_id": "doc-ufcn-generic-page",
                "model_sha256": "c" * 64,
                "parameters_sha256": "d" * 64,
                "inference_backend": "doc-ufcn-pytorch",
                "serving_contract": "RGB image -> class page polygons",
            }), encoding="utf-8")
            with patch.dict(os.environ, {"HTH_DOC_UFCN_PAGE_PROVENANCE": str(provenance)}, clear=False):
                standalone = learned_evidence.evidence_identity(
                    detector="doc_ufcn_page_mask", golden_set=golden,
                    maximum_dimension=1800, images=[image],
                )
                fusion = learned_evidence.evidence_identity(
                    detector="amsre_doc_ufcn_fusion", golden_set=golden,
                    maximum_dimension=1800, images=[image],
                )
            self.assertEqual(standalone, fusion)
            self.assertEqual(fusion["detector"], "doc_ufcn_page_mask")


if __name__ == "__main__":
    unittest.main()
