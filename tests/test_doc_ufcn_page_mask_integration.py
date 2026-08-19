import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hth.geometry import detector_doc_ufcn_page_mask as detector
from hth.geometry.registry import detector_names
from hth.regression.runner import PRECOMPUTED_EVIDENCE_LOADERS, PRECOMPUTED_EVIDENCE_PREPARERS

ROOT = Path(__file__).resolve().parents[1]


class DocUFCNPageMaskIntegrationTests(unittest.TestCase):
    def test_registry_config_lifecycle_and_workflows(self):
        self.assertIn("doc_ufcn_page_mask", detector_names())
        config = json.loads((ROOT / "config/detectors/doc_ufcn_page_mask.json").read_text(encoding="utf-8"))
        self.assertEqual(config["detector"], "doc_ufcn_page_mask")
        self.assertEqual(config["lifecycle"]["prepare"], "doc_ufcn_page_mask")
        self.assertEqual(len(config["parameters"]["minimum_confidence"]["values"]), 8)
        self.assertIn("doc_ufcn_page_mask", PRECOMPUTED_EVIDENCE_PREPARERS)
        self.assertIn("doc_ufcn_page_mask", PRECOMPUTED_EVIDENCE_LOADERS)
        for workflow in ("regress-detector.yml", "execution-optimizer.yml"):
            text = (ROOT / ".github/workflows" / workflow).read_text(encoding="utf-8")
            self.assertIn("          - doc_ufcn_page_mask\n", text)
            self.assertIn("HTH_NEED_DOC_UFCN", text)

    def test_parameter_count_is_2000(self):
        from hth.regression.strategies.cartesian import generate
        config = json.loads((ROOT / "config/detectors/doc_ufcn_page_mask.json").read_text(encoding="utf-8"))
        self.assertEqual(len(generate(config)), 8 * 5 * 5 * 10)

    def test_detect_selects_largest_qualifying_page_polygon(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        evidence = (
            {"confidence": 0.9, "polygon": [[10, 10], [10, 90], [110, 90], [110, 10]], "area": 8000.0},
            {"confidence": 0.99, "polygon": [[1, 1], [1, 5], [5, 5], [5, 1]], "area": 16.0},
        )
        provenance = {"model_id": "doc-ufcn-generic-page", "model_sha256": "abc"}
        with patch.object(detector, "_infer_evidence", return_value=evidence), patch.object(detector, "_provenance", return_value=provenance):
            candidate = detector.detect(
                image_bgr=image,
                mask=None,
                parameters={
                    "minimum_confidence": 0.5,
                    "minimum_component_area_fraction": 0.001,
                    "minimum_page_area_fraction": 0.1,
                    "page_padding_fraction": 0.0,
                },
            )
        self.assertEqual(candidate.method, "doc_ufcn_page_mask")
        self.assertIsNotNone(candidate.corners)
        self.assertGreater(candidate.score, 0.5)

    def test_shared_evidence_round_trip(self):
        image = np.zeros((32, 48, 3), dtype=np.uint8)
        key = detector._image_key(image)
        evidence = ({"confidence": 0.8, "polygon": [[1, 1], [1, 30], [46, 30], [46, 1]], "area": 1305.0},)
        import tempfile
        with tempfile.TemporaryDirectory() as tmp, patch.object(detector, "precompute_golden_set_evidence", return_value=(key,)):
            with detector._EVIDENCE_CACHE_LOCK:
                detector._EVIDENCE_CACHE[key] = evidence
            detector.export_precomputed_golden_set_evidence([image], tmp)
            with detector._EVIDENCE_CACHE_LOCK:
                detector._EVIDENCE_CACHE.clear()
            detector.load_precomputed_golden_set_evidence(tmp, [image])
            with detector._EVIDENCE_CACHE_LOCK:
                loaded = detector._EVIDENCE_CACHE[key]
            self.assertEqual(loaded[0]["confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
