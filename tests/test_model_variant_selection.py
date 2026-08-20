from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hth.calibration_store import resolve_best_parameter_reference
from hth.model_variants import resolve_model_variant


ROOT = Path(__file__).resolve().parents[1]


class ModelVariantSelectionTests(unittest.TestCase):
    def test_mask_rcnn_default_resolves_current_and_older_is_distinct(self):
        current = resolve_model_variant("mask_rcnn_page_mask", "default")
        older = resolve_model_variant("mask_rcnn_page_mask", "rcnn_hjdataset_older")
        self.assertEqual(current.key, "rcnn_hjdataset_current")
        self.assertTrue(current.is_current)
        self.assertEqual(older.key, "rcnn_hjdataset_older")
        self.assertNotEqual(current.model_id, older.model_id)
        self.assertNotEqual(current.model_url, older.model_url)

    def test_variant_cannot_be_applied_to_another_detector(self):
        with self.assertRaisesRegex(ValueError, "belongs to detector"):
            resolve_model_variant("doc_ufcn_page_mask", "rcnn_hjdataset_older")

    def test_manual_workflow_exposes_generic_model_variant_selector(self):
        workflow = (ROOT / ".github/workflows/regress-detector.yml").read_text(encoding="utf-8")
        self.assertIn("model_variant:", workflow)
        self.assertIn("rcnn_hjdataset_current", workflow)
        self.assertIn("rcnn_hjdataset_older", workflow)
        self.assertIn("HTH_MODEL_VARIANT:", workflow)

    def test_older_variant_does_not_inherit_legacy_current_historic_best(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = root / "record"
            record.mkdir()
            (record / "parameter-provenance.json").write_text(json.dumps({
                "schema_version": "1.0",
                "identity": {"detector": "mask_rcnn_page_mask", "parameter_schema_version": "1"},
                "grid": {"parameter_order": [], "values": {}, "cartesian_count": 0},
                "explicit_parameter_sets": {
                    "f" * 64: {
                        "legacy_parameter_set_id": "abc123",
                        "sha256": "f" * 64,
                        "parameters": {"minimum_confidence": 0.0},
                    }
                },
            }), encoding="utf-8")
            index = {
                "entries": [{
                    "detector_id": "mask_rcnn_page_mask",
                    "golden_set_sha256": "golden",
                    "calibration_status": "authoritative",
                    "created_at_utc": "2026-08-20T00:00:00Z",
                    "parameter_provenance_path": "record/parameter-provenance.json",
                    "selection": {"recommended_parameter_set_id": "abc123", "best_avg_iou": 0.9},
                    "build": {},
                }]
            }
            index_path = root / "calibration-index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            current = resolve_best_parameter_reference(
                index_path, detector="mask_rcnn_page_mask", golden_set_sha256="golden",
                model_variant="rcnn_hjdataset_current",
            )
            older = resolve_best_parameter_reference(
                index_path, detector="mask_rcnn_page_mask", golden_set_sha256="golden",
                model_variant="rcnn_hjdataset_older",
            )
            self.assertIsNotNone(current)
            self.assertIsNone(older)


if __name__ == "__main__":
    unittest.main()
