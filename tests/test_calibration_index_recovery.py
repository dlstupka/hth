import json
import tempfile
import unittest
from pathlib import Path

from hth.calibration_store import load_index_with_persisted_backfill
from hth.report_generator import calibration_run_dirs
from hth.resolve_document_detector import resolve_rank_one
from hth.regression.parameter_space import parameter_set_id
from hth.regression.parameter_provenance import parameter_identity_sha256


class CalibrationIndexRecoveryTests(unittest.TestCase):
    def test_incomplete_smoke_index_recovers_authoritative_full_history(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            results = root / "results"
            index_dir = results / "indexes"
            index_dir.mkdir(parents=True)

            detector = "amsre_doc_ufcn_fusion"
            params = {"alpha": 1}
            legacy = parameter_set_id(params)
            full_sha = parameter_identity_sha256(detector, params, schema_version="1")
            record = (
                results / "source-documents/source/golden-sets/hth-0001/gold123/"
                "calibrations/amsre_doc_ufcn_fusion/run-full"
            )
            record.mkdir(parents=True)
            intelligence = {
                "schema_version": "1.1",
                "available": True,
                "detector": detector,
                "calibration_status": "authoritative",
                "calibration_identity": {
                    "calibration_run_id": "run-full",
                    "created_at_utc": "2026-08-22T00:00:00Z",
                    "source_document": {"id": "source"},
                    "golden_set": {"collection_id": "HTH-0001", "sha256": "gold123"},
                    "detector_configuration": {"sha256": "cfg"},
                    "build": {"github_run_number": "782", "run_url": "https://example.invalid/782"},
                },
                "search": {"strategy": "exhaustive", "parameter_sets": 29, "exhaustive_complete": True},
                "detector_selection_intelligence": {
                    "recommended_parameter_set_id": legacy,
                    "best_avg_iou": 0.9897,
                    "minimum_iou": 0.9814,
                    "stddev_iou": 0.0063,
                    "failure_count": 0,
                    "calibration_evidence": {"rating": "High"},
                },
                "persistence": {"published_at_utc": "2026-08-22T01:00:00Z"},
            }
            (record / "calibration-intelligence.json").write_text(json.dumps(intelligence), encoding="utf-8")
            (record / "parameter-provenance.json").write_text(json.dumps({
                "identity": {"detector": detector, "parameter_schema_version": "1"},
                "explicit_parameter_sets": {
                    full_sha: {"sha256": full_sha, "legacy_parameter_set_id": legacy, "parameters": params}
                },
            }), encoding="utf-8")
            (record / "summary.json").write_text(json.dumps({"winner": {"parameter_set_id": legacy, "parameters": params}}), encoding="utf-8")

            # Simulate the broken migration state: canonical index has only a smoke row.
            smoke = {
                "calibration_id": "run-smoke",
                "calibration_status": "provisional",
                "record_path": "missing/smoke",
                "golden_set_id": "hth-0001",
                "golden_set_sha256": "gold123",
                "detector_id": detector,
                "created_at_utc": "2026-08-24T00:00:00Z",
                "search": {"strategy": "smoke", "parameter_sets": 10},
                "selection": {"best_avg_iou": 0.9897, "failure_count": 0, "calibration_evidence": "Medium"},
            }
            index_path = index_dir / "calibration-index.json"
            index_path.write_text(json.dumps({"entries": [smoke]}), encoding="utf-8")

            loaded = load_index_with_persisted_backfill(index_path)
            self.assertEqual(len(loaded["entries"]), 2)
            resolved = resolve_rank_one(index_path, golden_set_id="HTH-0001")
            self.assertEqual(resolved["calibration_id"], "run-full")
            self.assertEqual(resolved["build_number"], "782")
            self.assertEqual(resolved["parameters"], params)

            run_dirs = calibration_run_dirs(results)
            self.assertEqual(run_dirs, [record])


if __name__ == "__main__":
    unittest.main()
