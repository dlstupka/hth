import json
import tempfile
import unittest
from pathlib import Path

from hth.regression.parameter_space import parameter_set_id
from hth.regression.parameter_provenance import parameter_identity_sha256
from hth.resolve_document_detector import resolve_rank_one, render_summary


class PreferredDocumentDetectorTests(unittest.TestCase):
    def _entry(self, detector, avg, minimum, stddev, evidence, record_path, parameter_id):
        return {
            "calibration_id": f"cal-{detector}",
            "calibration_status": "authoritative",
            "record_path": record_path,
            "parameter_provenance_path": f"{record_path}/parameter-provenance.json",
            "golden_set_id": "hth-0001",
            "golden_set_sha256": "gold123",
            "detector_id": detector,
            "created_at_utc": "2026-08-22T00:00:00Z",
            "build": {"github_run_number": "732", "run_url": "https://example.invalid/732"},
            "search": {"strategy": "exhaustive", "parameter_sets": 29, "exhaustive_complete": True},
            "selection": {
                "recommended_parameter_set_id": parameter_id,
                "best_avg_iou": avg,
                "minimum_iou": minimum,
                "stddev_iou": stddev,
                "failure_count": 0,
                "calibration_evidence": evidence,
            },
        }

    def test_resolves_highest_scoring_approved_calibration_and_exact_parameters(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            params = {
                "amsre_rescue_score_ceiling": 0.95,
                "doc_ufcn_minimum_confidence": 0.9,
                "maximum_amsre_refined_support_fraction": 0.65,
                "minimum_corner_disagreement_fraction": 0.0075,
            }
            detector = "amsre_doc_ufcn_fusion"
            legacy = parameter_set_id(params)
            full = parameter_identity_sha256(detector, params, schema_version="1")
            rec = Path("source-documents/source/golden-sets/hth-0001/gold123/calibrations") / detector / "cal"
            rec_dir = root / rec
            rec_dir.mkdir(parents=True)
            provenance = {
                "identity": {"detector": detector, "parameter_schema_version": "1"},
                "explicit_parameter_sets": {
                    full: {"sha256": full, "legacy_parameter_set_id": legacy, "parameters": params}
                },
            }
            (rec_dir / "parameter-provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
            index = {
                "entries": [
                    self._entry("other", 0.9800, 0.9700, 0.01, "High", "missing", "missing"),
                    self._entry(detector, 0.9897, 0.9814, 0.0063, "High", rec.as_posix(), legacy),
                    self._entry("unapproved", 0.9999, 0.9990, 0.001, "Medium", "missing", "missing"),
                ]
            }
            index_path = root / "calibration-index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")

            resolved = resolve_rank_one(index_path, golden_set_id="HTH-0001")
            self.assertEqual(resolved["detector"], detector)
            self.assertEqual(resolved["parameters"], params)
            self.assertEqual(resolved["parameter_set_id"], legacy)
            self.assertEqual(resolved["approval_level"], "Approved")
            summary = render_summary(resolved, display_name="Fusion Gen3")
            self.assertIn("0.9897", summary)
            self.assertIn("0.9814", summary)
            self.assertIn("maximum_amsre_refined_support_fraction", summary)

    def test_approved_accepts_persisted_authoritative_index_semantics(self):
        from hth.resolve_document_detector import _approved

        # Best Known treats authoritative exhaustive-family records as complete;
        # historic index rows need not redundantly carry exhaustive_complete.
        entry = self._entry("detector", 0.9897, 0.9814, 0.0063, {"rating": "High"}, "record", "abc")
        entry["search"].pop("exhaustive_complete")
        entry["search"]["strategy"] = "exhaustive-with-zombies"
        self.assertTrue(_approved(entry))

    def test_workflows_use_preferred_without_detector_research_dropdown(self):
        root = Path(__file__).resolve().parents[1]
        preprocess = (root / ".github/workflows/preprocess.yml").read_text(encoding="utf-8")
        test = (root / ".github/workflows/preprocess-test.yml").read_text(encoding="utf-8")
        core = (root / ".github/workflows/_core-hth.yml").read_text(encoding="utf-8")
        self.assertNotIn("Run approved detector over every page for corpus review", preprocess)
        self.assertIn("document_detector: preferred", preprocess)
        self.assertIn("document_detector: preferred", test)
        self.assertIn("Resolve Rank #1 approved document detector", core)
        self.assertIn('if [[ ! -f "$calibration_index" && -f results-repo/calibration-index.json ]]', core)
        self.assertIn('calibration_index="results-repo/calibration-index.json"', core)
        self.assertIn('--index "../$calibration_index"', core)
        self.assertIn("--selection \"$RUNNER_TEMP/preferred-document-detector.json\"", core)


if __name__ == "__main__":
    unittest.main()
