import json
import tempfile
import unittest
from pathlib import Path

from hth.persistence import INDEX_FILENAMES, canonical_index_path, load_index, write_index
from hth.persistence_rebuild import rebuild_all
from hth.shape_prediction import record_prediction_observations

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


class PersistenceArchitectureTests(unittest.TestCase):
    def test_all_derived_indexes_have_one_canonical_registry(self):
        self.assertEqual(INDEX_FILENAMES, frozenset({
            "calibration-index.json", "multidetector-index.json", "optimizer-index.json",
            "optimizer-predictions.json",
            "orli-evidence-index.json", "parallelism-index.json",
            "parameter-provenance-index.json", "runtime-index.json",
        }))
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in INDEX_FILENAMES:
                self.assertEqual(canonical_index_path(root, name), root / "indexes" / name)

    def test_index_writers_use_canonical_persistence_boundary(self):
        for relative in (
            "hth/calibration_store.py", "hth/runtime_store.py", "hth/parallelism_store.py",
            "hth/multidetector_store.py", "hth/optimizer_store.py",
            "hth/regression/learned_evidence.py",
        ):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("def _write_json(path: Path", text, relative)
        self.assertIn("atomic_write_json", (ROOT / "hth/regression/learned_evidence.py").read_text(encoding="utf-8"))

    def test_delete_indexes_rebuilds_calibration_from_durable_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            record = root / "source-documents/source/golden-sets/hth-0001/gold/calibrations/det/run-1"
            record.mkdir(parents=True)
            intelligence = {
                "schema_version": "1.1",
                "available": True,
                "detector": "det",
                "calibration_status": "authoritative",
                "calibration_identity": {
                    "calibration_run_id": "run-1",
                    "created_at_utc": "2026-08-24T00:00:00Z",
                    "source_document": {"id": "source"},
                    "golden_set": {"collection_id": "HTH-0001", "sha256": "gold"},
                    "detector_configuration": {"sha256": "cfg"},
                    "build": {"github_run_number": "1"},
                },
                "search": {"strategy": "exhaustive", "parameter_sets": 2, "possible_parameter_sets": 2, "exhaustive_complete": True},
                "detector_selection_intelligence": {"recommended_parameter_set_id": "abc", "best_avg_iou": 0.9, "failure_count": 0},
            }
            (record / "calibration-intelligence.json").write_text(json.dumps(intelligence), encoding="utf-8")
            write_index(root, "calibration-index.json", {"entries": [], "preferred": {}})
            canonical_index_path(root, "calibration-index.json").unlink()
            rebuilt = rebuild_all(root)
            index = load_index(root, "calibration-index.json")
            self.assertEqual(rebuilt["calibration_entries"], 1)
            self.assertEqual(index["entries"][0]["calibration_id"], "run-1")
            self.assertEqual(index["entries"][0]["calibration_status"], "authoritative")

    def test_workflows_share_layout_cleanup_and_hardened_artifact_delivery(self):
        for workflow in WORKFLOWS.glob("*.yml"):
            text = workflow.read_text(encoding="utf-8")
            self.assertNotIn("uses: actions/upload-artifact@v6", text, workflow.name)
            self.assertNotIn("git -C results-repo rm -f --ignore-unmatch -- calibration-index.json", text, workflow.name)
        action = (ROOT / ".github/actions/hardened-upload-artifact/action.yml").read_text(encoding="utf-8")
        self.assertEqual(action.count("uses: actions/upload-artifact@v6"), 3)
        self.assertIn("durable-persistence-confirmed", action)
        self.assertIn("::warning::Artifact service failed after three attempts", action)
        regression = (WORKFLOWS / "regress-detector.yml").read_text(encoding="utf-8")
        self.assertIn("durable-persistence-confirmed: ${{ steps.persistence.outcome == 'success' }}", regression)

    def test_prediction_observation_writer_migrates_through_canonical_index_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            legacy = root / "optimizer-predictions.json"
            legacy.write_text(json.dumps({"schema_version": "1.0", "predictions": [{"prediction_id": "old"}]}), encoding="utf-8")
            canonical = root / "indexes" / "optimizer-predictions.json"
            observation = {
                "observation_id": "obs-1",
                "observed_at_utc": "2026-08-30T00:00:00Z",
                "run_id": "run-1",
                "detector_id": "example",
                "execution_shape_source": "predicted-low-linear-vcpu",
                "active_pipelines": 4,
                "threads_per_pipeline": 16,
                "allocated_threads": 64,
                "mode": "full",
                "strategy": "critical",
                "parameter_set_limit": 10,
                "search_scope": {"mode": "full", "strategy": "critical", "limit": 10},
                "detector_config_sha256": "cfg",
                "golden_set_sha256": "gold",
                "max_dimension": 1800,
                "runner": {"runner_name": "rh8-s32", "runner_label": "32t", "logical_cpu_count": 32},
                "build": {"github_run_id": "123", "github_run_attempt": "1"},
            }
            payload = record_prediction_observations(canonical, [observation])
            self.assertTrue(canonical.is_file())
            self.assertEqual(len(payload["predictions"]), 2)
            self.assertEqual(payload["predictions"][1]["search_scope"]["limit"], 10)
            self.assertEqual(load_index(root, "optimizer-predictions.json")["predictions"], payload["predictions"])

    def test_shape_prediction_has_no_ad_hoc_canonical_prediction_write(self):
        text = (ROOT / "hth/shape_prediction.py").read_text(encoding="utf-8")
        self.assertIn('write_index(index_results_root(path), "optimizer-predictions.json", payload)', text)
        self.assertNotIn('_write_json(predictions_index, payload)', text)

    def test_source_acquisition_uses_authenticated_github_token_fallback(self):
        core = (WORKFLOWS / "_core-hth.yml").read_text(encoding="utf-8")
        self.assertIn("secrets.HTH_SOURCE_TOKEN || github.token", core)
        source = (ROOT / "hth/source_release.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("HTH_SOURCE_TOKEN") or os.environ.get("GITHUB_TOKEN", "")', source)
        self.assertIn("_urlopen_with_retry", source)

    def test_execution_optimizer_uses_bounded_canonical_benchmark_workload(self):
        workflow = (WORKFLOWS / "execution-optimizer.yml").read_text(encoding="utf-8")
        self.assertIn("benchmark_parameter_sets:", workflow)
        self.assertIn('default: "256"', workflow)
        self.assertIn('HTH_BOUNDED_WORKLOAD: "1"', workflow)
        self.assertIn("HTH_OPTIMIZER_BENCHMARK_PARAMETER_SETS", workflow)
        runner = (ROOT / "tools/run-detector-regressions.sh").read_text(encoding="utf-8")
        self.assertIn('"${HTH_BOUNDED_WORKLOAD:-0}" != "1"', runner)


if __name__ == "__main__":
    unittest.main()
