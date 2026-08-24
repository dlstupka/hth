import json
import tempfile
import unittest
from pathlib import Path

from hth.calibration_store import publish_run, resolve, update_index


class CalibrationStoreTests(unittest.TestCase):
    def _run(self, root: Path, run_id: str, *, exhaustive: bool, mode: str, score: float = 0.9) -> Path:
        run = root / run_id
        (run / "reports").mkdir(parents=True)
        intelligence = {
            "schema_version": "1.1",
            "available": True,
            "detector": "grabcut",
            "calibration_identity": {
                "calibration_run_id": run_id,
                "created_at_utc": {"smoke": "2026-07-31T00:00:01Z", "full": "2026-07-31T00:00:02Z", "later-worse": "2026-07-31T00:00:03Z"}.get(run_id, "2026-07-31T00:00:04Z"),
                "source_document": {"id": "source-1"},
                "golden_set": {"collection_id": "GS-1", "sha256": "abc123"},
                "detector_configuration": {"sha256": "cfg123"},
            },
            "parameter_intelligence": {"classification_thresholds": {"critical": 0.25}},
            "search": {"strategy": "exhaustive", "parameter_sets": 10, "possible_parameter_sets": 10, "exhaustive_complete": exhaustive},
            "detector_selection_intelligence": {"recommended_parameter_set_id": f"p-{run_id}", "best_avg_iou": score, "minimum_iou": score - 0.1, "stddev_iou": 0.05, "failure_count": 0},
        }
        (run / "reports" / "calibration-intelligence.json").write_text(json.dumps(intelligence), encoding="utf-8")
        for name in ("manifest.json", "parameters.json"):
            (run / name).write_text("{}", encoding="utf-8")
        (run / "RUN-INFO.json").write_text(json.dumps({"elapsed_seconds": 3723}), encoding="utf-8")
        (run / "reports" / "summary.json").write_text("{}", encoding="utf-8")
        (run / "reports" / "winner-pages.json").write_text("{}", encoding="utf-8")
        return run

    def test_later_worse_authoritative_full_is_retained_but_cannot_usurp_incumbent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results"
            incumbent = self._run(root, "full", exhaustive=True, mode="full", score=0.9137)
            challenger = self._run(root, "later-worse", exhaustive=True, mode="full", score=0.9071)
            build_old = {"github_run_number": "290", "github_run_id": "290"}
            build_new = {"github_run_number": "429", "github_run_id": "429"}
            old_entry = publish_run(incumbent, results, mode="full", source_fallback="repo", build=build_old)
            new_entry = publish_run(challenger, results, mode="full", source_fallback="repo", build=build_new)
            index = update_index(results, [old_entry, new_entry])

            self.assertEqual(len(index["entries"]), 2)
            preferred = next(iter(index["preferred"].values()))
            self.assertEqual(preferred["calibration_id"], "full")
            self.assertEqual(preferred["build"]["github_run_number"], "290")
            self.assertEqual(preferred["selection"]["best_avg_iou"], 0.9137)

            selected = resolve(
                results / "indexes" / "calibration-index.json",
                detector="grabcut",
                golden_set_sha256="abc123",
                detector_config_sha256="cfg123",
            )
            self.assertIsNotNone(selected)
            self.assertIn("/full/", selected.as_posix())

    def test_publishes_records_and_prefers_authoritative(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "results"
            smoke = self._run(root, "smoke", exhaustive=False, mode="smoke")
            full = self._run(root, "full", exhaustive=True, mode="full", score=0.95)
            build = {
                "workflow": "Regress detectors against Golden Set",
                "github_run_id": "1",
                "github_run_number": "193",
                "github_run_attempt": "1",
                "run_url": "https://github.com/dlstupka/hth/actions/runs/1",
            }
            entries = [
                publish_run(smoke, results, mode="smoke", source_fallback="repo", build=build),
                publish_run(full, results, mode="full", source_fallback="repo", build=build),
            ]
            index = update_index(results, entries)
            self.assertEqual(len(index["entries"]), 2)
            preferred = next(iter(index["preferred"].values()))
            self.assertEqual(preferred["calibration_status"], "authoritative")
            selected = resolve(results / "indexes" / "calibration-index.json", detector="grabcut", golden_set_sha256="abc123", detector_config_sha256="cfg123")
            self.assertIsNotNone(selected)
            self.assertIn("full", selected.as_posix())
            stored = json.loads(selected.read_text(encoding="utf-8"))
            self.assertEqual(stored["calibration_status"], "authoritative")
            self.assertEqual(stored["calibration_identity"]["build"]["github_run_id"], "1")
            self.assertEqual(stored["calibration_identity"]["build"]["github_run_number"], "193")
            self.assertEqual(stored["calibration_identity"]["build"]["workflow"], "Regress detectors against Golden Set")
            self.assertEqual(stored["calibration_identity"]["build"]["run_time_seconds"], 3723)
            self.assertEqual(preferred["build"]["run_url"], "https://github.com/dlstupka/hth/actions/runs/1")
            self.assertEqual(preferred["build"]["run_time_seconds"], 3723)


if __name__ == "__main__":
    unittest.main()
