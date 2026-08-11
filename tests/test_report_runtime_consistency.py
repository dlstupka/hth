import json
import tempfile
import unittest
from pathlib import Path

from hth.runtime_store import coherent_execution_profile
from hth.write_regression_summary import _queue_rows


class ReportRuntimeConsistencyTests(unittest.TestCase):
    @staticmethod
    def runtime_index():
        observations = []
        for detector, seconds in (("a", 30.0), ("b", 20.0), ("c", 10.0)):
            observations.append({
                "detector_id": detector,
                "observed_at_utc": "2026-08-11T17:00:00Z",
                "golden_set_sha256": "golden",
                "mode": "smoke",
                "resolved_strategy": "exhaustive",
                "configured_threads": 2,
                "max_dimension": 1800,
                "detector_pipelines": 4,
                "detector_loading_strategy": "lpt",
                "wall_clock_seconds": seconds,
                "actual_parameter_sets": 10,
                "runner": {"runner_labels": ["github-hosted"]},
                "build": {"github_run_id": "123", "github_run_number": 999},
            })
        return {"observations": observations}

    def test_profile_is_recovered_from_one_coherent_runtime_build(self):
        profile = coherent_execution_profile(
            self.runtime_index(), ["a", "b", "c"], golden_set_sha256="golden"
        )
        self.assertEqual(profile["coverage"], 3)
        self.assertEqual(profile["threads"], 2)
        self.assertEqual(profile["pipeline_count"], 4)
        self.assertEqual(profile["loading_strategy"], "lpt")

    def test_regenerated_queue_uses_runtime_index_for_every_detector(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dirs = []
            for detector in ("a", "b", "c"):
                run = root / detector / "run"
                (run / "reports").mkdir(parents=True)
                (run / "manifest.json").write_text(json.dumps({"detector": detector}), encoding="utf-8")
                (run / "RUN-INFO.json").write_text(json.dumps({
                    "golden_set_sha256": "golden",
                    "threads": 99,
                    "detector_pipeline": {},
                }), encoding="utf-8")
                (run / "parameters.json").write_text("{}", encoding="utf-8")
                (run / "reports" / "summary.json").write_text(json.dumps({
                    "winner": {"summary": {"mean_iou": 1.0, "failure_count": 0}},
                    "parameter_set_count": 1,
                }), encoding="utf-8")
                run_dirs.append(run)

            runtime_path = root / "runtime-index.json"
            runtime_path.write_text(json.dumps(self.runtime_index()), encoding="utf-8")
            profile = coherent_execution_profile(
                self.runtime_index(), ["a", "b", "c"], golden_set_sha256="golden"
            )
            rows = _queue_rows(
                run_dirs,
                runtime_index_path=runtime_path,
                execution_profile=profile,
            )
            self.assertEqual([row["detector"] for row in rows], ["a", "b", "c"])
            self.assertEqual([row["estimate_seconds"] for row in rows], [30.0, 20.0, 10.0])
            self.assertTrue(all(str(row["estimate_source"]).startswith("runtime-index:") for row in rows))
            self.assertEqual([row["queue_position"] for row in rows], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
