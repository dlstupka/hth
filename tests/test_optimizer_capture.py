from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hth.optimizer_capture import capture_observation, replay_observations


class OptimizerCaptureTests(unittest.TestCase):
    @staticmethod
    def _run_dir(root: Path) -> Path:
        run = root / "run-1"
        (run / "reports").mkdir(parents=True)
        (run / "RUN-INFO.json").write_text(json.dumps({
            "run_id": "run-1",
            "detector": "adaptive_radial_edge",
            "threads": 8,
            "shard_count": 8,
            "detector_pipeline": {"pipeline_count": 8},
            "strategy": "exhaustive",
            "possible_parameter_sets": 100,
            "actual_parameter_sets": 100,
            "golden_set_pages": 5,
            "max_dimension": 1800,
            "estimated_serial_runtime_seconds": 800.0,
            "wall_elapsed_seconds": 200.0,
        }), encoding="utf-8")
        (run / "reports" / "summary.json").write_text(json.dumps({
            "runner": {
                "runner_name": "rh8-a197",
                "logical_cpu_count": 96,
                "cpu_model": "AMD EPYC",
            },
            "parameter_space": {"possible_parameter_sets": 100, "actual_parameter_sets": 100},
            "golden_set": {"pages": 5},
        }), encoding="utf-8")
        return run

    def test_optimizer_capture_uses_outer_shape_wall_clock_and_logs_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            log = root / "observations.jsonl"
            observation = capture_observation(
                results_root=results,
                run_dir=self._run_dir(root),
                wall_clock_seconds=100.0,
                runner_label="e7k",
                github_run_id="1234",
                shape_sequence=2,
                observation_log=log,
            )
            self.assertEqual(observation["wall_clock_seconds"], 100.0)
            self.assertEqual(observation["effective_acceleration"], 8.0)
            self.assertEqual(observation["execution_shape"], "8p/8s/8t")
            self.assertTrue(observation["observation_id"].startswith("optimizer:1234:2:"))
            self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 1)

    def test_optimizer_observations_can_be_replayed_on_fresh_results_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            first.mkdir()
            log = root / "observations.jsonl"
            capture_observation(
                results_root=first,
                run_dir=self._run_dir(root),
                wall_clock_seconds=100.0,
                runner_label="e7k",
                github_run_id="1234",
                shape_sequence=1,
                observation_log=log,
            )
            fresh = root / "fresh"
            fresh.mkdir()
            self.assertEqual(replay_observations(results_root=fresh, observation_log=log), 1)
            payload = json.loads((fresh / "parallelism-index.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["observations"][0]["wall_clock_seconds"], 100.0)


if __name__ == "__main__":
    unittest.main()
