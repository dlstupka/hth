from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hth.optimizer_capture import (
    assess_early_stop,
    capture_observation,
    capture_shard_observation,
    replay_observations,
    replay_shard_observations,
)


class OptimizerCaptureTests(unittest.TestCase):
    @staticmethod
    def _run_dir(root: Path, *, actual_sets: int = 100, shard_count: int = 8, threads: int = 8) -> Path:
        run = root / f"run-{actual_sets}-{shard_count}-{threads}"
        (run / "reports").mkdir(parents=True)
        (run / "RUN-INFO.json").write_text(json.dumps({
            "run_id": run.name,
            "detector": "adaptive_radial_edge",
            "threads": threads,
            "shard_count": shard_count,
            "detector_pipeline": {"pipeline_count": shard_count},
            "strategy": "exhaustive",
            "possible_parameter_sets": 100,
            "actual_parameter_sets": actual_sets,
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
            "parameter_space": {"possible_parameter_sets": 100, "actual_parameter_sets": actual_sets},
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
            self.assertEqual(observation["optimizer_run_id"], "1234")
            self.assertEqual(observation["source"], "execution-optimizer")
            self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 1)

    def test_optimizer_shard_is_persisted_immediately_and_replayable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            shard_log = root / "shards.jsonl"
            run = self._run_dir(root, actual_sets=13, shard_count=8, threads=8)
            record = capture_shard_observation(
                results_root=results,
                run_dir=run,
                runner_label="e7k",
                github_run_id="1234",
                shape_sequence=3,
                pipeline_number=2,
                shard_index=4,
                shard_count=8,
                threads=8,
                wall_clock_seconds=20.0,
                shard_log=shard_log,
            )
            self.assertEqual(record["shard_number"], 5)
            self.assertAlmostEqual(record["parameter_sets_per_second"], 0.65)
            payload = json.loads((results / "parallelism-index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["shard_observations"]), 1)
            fresh = root / "fresh"
            fresh.mkdir()
            self.assertEqual(replay_shard_observations(results_root=fresh, shard_log=shard_log), 1)
            replayed = json.loads((fresh / "parallelism-index.json").read_text(encoding="utf-8"))
            self.assertEqual(replayed["shard_observations"][0]["optimizer_run_id"], "1234")

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

    def test_early_stop_requires_three_consecutive_sub_one_percent_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "observations.jsonl"
            rates = [10.0, 20.0, 20.1, 19.9, 19.8]
            with log.open("w", encoding="utf-8") as stream:
                for sequence, rate in enumerate(rates, 1):
                    stream.write(json.dumps({
                        "optimizer_shape_sequence": sequence,
                        "execution_shape": f"{sequence}p/{sequence}s/1t",
                        "parameter_sets_per_second": rate,
                    }) + "\n")
            assessment = assess_early_stop(log, threshold_pct=1.0, consecutive=3)
            self.assertTrue(assessment["should_stop"])
            self.assertEqual(assessment["stop_reason"], "throughput_plateau")
            self.assertEqual(assessment["non_improving_streak"], 3)
            self.assertEqual(assessment["best_parameter_sets_per_second"], 20.1)


if __name__ == "__main__":
    unittest.main()
