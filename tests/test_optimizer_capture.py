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
                "runner_name": "rh8-al97",
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


    def test_optimizer_capture_keeps_startup_overhead_separate_but_included_in_raw_wall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            observation = capture_observation(
                results_root=results,
                run_dir=self._run_dir(root),
                wall_clock_seconds=240.0,
                runner_label="e7k",
                github_run_id="5678",
                shape_sequence=1,
                startup_overhead_seconds=180.0,
            )
            self.assertEqual(observation["wall_clock_seconds"], 240.0)
            self.assertEqual(observation["startup_overhead_seconds"], 180.0)
            self.assertTrue(observation["startup_overhead_included_in_wall_clock"])
            self.assertAlmostEqual(observation["parameter_sets_per_second"], 100 / 240.0)

    def test_optimizer_shard_log_defers_shared_index_write_until_replay(self) -> None:
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
            self.assertFalse((results / "indexes" / "parallelism-index.json").exists())
            self.assertEqual(len(shard_log.read_text(encoding="utf-8").splitlines()), 1)
            fresh = root / "fresh"
            fresh.mkdir()
            self.assertEqual(replay_shard_observations(results_root=fresh, shard_log=shard_log), 1)
            replayed = json.loads((fresh / "indexes" / "parallelism-index.json").read_text(encoding="utf-8"))
            self.assertEqual(replayed["shard_observations"][0]["optimizer_run_id"], "1234")

    def test_optimizer_shard_without_log_still_updates_shared_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            capture_shard_observation(
                results_root=results,
                run_dir=self._run_dir(root, actual_sets=13, shard_count=8, threads=8),
                runner_label="e7k",
                github_run_id="1234",
                shape_sequence=3,
                pipeline_number=2,
                shard_index=4,
                shard_count=8,
                threads=8,
                wall_clock_seconds=20.0,
            )
            payload = json.loads((results / "indexes" / "parallelism-index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload["shard_observations"]), 1)

    def test_optimizer_shard_throughput_uses_locally_evaluated_sets_when_baseline_is_shared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            run = self._run_dir(root, actual_sets=5, shard_count=4, threads=8)
            info_path = run / "RUN-INFO.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["locally_evaluated_parameter_sets"] = 4
            info["baseline_execution"] = "shared-cache"
            info_path.write_text(json.dumps(info), encoding="utf-8")
            record = capture_shard_observation(
                results_root=results,
                run_dir=run,
                runner_label="e7k",
                github_run_id="1234",
                shape_sequence=3,
                pipeline_number=2,
                shard_index=1,
                shard_count=4,
                threads=8,
                wall_clock_seconds=20.0,
            )
            self.assertEqual(record["actual_parameter_sets"], 5)
            self.assertEqual(record["locally_evaluated_parameter_sets"], 4)
            self.assertEqual(record["baseline_execution"], "shared-cache")
            self.assertAlmostEqual(record["parameter_sets_per_second"], 0.2)

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
            payload = json.loads((fresh / "indexes" / "parallelism-index.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["observations"][0]["wall_clock_seconds"], 100.0)

    def _write_optimizer_rates(self, log: Path, rates: list[tuple[int, float]]) -> None:
        with log.open("w", encoding="utf-8") as stream:
            for sequence, (pipelines, rate) in enumerate(rates, 1):
                stream.write(json.dumps({
                    "optimizer_shape_sequence": sequence,
                    "active_pipelines": pipelines,
                    "execution_shape": f"{pipelines}p/{pipelines}s/1t",
                    "parameter_sets_per_second": rate,
                }) + "\n")

    def test_early_stop_requires_more_than_two_percent_degradation_on_both_sides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "observations.jsonl"
            self._write_optimizer_rates(log, [
                (1, 95.0), (2, 96.0), (3, 97.0), (4, 100.0),
                (5, 97.9), (6, 97.0), (7, 96.0),
            ])
            assessment = assess_early_stop(log, threshold_pct=2.0, pipeline_min=1, pipeline_max=7)
            self.assertTrue(assessment["should_stop"])
            self.assertTrue(assessment["left_boundary_confirmed"])
            self.assertTrue(assessment["right_boundary_confirmed"])
            self.assertEqual(assessment["required_consecutive_shapes"], 3)
            self.assertEqual(assessment["left_consecutive_below_peak"], 3)
            self.assertEqual(assessment["right_consecutive_below_peak"], 3)
            self.assertEqual(assessment["peak_region_pipeline_min"], 4)
            self.assertEqual(assessment["peak_region_pipeline_max"], 4)

    def test_early_stop_does_not_accept_exactly_two_percent_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "observations.jsonl"
            self._write_optimizer_rates(log, [(1, 100.0), (2, 98.0), (3, 97.0), (4, 96.0)])
            assessment = assess_early_stop(log, threshold_pct=2.0, pipeline_min=1, pipeline_max=8)
            self.assertFalse(assessment["should_stop"])
            self.assertEqual(assessment["peak_region_pipeline_max"], 1)
            self.assertEqual(assessment["right_consecutive_below_peak"], 2)
            self.assertFalse(assessment["right_boundary_confirmed"])

    def test_early_stop_near_peak_reading_breaks_consecutive_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "observations.jsonl"
            self._write_optimizer_rates(log, [
                (1, 100.0), (2, 97.0), (3, 99.7), (4, 97.8),
                (5, 97.0), (6, 96.0),
            ])
            assessment = assess_early_stop(log, threshold_pct=2.0, pipeline_min=1, pipeline_max=8)
            self.assertTrue(assessment["should_stop"])
            self.assertEqual(assessment["right_consecutive_below_peak"], 3)
            self.assertFalse(assessment["left_boundary_required"])
            self.assertTrue(assessment["right_boundary_confirmed"])

    def test_early_stop_allows_one_sided_boundary_peak(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "observations.jsonl"
            self._write_optimizer_rates(log, [(1, 100.0), (2, 75.0)])
            assessment = assess_early_stop(log, threshold_pct=2.0, pipeline_min=1, pipeline_max=8)
            self.assertFalse(assessment["should_stop"])
            self.assertFalse(assessment["left_boundary_required"])
            self.assertTrue(assessment["right_boundary_required"])
            self.assertEqual(assessment["right_consecutive_below_peak"], 1)

            self._write_optimizer_rates(log, [(1, 100.0), (2, 75.0), (3, 70.0), (4, 65.0)])
            assessment = assess_early_stop(log, threshold_pct=2.0, pipeline_min=1, pipeline_max=8)
            self.assertTrue(assessment["should_stop"])
            self.assertTrue(assessment["right_boundary_confirmed"])
            self.assertEqual(assessment["right_consecutive_below_peak"], 3)

    def test_early_stop_does_not_invent_peak_when_full_range_is_flat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "observations.jsonl"
            self._write_optimizer_rates(log, [(1, 100.0), (2, 99.8), (3, 99.5), (4, 99.0)])
            assessment = assess_early_stop(log, threshold_pct=2.0, pipeline_min=1, pipeline_max=4)
            self.assertFalse(assessment["should_stop"])
            self.assertFalse(assessment["left_boundary_required"])
            self.assertTrue(assessment["right_boundary_required"])
            self.assertFalse(assessment["right_boundary_confirmed"])



if __name__ == "__main__":
    unittest.main()
