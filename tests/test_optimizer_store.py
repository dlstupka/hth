from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hth.optimizer_store import build_optimizer_index, update_optimizer_artifacts


def _row(identifier: str, runner: str, pipelines: int, threads: int, wall: float, *, optimizer_run_id: str = "100") -> dict:
    return {
        "observation_id": identifier,
        "source": "execution-optimizer",
        "optimizer_run_id": optimizer_run_id,
        "optimizer_shape_sequence": pipelines,
        "detector_id": "adaptive_radial_edge",
        "mode": "full",
        "strategy": "exhaustive",
        "possible_parameter_sets": 6562,
        "actual_parameter_sets": 6562,
        "execution_shape": f"{pipelines}p/{pipelines}s/{threads}t",
        "shards": pipelines,
        "active_pipelines": pipelines,
        "threads_per_pipeline": threads,
        "allocated_threads": pipelines * threads,
        "wall_clock_seconds": wall,
        "parameter_sets_per_second": 6562 / wall,
        "effective_acceleration": 12.0,
        "parallel_efficiency": 12.0 / (pipelines * threads),
        "runner_metrics": {"sample_count": 2, "avg_load1": 10.0, "peak_load1": 20.0, "avg_cpu_pct": 75.0, "peak_ram_used_bytes": 8 * 1024**3},
        "runner": {
            "runner_label": runner,
            "runner_name": "rh8-a197" if runner == "e7k" else runner,
            "runner_labels": ["self-hosted", "linux", runner],
            "cpu_model": f"CPU {runner}",
            "logical_cpu_count": 96 if runner == "e7k" else 32,
        },
    }


class OptimizerStoreTests(unittest.TestCase):
    def test_optimizer_index_can_filter_to_current_execution_only(self) -> None:
        parallelism = {"schema_version": "2.2", "observations": [
            _row("a", "e7k", 1, 64, 2600, optimizer_run_id="100"),
            _row("b", "e7k", 8, 8, 420, optimizer_run_id="100"),
            _row("old", "e7k", 64, 1, 90, optimizer_run_id="99"),
        ]}
        index = build_optimizer_index(parallelism, "adaptive_radial_edge", "100")
        self.assertEqual(index["observation_count"], 2)
        self.assertTrue(all(shape["pipelines"] != 64 for runner in index["runners"] for shape in runner["shapes"]))

    def test_optimizer_index_keeps_detector_specific_historical_preferences(self) -> None:
        parallelism = {"schema_version": "2.2", "observations": [
            _row("a", "e7k", 1, 64, 2600),
            _row("b", "e7k", 8, 8, 420),
        ]}
        index = build_optimizer_index(parallelism, "adaptive_radial_edge")
        e7k = index["runners"][0]
        self.assertEqual(e7k["best_shape"]["pipelines"], 8)

    def test_optimizer_artifacts_use_current_run_table_and_pipeline_sets_per_second_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parallelism-index.json").write_text(json.dumps({
                "schema_version": "2.2",
                "observations": [
                    _row("a", "e7k", 1, 64, 2600, optimizer_run_id="100"),
                    _row("b", "e7k", 8, 8, 420, optimizer_run_id="100"),
                    _row("old", "e9k", 4, 8, 500, optimizer_run_id="99"),
                ],
                "shard_observations": [{"optimizer_run_id": "100", "detector_id": "adaptive_radial_edge", "observation_id": "s1"}],
            }), encoding="utf-8")
            metadata = root / "run.json"
            metadata.write_text(json.dumps({"stop_reason": "throughput_plateau", "early_stop": {"stop_reason": "throughput_plateau", "required_consecutive_shapes": 3, "threshold_pct": 1.0}}), encoding="utf-8")
            paths = update_optimizer_artifacts(root, "adaptive_radial_edge", optimizer_run_id="100", run_metadata_path=metadata)
            payload = json.loads(paths["index"].read_text(encoding="utf-8"))
            self.assertIn("adaptive_radial_edge", payload["detectors"])
            self.assertEqual(payload["runs"]["100"]["shard_observation_count"], 1)
            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("this table contains only shapes completed in this execution", markdown)
            self.assertIn("e7k", markdown)
            self.assertNotIn("e9k", markdown)
            svg = paths["heatmap"].read_text(encoding="utf-8")
            self.assertTrue(svg.startswith("<svg"))
            self.assertIn("detector pipelines (log₂ scale)", svg)
            self.assertIn("parameter sets / second", svg)
            self.assertIn("8t", svg)


if __name__ == "__main__":
    unittest.main()
