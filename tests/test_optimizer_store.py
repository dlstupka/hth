from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hth.optimizer_store import build_optimizer_index, render_heatmap_svg, render_markdown, select_preferred_shape, update_optimizer_artifacts
from hth.parallelism_store import update_parallelism_index, update_parallelism_shards


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

    def test_canonical_preferred_shape_breaks_equal_displayed_throughput_ties_by_resources(self) -> None:
        shapes = [
            {"execution_shape": "6p/6s/64t", "pipelines": 6, "threads_per_pipeline": 64, "allocated_threads": 384, "fastest_wall_clock_seconds": 9.0, "parameter_sets_per_second": 27.004, "optimizer_shape_sequence": 2},
            {"execution_shape": "5p/5s/76t", "pipelines": 5, "threads_per_pipeline": 76, "allocated_threads": 380, "fastest_wall_clock_seconds": 9.0, "parameter_sets_per_second": 27.003, "optimizer_shape_sequence": 1},
        ]
        best = select_preferred_shape(shapes)
        self.assertIsNotNone(best)
        self.assertEqual(best["pipelines"], 5)

    def test_canonical_preferred_shape_never_trades_visible_throughput_for_resources(self) -> None:
        shapes = [
            {"execution_shape": "6p/6s/64t", "pipelines": 6, "threads_per_pipeline": 64, "allocated_threads": 384, "fastest_wall_clock_seconds": 9.0, "parameter_sets_per_second": 27.01},
            {"execution_shape": "5p/5s/76t", "pipelines": 5, "threads_per_pipeline": 76, "allocated_threads": 380, "fastest_wall_clock_seconds": 9.0, "parameter_sets_per_second": 27.00},
        ]
        best = select_preferred_shape(shapes)
        self.assertEqual(best["pipelines"], 6)

    def test_profile_plot_uses_consistent_thread_label_placement_and_peak_headroom(self) -> None:
        rows = [
            _row("p5", "e7k", 5, 76, 243.04),
            _row("p6", "e7k", 6, 64, 243.04),
            _row("p7", "e7k", 7, 54, 270.0),
            _row("p8", "e7k", 8, 48, 270.0),
            _row("p9", "e7k", 9, 42, 270.0),
        ]
        index = build_optimizer_index({"observations": rows}, "adaptive_radial_edge")
        svg = render_heatmap_svg(index)
        self.assertIn('text-anchor="start" font-size="10">76t</text>', svg)
        self.assertIn('text-anchor="start" font-size="10">64t</text>', svg)
        self.assertIn('76t</text>', svg)
        self.assertIn('64t</text>', svg)
        self.assertIn('42t</text>', svg)
        self.assertIn('y1="112"', svg)

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

    def test_historical_optimizer_profile_coalesces_compatible_shapes_across_runs(self) -> None:
        rows = [
            _row("r1-p2", "e7k", 2, 96, 1200, optimizer_run_id="100"),
            _row("r2-p3", "e7k", 3, 64, 900, optimizer_run_id="101"),
            _row("r2-p4", "e7k", 4, 48, 700, optimizer_run_id="101"),
            _row("r2-p5", "e7k", 5, 38, 650, optimizer_run_id="101"),
            _row("r2-p6", "e7k", 6, 32, 600, optimizer_run_id="101"),
            _row("r2-p7", "e7k", 7, 27, 590, optimizer_run_id="101"),
            _row("r1-p8", "e7k", 8, 24, 580, optimizer_run_id="100"),
        ]
        index = build_optimizer_index({"observations": rows}, "adaptive_radial_edge", optimizer_run_ids={"100", "101"})
        shapes = index["runners"][0]["shapes"]
        self.assertEqual([shape["pipelines"] for shape in shapes], [2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(index["runners"][0]["best_shape"]["pipelines"], 8)

    def test_historical_optimizer_profile_keeps_all_repeated_shape_observations(self) -> None:
        first = _row("r1-p4", "e7k", 4, 48, 700, optimizer_run_id="100")
        second = _row("r2-p4", "e7k", 4, 48, 680, optimizer_run_id="101")
        index = build_optimizer_index({"observations": [first, second]}, "adaptive_radial_edge", optimizer_run_ids={"100", "101"})
        shape = index["runners"][0]["shapes"][0]
        self.assertEqual(shape["observation_count"], 2)
        self.assertEqual(shape["fastest_wall_clock_seconds"], 680)
        self.assertEqual(shape["median_wall_clock_seconds"], 690)


    def test_optimizer_persistence_does_not_age_out_aggregate_or_shard_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregates = []
            for idx in range(520):
                row = _row(f"agg-{idx}", "e7k", idx + 1, 1, 100 + idx, optimizer_run_id=str(1000 + idx))
                aggregates.append(row)
            update_parallelism_index(root, aggregates)
            shards = [{
                "observation_id": f"shard-{idx}", "source": "execution-optimizer",
                "detector_id": "adaptive_radial_edge", "optimizer_run_id": "2000",
                "shape_sequence": idx // 100, "shard_index": idx,
            } for idx in range(5005)]
            update_parallelism_shards(root, shards)
            payload = json.loads((root / "parallelism-index.json").read_text(encoding="utf-8"))
            self.assertEqual(len([row for row in payload["observations"] if row.get("source") == "execution-optimizer"]), 520)
            self.assertEqual(len([row for row in payload["shard_observations"] if row.get("source") == "execution-optimizer"]), 5005)


    def test_optimizer_artifacts_filter_runner_metrics_to_requested_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "parallelism-index.json").write_text(json.dumps({
                "schema_version": "2.2",
                "observations": [_row("a", "e7k", 1, 192, 6840, optimizer_run_id="100")],
                "shard_observations": [],
            }), encoding="utf-8")
            metrics = root / "runner-metrics.jsonl"
            metrics.write_text(
                json.dumps({"optimizer_run_id": "99", "shape_sequence": 1, "load1": 1.0}) + "\n" +
                json.dumps({"optimizer_run_id": "100", "shape_sequence": 1, "load1": 2.0}) + "\n",
                encoding="utf-8",
            )
            paths = update_optimizer_artifacts(
                root,
                "adaptive_radial_edge",
                optimizer_run_id="100",
                runner_metrics_log=metrics,
            )
            payload = json.loads(paths["index"].read_text(encoding="utf-8"))
            samples = payload["runs"]["100"]["runner_metrics_samples"]
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["optimizer_run_id"], "100")

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
            metadata.write_text(json.dumps({"stop_reason": "throughput_plateau", "pipeline_enumeration": "adaptive", "optimization_wall_seconds": 302, "early_stop": {"stop_reason": "throughput_plateau", "required_consecutive_shapes": 3, "threshold_pct": 2.0}}), encoding="utf-8")
            paths = update_optimizer_artifacts(root, "adaptive_radial_edge", optimizer_run_id="100", run_metadata_path=metadata)
            payload = json.loads(paths["index"].read_text(encoding="utf-8"))
            self.assertIn("adaptive_radial_edge", payload["detectors"])
            self.assertEqual(payload["runs"]["100"]["shard_observation_count"], 1)
            preferred = payload["preferred_executor_configurations"]
            self.assertTrue(any(row["detector_id"] == "adaptive_radial_edge" and row["preferred_shape"]["pipelines"] == 8 for row in preferred))
            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("execution data below contains only shapes completed in this execution", markdown)
            self.assertIn("e7k", markdown)
            self.assertIn("Preferred Detector Run Configuration", markdown)
            self.assertIn("Preferred shape range (≤2%)", markdown)
            self.assertIn("Search method", markdown)
            self.assertIn("Search method legend", markdown)
            self.assertIn("powers-of-2", markdown)
            self.assertIn("Optimization time", markdown)
            self.assertIn("Shape time", markdown)
            self.assertIn("adaptive", markdown)
            self.assertIn("5m 2s", markdown)
            self.assertIn("8p/8t", markdown)
            self.assertIn("Detector Run Profile Plot", markdown)
            self.assertIn("Detector Pipeline-Thread Shape Optimization Data", markdown)
            self.assertIn("Δ from best", markdown)
            self.assertIn("0.00%", markdown)
            self.assertIn("<details>", markdown)
            self.assertIn("<summary><strong>Navigation</strong></summary>", markdown)
            self.assertIn("e9k", markdown)
            current_section = markdown.split("<summary><strong>3. Detector Pipeline-Thread Shape Optimization Data</strong></summary>", 1)[1]
            self.assertNotIn("e9k", current_section)
            svg = paths["heatmap"].read_text(encoding="utf-8")
            self.assertTrue(svg.startswith("<svg"))
            self.assertIn("detector pipelines (log₂ scale)", svg)
            self.assertIn("parameter sets / second", svg)
            self.assertIn("8t", svg)


    def test_shape_data_table_sorts_pipeline_count_least_to_greatest(self) -> None:
        index = build_optimizer_index(
            {"schema_version": "2.2", "observations": [
                _row("p8", "e7k", 8, 8, 420, optimizer_run_id="100"),
                _row("p4", "e7k", 4, 16, 500, optimizer_run_id="100"),
                _row("p7", "e7k", 7, 9, 430, optimizer_run_id="100"),
            ]},
            "adaptive_radial_edge",
            "100",
        )
        markdown = render_markdown(
            index,
            {"pipeline_enumeration": "adaptive", "optimization_wall_seconds": 1350},
            preferred_index=index,
        )
        section = markdown.split("<summary><strong>3. Detector Pipeline-Thread Shape Optimization Data</strong></summary>", 1)[1]
        self.assertLess(section.index("| 4 | 4 | 16 |"), section.index("| 7 | 7 | 9 |"))
        self.assertLess(section.index("| 7 | 7 | 9 |"), section.index("| 8 | 8 | 8 |"))
        self.assertIn("**Search method:** `adaptive`", markdown)


    def test_resumed_run_shape_section_describes_reused_checkpoint_shapes(self) -> None:
        index = build_optimizer_index(
            {"schema_version": "2.2", "observations": [_row("resumed", "e7k", 1, 192, 6840, optimizer_run_id="200")]},
            "adaptive_radial_edge",
            "200",
        )
        markdown = render_markdown(
            index,
            {"resumed_from_optimizer_run_id": "199", "stop_reason": "shape_range_complete"},
            preferred_index=index,
        )
        self.assertIn(
            "Shapes completed in this execution or reused from its compatible checkpoint are shown below.",
            markdown,
        )
        section = markdown.split("<summary><strong>3. Detector Pipeline-Thread Shape Optimization Data</strong></summary>", 1)[1]
        self.assertIn("| 1 | 1 | 192 | 192 |", section)


    def test_legacy_binary_search_method_displays_as_powers_of_2(self) -> None:
        parallelism_index = {"observations": [_row("a", "e7k", 1, 64, 2600, optimizer_run_id="100")]}
        index = build_optimizer_index(parallelism_index, "adaptive_radial_edge", optimizer_run_id="100")
        index["run_metadata_by_id"] = {"100": {"pipeline_enumeration": "binary", "optimization_wall_seconds": 2600}}
        markdown = render_markdown(index, run_metadata={"pipeline_enumeration": "binary"})
        self.assertIn("`powers-of-2`", markdown)
        self.assertNotIn("`binary`", markdown)


    def test_legacy_binary_search_method_is_normalized_inside_multi_method_display(self) -> None:
        parallelism_index = {"observations": [_row("a", "e7k", 1, 64, 2600, optimizer_run_id="100")]}
        index = build_optimizer_index(parallelism_index, "adaptive_radial_edge", optimizer_run_id="100")
        markdown = render_markdown(
            index,
            run_metadata={"pipeline_enumeration": "binary, exhaustive", "optimization_wall_seconds": 2600},
        )
        self.assertIn("`powers-of-2, exhaustive`", markdown)
        self.assertNotIn("`binary, exhaustive`", markdown)



if __name__ == "__main__":
    unittest.main()
