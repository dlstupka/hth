from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hth.optimizer_store import build_optimizer_index, render_all_markdown, render_heatmap_svg, render_markdown, select_preferred_shape, update_optimizer_artifacts
from hth.domain.execution_shape import optimizer_evidence_key
from hth.parallelism_store import update_parallelism_index, update_parallelism_shards
from hth.shape_prediction import record_prediction_observations


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
            "runner_name": "rh8-al97" if runner == "e7k" else runner,
            "runner_labels": ["self-hosted", "linux", runner],
            "cpu_model": f"CPU {runner}",
            "logical_cpu_count": 96 if runner == "e7k" else 32,
        },
    }


class OptimizerStoreTests(unittest.TestCase):


    def test_critical_optimizer_subset_is_retained_as_execution_evidence(self) -> None:
        row = _row("critical-p2", "e7k", 2, 192, 10.0, optimizer_run_id="200")
        row["strategy"] = "critical"
        row["possible_parameter_sets"] = 10000
        row["actual_parameter_sets"] = 11
        row["parameter_sets_per_second"] = 1.1
        index = build_optimizer_index({"observations": [row]}, "adaptive_radial_edge", "200")
        self.assertEqual(index["observation_count"], 1)
        self.assertEqual(index["runners"][0]["shapes"][0]["pipelines"], 2)
        self.assertEqual(index["runners"][0]["shapes"][0]["shards"], 2)

    def test_incomplete_exhaustive_optimizer_observation_is_still_rejected(self) -> None:
        row = _row("partial-exhaustive", "e7k", 2, 192, 10.0, optimizer_run_id="201")
        row["actual_parameter_sets"] = row["possible_parameter_sets"] - 1
        index = build_optimizer_index({"observations": [row]}, "adaptive_radial_edge", "201")
        self.assertEqual(index["observation_count"], 0)
        self.assertEqual(index["runners"], [])

    def test_search_scope_does_not_split_optimizer_evidence_identity(self) -> None:
        exhaustive = _row("old-exhaustive", "e7k", 2, 192, 20.0, optimizer_run_id="100")
        exhaustive["compatibility_key"] = "stable-evidence-and-runner"
        exhaustive["evidence_key"] = "stable-evidence"
        critical = _row("new-critical", "e7k", 3, 128, 10.0, optimizer_run_id="200")
        critical["strategy"] = "critical"
        critical["possible_parameter_sets"] = 10000
        critical["actual_parameter_sets"] = 11
        critical["parameter_sets_per_second"] = 1.1
        critical["compatibility_key"] = "stable-evidence-and-runner"
        critical["evidence_key"] = "stable-evidence"

        index = build_optimizer_index(
            {"observations": [exhaustive, critical]},
            "adaptive_radial_edge",
            optimizer_run_ids={"100", "200"},
        )

        self.assertEqual(len(index["runners"]), 1)
        self.assertEqual(index["runners"][0]["evidence_key"], optimizer_evidence_key(exhaustive))
        self.assertEqual(
            {shape["pipelines"] for shape in index["runners"][0]["shapes"]},
            {2, 3},
        )


    def test_canonical_preferred_shape_breaks_equal_displayed_throughput_ties_by_resources(self) -> None:
        shapes = [
            {"execution_shape": "6p/6s/64t", "pipelines": 6, "threads_per_pipeline": 64, "allocated_threads": 384, "fastest_wall_clock_seconds": 9.0, "parameter_sets_per_second": 27.004, "optimizer_shape_sequence": 2},
            {"execution_shape": "5p/5s/76t", "pipelines": 5, "threads_per_pipeline": 76, "allocated_threads": 380, "fastest_wall_clock_seconds": 9.0, "parameter_sets_per_second": 27.003, "optimizer_shape_sequence": 1},
        ]
        best = select_preferred_shape(shapes)
        self.assertIsNotNone(best)
        self.assertEqual(best["pipelines"], 5)

    def test_canonical_preferred_shape_uses_newest_run_before_resources_on_exact_rate_tie(self) -> None:
        shapes = [
            {"pipelines": 11, "threads_per_pipeline": 34, "allocated_threads": 374, "parameter_sets_per_second": 64.0, "optimizer_run_id": "100"},
            {"pipelines": 15, "threads_per_pipeline": 25, "allocated_threads": 375, "parameter_sets_per_second": 64.0, "optimizer_run_id": "101"},
        ]
        best = select_preferred_shape(shapes)
        self.assertIsNotNone(best)
        self.assertEqual(best["pipelines"], 15)

    def test_canonical_preferred_shape_keeps_resource_tiebreak_within_same_run(self) -> None:
        shapes = [
            {"pipelines": 11, "threads_per_pipeline": 34, "allocated_threads": 374, "parameter_sets_per_second": 64.0, "optimizer_run_id": "101"},
            {"pipelines": 15, "threads_per_pipeline": 25, "allocated_threads": 375, "parameter_sets_per_second": 64.0, "optimizer_run_id": "101"},
        ]
        best = select_preferred_shape(shapes)
        self.assertEqual(best["pipelines"], 11)

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
            {**_row("invalid", "e7k", 16, 4, 100, optimizer_run_id="100"),
             "valid": False, "invalid_reason": "single-detector pipeline fan-out bug"},
        ]}
        index = build_optimizer_index(parallelism, "adaptive_radial_edge", "100")
        self.assertEqual(index["observation_count"], 2)
        self.assertTrue(all(shape["pipelines"] not in {16, 64} for runner in index["runners"] for shape in runner["shapes"]))
        self.assertNotIn(">4t<", render_heatmap_svg(index))

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

    def test_preferred_configuration_observations_counts_all_compatible_measurements(self) -> None:
        rows = [
            _row("r1-p1", "e7k", 1, 192, 1200, optimizer_run_id="100"),
            _row("r1-p2", "e7k", 2, 96, 700, optimizer_run_id="100"),
            _row("r2-p2", "e7k", 2, 96, 680, optimizer_run_id="101"),
            _row("r2-p3", "e7k", 3, 64, 650, optimizer_run_id="101"),
        ]
        index = build_optimizer_index({"observations": rows}, "adaptive_radial_edge", optimizer_run_ids={"100", "101"})
        markdown = render_markdown(index, preferred_index=index)
        preferred = markdown.split("<summary><strong>1. Preferred Detector Run Configuration</strong></summary>", 1)[1].split("</details>", 1)[0]
        self.assertIn("| 4 |", preferred)

    def test_historical_optimizer_profile_keeps_all_repeated_shape_observations(self) -> None:
        first = _row("r1-p4", "e7k", 4, 48, 700, optimizer_run_id="100")
        second = _row("r2-p4", "e7k", 4, 48, 680, optimizer_run_id="101")
        recovered_copy = {
            **second,
            "observation_id": "legacy-copy-r2-p4",
            "optimizer_run_id": "legacy-published-deadbeef",
            "optimizer_intelligence_recovery": "published-summary-history",
            "runner": {
                "runner_label": "96t",
                "runner_name": "rh8-al97",
                "runner_labels": ["self-hosted", "96t"],
                "logical_cpu_count": 96,
            },
        }
        index = build_optimizer_index(
            {"observations": [first, second, recovered_copy]},
            "adaptive_radial_edge",
            optimizer_run_ids={"100", "101", "legacy-published-deadbeef"},
        )
        shape = index["runners"][0]["shapes"][0]
        # Native repeated measurements remain evidence; a recovered published
        # representation of an already-native measurement is not counted twice.
        self.assertEqual(shape["observation_count"], 2)
        self.assertEqual(index["runner_count"], 1)
        self.assertEqual(index["observation_count"], 2)
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
            payload = json.loads((root / "indexes" / "parallelism-index.json").read_text(encoding="utf-8"))
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
            self.assertIn("Δ from run best", markdown)
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



    def test_profile_plot_keeps_optimizer_runs_as_separate_provenance_series(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _row("old-a", "192t", 7, 54, 22, optimizer_run_id="33279699408"),
                _row("old-b", "192t", 9, 42, 20, optimizer_run_id="33279699408"),
                _row("new-a", "192t", 1, 384, 6, optimizer_run_id="33283602734"),
                _row("new-b", "192t", 2, 192, 8, optimizer_run_id="33283602734"),
            ]
            for row in rows:
                row["compatibility_key"] = "same-compatible-profile"
                row["runner"]["runner_name"] = "rh8-al318"

            index = build_optimizer_index({"observations": rows}, "adaptive_radial_edge")

            # Preferred-shape selection intentionally remains compatibility scoped.
            self.assertEqual(index["runner_count"], 1)
            # Plotting must preserve separate optimizer executions even on the same
            # concrete runner so distinct physical runs are never drawn as one curve.
            self.assertEqual(len(index["plot_series"]), 2)
            svg = render_heatmap_svg(index)
            self.assertIn("rh8-al318", svg)
            self.assertIn("run 33279699408", svg)
            self.assertIn("run 33283602734", svg)
            self.assertEqual(svg.count('<path d="'), 2)
            markdown = render_all_markdown([index])
            self.assertIn("Optimizer run", markdown)
            self.assertIn("33279699408", markdown)
            self.assertIn("33283602734", markdown)

    def test_shape_report_preserves_and_labels_startup_overhead(self) -> None:
        row = _row("startup", "e7k", 8, 48, 240.0, optimizer_run_id="555")
        row["startup_overhead_seconds"] = 180.0
        row["startup_overhead_included_in_wall_clock"] = True
        index = build_optimizer_index({"observations": [row]}, "adaptive_radial_edge", "555")
        shape = index["runners"][0]["shapes"][0]
        self.assertEqual(shape["startup_overhead_seconds"], 180.0)
        markdown = render_markdown(index, {"pipeline_enumeration": "adaptive"}, preferred_index=index)
        self.assertIn("Startup overhead", markdown)
        self.assertIn("| 3m |", markdown)
        self.assertIn("remains included in **Wall**", markdown)
        self.assertIn("Per-shard parameter-set throughput is timed after fan-out", markdown)

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



    def test_optimizer_report_notates_prediction_coverage_and_verifies_saved_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = _row("p8", "e7k", 8, 24, 100.0, optimizer_run_id="321")
            (root / "parallelism-index.json").write_text(json.dumps({"observations": [row], "shard_observations": []}), encoding="utf-8")
            (root / "indexes").mkdir(exist_ok=True)
            # Legacy optimizer indexes embedded prediction history before the
            # canonical optimizer-predictions index was guaranteed to exist.
            # Publishing must migrate that history rather than reporting 0/0.
            (root / "indexes" / "optimizer-index.json").write_text(json.dumps({
                "schema_version": 1,
                "detectors": {"adaptive_radial_edge": {"prediction_history": [{
                    "prediction_id": "pred1",
                    "detector_id": "adaptive_radial_edge",
                    "status": "pending",
                    "target_runner": {"runner_name": "rh8-al97", "logical_cpu_count": 96},
                    "predicted_shape": {"pipelines": 7, "threads_per_pipeline": 24, "allocated_threads": 168},
                    "workload": {},
                }]}},
                "runs": {},
            }), encoding="utf-8")
            # A completed regression that actually used a predicted shape is the
            # canonical producer of a prediction check. This survives sharding
            # because records are keyed to the detector/GitHub execution, not a
            # transient planner file or individual shard.
            record_prediction_observations(
                root / "indexes" / "optimizer-predictions.json",
                [{
                    "observation_id": "900:scantailor_page_frame:shard-1",
                    "run_id": "shard-1",
                    "detector_id": "scantailor_page_frame",
                    "mode": "full",
                    "strategy": "exhaustive",
                    "execution_shape_source": "predicted-low-linear-vcpu",
                    "active_pipelines": 5,
                    "threads_per_pipeline": 12,
                    "allocated_threads": 60,
                    "detector_config_sha256": "scan-cfg",
                    "golden_set_sha256": "gold",
                    "max_dimension": 1800,
                    "observed_at_utc": "2026-08-30T12:00:00Z",
                    "runner": {"runner_name": "github-hosted", "runner_label": "github-hosted", "logical_cpu_count": 32},
                    "build": {"github_run_id": "900"},
                }, {
                    "observation_id": "900:scantailor_page_frame:shard-2",
                    "run_id": "shard-2",
                    "detector_id": "scantailor_page_frame",
                    "mode": "full",
                    "strategy": "exhaustive",
                    "execution_shape_source": "predicted-low-linear-vcpu",
                    "active_pipelines": 5,
                    "threads_per_pipeline": 12,
                    "allocated_threads": 60,
                    "detector_config_sha256": "scan-cfg",
                    "golden_set_sha256": "gold",
                    "max_dimension": 1800,
                    "observed_at_utc": "2026-08-30T12:00:01Z",
                    "runner": {"runner_name": "github-hosted", "runner_label": "github-hosted", "logical_cpu_count": 32},
                    "build": {"github_run_id": "900"},
                }],
            )
            metadata = root / "run-metadata.json"
            metadata.write_text(json.dumps({"pipeline_enumeration": "adaptive"}), encoding="utf-8")
            update_optimizer_artifacts(root, "adaptive_radial_edge", optimizer_run_id="321", run_metadata_path=metadata)
            predictions = json.loads((root / "indexes" / "optimizer-predictions.json").read_text(encoding="utf-8"))
            adaptive = next(row for row in predictions["predictions"] if row.get("detector_id") == "adaptive_radial_edge")
            scantailor = [row for row in predictions["predictions"] if row.get("detector_id") == "scantailor_page_frame"]
            self.assertEqual(adaptive["status"], "verified")
            self.assertEqual(adaptive["verification"]["actual_shape"]["pipelines"], 8)
            self.assertEqual(len(scantailor), 1)
            self.assertEqual(scantailor[0]["status"], "pending")
            self.assertEqual(scantailor[0]["predicted_shape"]["pipelines"], 5)
            summary = (root / "execution-optimizer" / "adaptive_radial_edge" / "summary.md").read_text(encoding="utf-8")
            self.assertIn("Shape-prediction coverage", summary)
            self.assertIn("Desired / missing optimization data", summary)
            self.assertIn("1 verified / 0 pending", summary)


if __name__ == "__main__":
    unittest.main()
