from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from hth.report_generator import calibration_run_dirs, smoke_run_dirs, generate_calibration_manifest, generate_optimizer_report, generate_optimizer_report_all


class ReportGeneratorTests(unittest.TestCase):
    @staticmethod
    def _write_completed_optimizer_summary(root: Path, detector: str, run_id: str) -> None:
        path = root / "execution-optimizer" / detector / "summary.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"### Execution optimizer summary\n\nDetector: `{detector}`  \nOptimizer run: **{run_id}** — this table contains only shapes completed in this execution.\n",
            encoding="utf-8",
        )

    def test_calibration_manifest_resolves_best_record_per_detector(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "records" / "a-smoke").mkdir(parents=True)
            (root / "records" / "a-full").mkdir(parents=True)
            (root / "records" / "b-full").mkdir(parents=True)
            index = {
                "entries": [
                    {"detector_id": "a", "calibration_status": "provisional", "record_path": "records/a-smoke", "created_at_utc": "2026-01-01"},
                    {"detector_id": "a", "calibration_status": "provisional", "record_path": "records/a-full", "created_at_utc": "2026-01-02"},
                    {"detector_id": "b", "calibration_status": "authoritative", "record_path": "records/b-full", "created_at_utc": "2026-01-03"},
                ]
            }
            (root / "calibration-index.json").write_text(json.dumps(index), encoding="utf-8")
            runs = calibration_run_dirs(root)
            self.assertEqual([path.name for path in runs], ["a-full", "b-full"])

    def test_calibration_manifest_keeps_full_and_smoke_views_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "records" / "old-full").mkdir(parents=True)
            (root / "records" / "new-smoke").mkdir(parents=True)
            index = {
                "entries": [
                    {
                        "detector_id": "doc_ufcn_page_mask",
                        "calibration_status": "authoritative",
                        "record_path": "records/old-full",
                        "created_at_utc": "2026-08-19T23:00:00Z",
                        "search": {"strategy": "exhaustive", "exhaustive_complete": True},
                        "selection": {"best_avg_iou": 0.97, "minimum_iou": 0.95},
                        "build": {"pipeline_commit": "old-pipeline"},
                    },
                    {
                        "detector_id": "doc_ufcn_page_mask",
                        "calibration_status": "provisional",
                        "record_path": "records/new-smoke",
                        "created_at_utc": "2026-08-20T03:00:00Z",
                        "search": {"strategy": "smoke", "exhaustive_complete": False},
                        "selection": {"best_avg_iou": 0.90, "minimum_iou": 0.80},
                        "build": {"pipeline_commit": "new-pipeline"},
                    },
                ]
            }
            (root / "calibration-index.json").write_text(json.dumps(index), encoding="utf-8")
            self.assertEqual([path.name for path in calibration_run_dirs(root)], ["old-full"])
            self.assertEqual([path.name for path in smoke_run_dirs(root)], ["new-smoke"])


    def test_calibration_manifest_reads_flattened_persisted_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = root / "records" / "a-full"
            record.mkdir(parents=True)
            (root / "calibration-index.json").write_text(json.dumps({"entries": [{
                "detector_id": "a", "calibration_status": "provisional",
                "record_path": "records/a-full", "intelligence_path": "records/a-full/calibration-intelligence.json",
                "created_at_utc": "2026-01-02", "golden_set_sha256": "unknown"
            }]}), encoding="utf-8")
            (record / "manifest.json").write_text(json.dumps({"detector": "a", "status": "passed"}), encoding="utf-8")
            (record / "parameters.json").write_text(json.dumps({}), encoding="utf-8")
            (record / "RUN-INFO.json").write_text(json.dumps({"elapsed_seconds": 1}), encoding="utf-8")
            (record / "summary.json").write_text(json.dumps({
                "winner": {"parameter_set_id": "p1", "summary": {"mean_iou": 0.9, "minimum_iou": 0.8, "stddev_iou": 0.01, "failure_count": 0}},
                "baseline": {"summary": {"mean_iou": 0.7}}, "page_ordinals": [1], "parameter_set_count": 1
            }), encoding="utf-8")
            (record / "calibration-intelligence.json").write_text(json.dumps({"available": True, "detector": "a"}), encoding="utf-8")
            output = root / "report.md"
            generate_calibration_manifest(root, output, golden_set=None, pipeline_repository="", results_repository="", results_commit="", run_url="")
            self.assertTrue(output.is_file())
            self.assertIn("Regression Manifest", output.read_text(encoding="utf-8"))

    def test_optimizer_report_recovers_run_from_parallelism_for_legacy_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            optimizer = {"schema_version": 1, "detectors": {"adaptive_radial_edge": {}}}
            parallelism = {
                "observations": [
                    {"detector_id": "adaptive_radial_edge", "optimizer_run_id": "100", "source": "execution-optimizer", "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10, "active_pipelines": 1, "shards": 1, "threads_per_pipeline": 2, "allocated_threads": 2, "wall_clock_seconds": 20, "parameter_sets_per_second": 1.0, "execution_shape": "1p/2t", "captured_at_utc": "2026-01-01T00:00:00Z", "runner": {"runner_label": "e7k", "name": "host", "logical_cpus": 96}},
                    {"detector_id": "adaptive_radial_edge", "optimizer_run_id": "200", "source": "execution-optimizer", "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10, "active_pipelines": 2, "shards": 2, "threads_per_pipeline": 1, "allocated_threads": 2, "wall_clock_seconds": 10, "parameter_sets_per_second": 2.0, "execution_shape": "2p/1t", "optimizer_shape_sequence": 1, "captured_at_utc": "2026-01-02T00:00:00Z", "runner": {"runner_label": "e7k", "name": "host", "logical_cpus": 96}},
                ]
            }
            (root / "optimizer-index.json").write_text(json.dumps(optimizer), encoding="utf-8")
            (root / "parallelism-index.json").write_text(json.dumps(parallelism), encoding="utf-8")
            self._write_completed_optimizer_summary(root, "adaptive_radial_edge", "200")
            paths = generate_optimizer_report(root, "adaptive_radial_edge", root / "out")
            text = paths["summary"].read_text(encoding="utf-8")
            self.assertIn("Optimizer run: **200**", text)
            self.assertIn("| 2 | 2 | 1 |", text)
            self.assertNotIn("| 1 | 1 | 2 |", text)

    def test_optimizer_report_uses_latest_run_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            optimizer = {
                "runs": {
                    "1": {"detector_id": "adaptive_radial_edge", "updated_at_utc": "2026-01-01T00:00:00Z", "run_metadata": {}},
                    "2": {"detector_id": "adaptive_radial_edge", "updated_at_utc": "2026-01-02T00:00:00Z", "run_metadata": {}},
                }
            }
            parallelism = {
                "observations": [
                    {"detector_id": "adaptive_radial_edge", "optimizer_run_id": "1", "source": "execution-optimizer", "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10, "active_pipelines": 1, "shards": 1, "threads_per_pipeline": 1, "allocated_threads": 1, "wall_clock_seconds": 20, "parameter_sets_per_second": 1.0, "execution_shape": "1p/1t", "runner": {"runner_label": "e7k", "name": "host", "logical_cpus": 96}},
                    {"detector_id": "adaptive_radial_edge", "optimizer_run_id": "2", "source": "execution-optimizer", "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10, "active_pipelines": 2, "shards": 2, "threads_per_pipeline": 1, "allocated_threads": 2, "wall_clock_seconds": 10, "parameter_sets_per_second": 2.0, "execution_shape": "2p/1t", "optimizer_shape_sequence": 1, "runner": {"runner_label": "e7k", "name": "host", "logical_cpus": 96}},
                ]
            }
            (root / "optimizer-index.json").write_text(json.dumps(optimizer), encoding="utf-8")
            (root / "parallelism-index.json").write_text(json.dumps(parallelism), encoding="utf-8")
            self._write_completed_optimizer_summary(root, "adaptive_radial_edge", "2")
            paths = generate_optimizer_report(root, "adaptive_radial_edge", root / "out")
            text = paths["summary"].read_text(encoding="utf-8")
            self.assertIn("Optimizer run: **2**", text)
            self.assertIn("| 2 | 2 | 1 |", text)
            self.assertNotIn("| 1 | 1 | 1 |", text)

    def test_optimizer_report_profile_uses_current_run_not_coalesced_historical_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            optimizer = {
                "schema_version": 2,
                "runs": {
                    "1": {"detector_id": "eynollah_page_mask", "status": "completed", "run_metadata": {"pipeline_enumeration": "exhaustive"}},
                    "2": {"detector_id": "eynollah_page_mask", "status": "completed", "run_metadata": {"pipeline_enumeration": "exhaustive"}},
                },
                "detectors": {"eynollah_page_mask": {}},
            }
            parallelism = {"observations": [
                {"detector_id": "eynollah_page_mask", "optimizer_run_id": "1", "source": "execution-optimizer", "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10, "active_pipelines": 32, "shards": 32, "threads_per_pipeline": 12, "allocated_threads": 384, "wall_clock_seconds": 40, "parameter_sets_per_second": 2.0, "execution_shape": "32p/12t", "optimizer_shape_sequence": 1, "runner": {"runner_label": "e7k", "runner_name": "host", "logical_cpu_count": 192}},
                {"detector_id": "eynollah_page_mask", "optimizer_run_id": "2", "source": "execution-optimizer", "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10, "active_pipelines": 2, "shards": 2, "threads_per_pipeline": 192, "allocated_threads": 384, "wall_clock_seconds": 20, "parameter_sets_per_second": 4.0, "execution_shape": "2p/192t", "optimizer_shape_sequence": 1, "runner": {"runner_label": "e7k", "runner_name": "host", "logical_cpu_count": 192}},
                {"detector_id": "eynollah_page_mask", "optimizer_run_id": "2", "source": "execution-optimizer", "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10, "active_pipelines": 16, "shards": 16, "threads_per_pipeline": 24, "allocated_threads": 384, "wall_clock_seconds": 10, "parameter_sets_per_second": 8.0, "execution_shape": "16p/24t", "optimizer_shape_sequence": 2, "runner": {"runner_label": "e7k", "runner_name": "host", "logical_cpu_count": 192}},
            ]}
            (root / "optimizer-index.json").write_text(json.dumps(optimizer), encoding="utf-8")
            (root / "parallelism-index.json").write_text(json.dumps(parallelism), encoding="utf-8")
            self._write_completed_optimizer_summary(root, "eynollah_page_mask", "2")
            paths = generate_optimizer_report(root, "eynollah_page_mask", root / "out")
            profile = paths["profile"].read_text(encoding="utf-8")
            self.assertIn(">16<", profile)
            self.assertNotIn(">32<", profile)
            self.assertIn("192t", profile)
            self.assertIn("24t", profile)

    def test_optimizer_report_retains_legacy_completed_history_while_current_table_stays_latest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "test"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            persisted = root / "execution-optimizer" / "adaptive_radial_edge"
            persisted.mkdir(parents=True)
            persisted.joinpath("summary.md").write_text(
                "| Runner | Pipelines | Shards | Threads / pipeline | Allocated | Wall | Sets/s |\n"
                "|---|---:|---:|---:|---:|---:|---:|\n"
                "| 192t — rh8-legacy (192 vCPU) | 11 | 11 | 34 | 374 | 6s | 42.67 |\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "legacy optimizer"], check=True)

            # A later modern report is derived presentation state.  Even if it
            # contains a runner/shape combination that looks like a physical
            # measurement, git-history recovery must not import it as legacy
            # evidence and leak the shape onto that runner.
            persisted.joinpath("summary.md").write_text(
                "### Execution optimizer summary\n\n"
                "Optimizer run: **150**\n\n"
                "| Runner | Pipelines | Shards | Threads / pipeline | Allocated | Wall | Sets/s |\n"
                "|---|---:|---:|---:|---:|---:|---:|\n"
                "| 192t — rh8-modern (192 vCPU) | 1 | 1 | 384 | 384 | 6s | 9.17 |\n",
                encoding="utf-8",
            )
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "derived modern report"], check=True)

            optimizer = {"schema_version": 1, "runs": {
                "200": {"detector_id": "adaptive_radial_edge", "status": "completed", "run_metadata": {"stop_reason": "range_complete"}}
            }}
            parallelism = {"observations": [
                {"observation_id": "modern", "detector_id": "adaptive_radial_edge", "optimizer_run_id": "200", "source": "execution-optimizer", "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10, "active_pipelines": 2, "shards": 2, "threads_per_pipeline": 96, "allocated_threads": 192, "wall_clock_seconds": 1200, "parameter_sets_per_second": 5.5, "execution_shape": "2p/2s/96t", "optimizer_shape_sequence": 2, "captured_at_utc": "2026-01-02T00:00:00Z", "runner": {"runner_label": "e7k", "runner_name": "host", "logical_cpu_count": 96}},
            ]}
            (root / "optimizer-index.json").write_text(json.dumps(optimizer), encoding="utf-8")
            (root / "parallelism-index.json").write_text(json.dumps(parallelism), encoding="utf-8")
            self._write_completed_optimizer_summary(root, "adaptive_radial_edge", "200")
            subprocess.run(["git", "-C", str(root), "add", "."], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "modern optimizer"], check=True)

            paths = generate_optimizer_report(root, "adaptive_radial_edge", root / "out")
            summary = paths["summary"].read_text(encoding="utf-8")
            self.assertIn("Optimizer run: **200**", summary)
            self.assertIn("| **e7k — host (96 vCPU)** | 2 | 2 | 96 |", summary)
            self.assertIn("rh8-legacy", summary)
            self.assertNotIn("rh8-modern", summary)
            self.assertNotIn("| **192t — rh8-legacy", summary)
            aggregate = generate_optimizer_report_all(root, root / "all")
            aggregate_profile = (aggregate["profiles"] / "adaptive_radial_edge.svg").read_text(encoding="utf-8")
            self.assertIn("rh8-legacy", aggregate_profile)
            self.assertIn("e7k — host", aggregate_profile)


    def test_optimizer_report_recovers_pre_run_id_published_table_without_history_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "optimizer-index.json").write_text(json.dumps({"schema_version": 1, "detectors": {}}), encoding="utf-8")
            (root / "parallelism-index.json").write_text(json.dumps({"observations": []}), encoding="utf-8")
            persisted = root / "execution-optimizer" / "adaptive_radial_edge"
            persisted.mkdir(parents=True)
            persisted.joinpath("summary.md").write_text("""### Execution optimizer summary

| Runner | Pipelines | Shards | Threads / pipeline | Allocated | Wall | Sets/s | Speedup | Efficiency | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e7k — rh8-a197 (96 vCPU) | 1 | 1 | 192 | 192 | 44m 32s | 2.46 | 1.00× | unknown | 1 |
| **e7k — rh8-a197 (96 vCPU)** | 64 | 64 | 3 | 192 | 1m 32s | 71.33 | 29.04× | 84.9% | 1 |
| unknown — rh8-a197 (96 vCPU) | 1 | 1 | 192 | 192 | 45m 27s | 2.41 | 1.00× | unknown | 1 |
| **unknown — rh8-a197 (96 vCPU)** | 64 | 64 | 1 | 64 | 1m 28s | unknown | 31.02× | unknown | 4 |
""", encoding="utf-8")
            persisted.joinpath("heatmap.svg").write_text("<svg>legacy</svg>\n", encoding="utf-8")
            paths = generate_optimizer_report(root, "adaptive_radial_edge", root / "out")
            summary = paths["summary"].read_text(encoding="utf-8")
            profile = paths["profile"].read_text(encoding="utf-8")
            self.assertIn("Optimizer run: **legacy-published**", summary)
            self.assertIn("e7k — rh8-a197", summary)
            self.assertNotIn("unknown — rh8-a197", summary)
            self.assertIn("detector pipelines (log₂ scale)", profile)
            self.assertIn("parameter sets / second", profile)

    def test_optimizer_report_recovers_wide_pre_run_id_published_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "optimizer-index.json").write_text(json.dumps({"schema_version": 1, "detectors": {}}), encoding="utf-8")
            (root / "parallelism-index.json").write_text(json.dumps({"observations": []}), encoding="utf-8")
            persisted = root / "execution-optimizer" / "adaptive_radial_edge"
            persisted.mkdir(parents=True)
            persisted.joinpath("summary.md").write_text("""### Execution optimizer summary

| Runner | Pipelines | Shards | Threads / pipeline | Allocated threads | Fastest wall | Median wall | Sets/s | Speedup vs 1 pipeline | Efficiency | Runs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e7k — rh8-a197 (96 vCPU) | 1 | 1 | 192 | 192 | 44m 32s | 44m 32s | 2.46 | 1.00× | unknown | 1 |
| **e7k — rh8-a197 (96 vCPU)** | 64 | 64 | 3 | 192 | 1m 32s | 1m 32s | 71.33 | 29.04× | 84.9% | 1 |
| unknown — rh8-a197 (96 vCPU) | 1 | 1 | 192 | 192 | 45m 27s | 45m 27s | 2.41 | 1.00× | unknown | 1 |
""", encoding="utf-8")
            persisted.joinpath("heatmap.svg").write_text("<svg>legacy</svg>\n", encoding="utf-8")
            paths = generate_optimizer_report(root, "adaptive_radial_edge", root / "out")
            summary = paths["summary"].read_text(encoding="utf-8")
            profile = paths["profile"].read_text(encoding="utf-8")
            self.assertIn("Optimizer run: **legacy-published**", summary)
            self.assertIn("71.33", summary)
            self.assertNotIn("unknown — rh8-a197", summary)
            self.assertIn("detector pipelines (log₂ scale)", profile)
            self.assertIn("parameter sets / second", profile)

    def test_optimizer_report_rejects_legacy_completion_marker_without_run_tagged_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "optimizer-index.json").write_text(json.dumps({"schema_version": 1, "detectors": {}}), encoding="utf-8")
            (root / "parallelism-index.json").write_text(json.dumps({"observations": []}), encoding="utf-8")
            persisted = root / "execution-optimizer" / "adaptive_radial_edge"
            persisted.mkdir(parents=True)
            (persisted / "summary.md").write_text("legacy completed optimizer report\n", encoding="utf-8")
            (persisted / "heatmap.svg").write_text("<svg>legacy</svg>\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No completed persisted optimizer run"):
                generate_optimizer_report(root, "adaptive_radial_edge", root / "out")

    def test_optimizer_report_does_not_report_incomplete_shard_only_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "optimizer-index.json").write_text(json.dumps({"detectors": {"adaptive_radial_edge": {}}}), encoding="utf-8")
            shards = []
            for idx in range(2):
                shards.append({"detector_id": "adaptive_radial_edge", "optimizer_run_id": "777", "shape_sequence": 2,
                    "shard_index": idx, "shard_count": 2, "threads_per_pipeline": 48, "wall_clock_seconds": 10 + idx,
                    "actual_parameter_sets": 50, "observed_at_utc": f"2026-01-03T00:00:0{idx}Z",
                    "runner": {"runner_label": "e7k", "runner_name": "host", "logical_cpu_count": 96}})
            (root / "parallelism-index.json").write_text(json.dumps({"observations": [], "shard_observations": shards}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No completed persisted optimizer run"):
                generate_optimizer_report(root, "adaptive_radial_edge", root / "out")

    def test_optimizer_report_accepts_completed_run_marker_from_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            optimizer = {"runs": {"42": {
                "detector_id": "adaptive_radial_edge",
                "updated_at_utc": "2026-01-04T00:00:00Z",
                "run_metadata": {"stop_reason": "throughput_plateau"},
            }}}
            row = {"detector_id": "adaptive_radial_edge", "optimizer_run_id": "42", "source": "execution-optimizer",
                   "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10,
                   "active_pipelines": 4, "shards": 4, "threads_per_pipeline": 2, "allocated_threads": 8,
                   "wall_clock_seconds": 5, "parameter_sets_per_second": 2.0, "execution_shape": "4p/4s/2t",
                   "optimizer_shape_sequence": 1, "runner": {"runner_label": "e7k", "runner_name": "host", "logical_cpu_count": 96}}
            (root / "optimizer-index.json").write_text(json.dumps(optimizer), encoding="utf-8")
            (root / "parallelism-index.json").write_text(json.dumps({"observations": [row]}), encoding="utf-8")
            paths = generate_optimizer_report(root, "adaptive_radial_edge", root / "out")
            self.assertIn("Optimizer run: **42**", paths["summary"].read_text(encoding="utf-8"))

    def test_optimizer_report_uses_manifest_style_navigation_and_collapsible_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            optimizer = {"runs": {"42": {
                "detector_id": "adaptive_radial_edge",
                "updated_at_utc": "2026-01-04T00:00:00Z",
                "run_metadata": {"stop_reason": "range_complete"},
            }}}
            row = {"detector_id": "adaptive_radial_edge", "optimizer_run_id": "42", "source": "execution-optimizer",
                   "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10,
                   "active_pipelines": 4, "shards": 4, "threads_per_pipeline": 2, "allocated_threads": 8,
                   "wall_clock_seconds": 5, "parameter_sets_per_second": 2.0, "execution_shape": "4p/4s/2t",
                   "optimizer_shape_sequence": 1, "runner": {"runner_label": "e7k", "runner_name": "host", "logical_cpu_count": 96}}
            (root / "optimizer-index.json").write_text(json.dumps(optimizer), encoding="utf-8")
            (root / "parallelism-index.json").write_text(json.dumps({"observations": [row]}), encoding="utf-8")
            paths = generate_optimizer_report(root, "adaptive_radial_edge", root / "out")
            summary = paths["summary"].read_text(encoding="utf-8")
            self.assertIn("<summary><strong>Navigation</strong></summary>", summary)
            self.assertIn("<summary><strong>1. Preferred Detector Run Configuration</strong></summary>", summary)
            self.assertIn("Preferred shape range (≤2%)", summary)
            self.assertIn("<summary><strong>2. Detector Run Profile Plot</strong></summary>", summary)
            self.assertIn("<summary><strong>3. Detector Pipeline-Thread Shape Optimization Data</strong></summary>", summary)
            self.assertGreaterEqual(summary.count("<details open>"), 3)
            self.assertIn("[↑ Back to Navigation](#table-of-contents)", summary)

    def test_optimizer_report_missing_detector_falls_back_to_all_available_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            optimizer = {"runs": {"42": {
                "detector_id": "adaptive_radial_edge",
                "updated_at_utc": "2026-01-04T00:00:00Z",
                "run_metadata": {"stop_reason": "range_complete"},
            }}}
            row = {"detector_id": "adaptive_radial_edge", "optimizer_run_id": "42", "source": "execution-optimizer",
                   "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10,
                   "active_pipelines": 4, "shards": 4, "threads_per_pipeline": 2, "allocated_threads": 8,
                   "wall_clock_seconds": 5, "parameter_sets_per_second": 2.0, "execution_shape": "4p/4s/2t",
                   "optimizer_shape_sequence": 1,
                   "runner": {"runner_label": "e7k", "runner_name": "host", "logical_cpu_count": 96,
                              "cpu_model": "Example CPU", "physical_core_count": 48, "memory_gib": 2000.0}}
            (root / "optimizer-index.json").write_text(json.dumps(optimizer), encoding="utf-8")
            (root / "parallelism-index.json").write_text(json.dumps({"observations": [row]}), encoding="utf-8")

            paths = generate_optimizer_report(root, "contour", root / "out")
            summary = paths["summary"].read_text(encoding="utf-8")

            self.assertIn("This is currently all available optimization data.", summary)
            self.assertIn("Detector: `all`", summary)
            self.assertIn("| adaptive_radial_edge |", summary)
            self.assertTrue((root / "out" / "profiles" / "adaptive_radial_edge.svg").is_file())


    def test_optimizer_report_all_profile_always_includes_latest_completed_runner_series(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            detector = "signed_polar_boundary_vote"
            # Run 100 models a legacy completed execution: its aggregate
            # observations survive, but it predates per-run completion metadata.
            # Publishing run 101 must not make that valid historical runner vanish.
            optimizer = {"runs": {
                "101": {"detector_id": detector, "updated_at_utc": "2026-08-27T00:00:00Z", "run_metadata": {"pipeline_enumeration": "adaptive", "stop_reason": "range_complete"}},
            }}
            old = {"detector_id": detector, "optimizer_run_id": "100", "source": "execution-optimizer",
                   "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 256, "possible_parameter_sets": 256,
                   "optimizer_benchmark_parameter_sets": 256, "active_pipelines": 40, "shards": 40,
                   "threads_per_pipeline": 9, "allocated_threads": 360, "wall_clock_seconds": 3,
                   "parameter_sets_per_second": 87.48, "execution_shape": "40p/40s/9t", "optimizer_shape_sequence": 1,
                   "compatibility_key": "compatible",
                   "runner": {"runner_label": "192t", "runner_name": "rh8-a1319", "logical_cpu_count": 192}}
            new = dict(old)
            new.update({"optimizer_run_id": "101", "active_pipelines": 11, "shards": 11, "threads_per_pipeline": 34,
                        "allocated_threads": 374, "wall_clock_seconds": 4, "parameter_sets_per_second": 64.0,
                        "execution_shape": "11p/11s/34t", "optimizer_shape_sequence": 1})
            new["runner"] = {"runner_label": "192t", "runner_name": "rh8-a1328", "logical_cpu_count": 192}
            (root / "optimizer-index.json").write_text(json.dumps(optimizer), encoding="utf-8")
            (root / "parallelism-index.json").write_text(json.dumps({"observations": [old, new]}), encoding="utf-8")
            self._write_completed_optimizer_summary(root, detector, "101")

            paths = generate_optimizer_report_all(root, root / "out")
            svg = (paths["profiles"] / f"{detector}.svg").read_text(encoding="utf-8")
            self.assertIn("rh8-a1319", svg)
            self.assertIn("rh8-a1328", svg)
            self.assertIn("34t", svg)

    def test_optimizer_report_all_coalesces_completed_detectors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            optimizer = {"runs": {}}
            observations = []
            for ordinal, detector in enumerate(("adaptive_radial_edge", "grabcut"), start=1):
                run_id = str(40 + ordinal)
                optimizer["runs"][run_id] = {
                    "detector_id": detector,
                    "updated_at_utc": f"2026-01-04T00:00:0{ordinal}Z",
                    "run_metadata": {"stop_reason": "range_complete"},
                }
                observations.append({
                    "detector_id": detector, "optimizer_run_id": run_id, "source": "execution-optimizer",
                    "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10,
                    "active_pipelines": ordinal * 2, "shards": ordinal * 2, "threads_per_pipeline": 4,
                    "allocated_threads": ordinal * 8, "wall_clock_seconds": 10 - ordinal,
                    "parameter_sets_per_second": float(ordinal), "execution_shape": f"{ordinal * 2}p/{ordinal * 2}s/4t",
                    "optimizer_shape_sequence": 1,
                    "runner": {"runner_label": "e7k", "runner_name": "host", "logical_cpu_count": 96,
                               "cpu_model": "Example CPU", "physical_core_count": 48, "memory_gib": 2000.0},
                })
            (root / "optimizer-index.json").write_text(json.dumps(optimizer), encoding="utf-8")
            (root / "parallelism-index.json").write_text(json.dumps({"observations": observations}), encoding="utf-8")
            paths = generate_optimizer_report(root, "all", root / "out")
            summary = paths["summary"].read_text(encoding="utf-8")
            self.assertIn("Detector: `all`", summary)
            self.assertIn("| adaptive_radial_edge |", summary)
            self.assertIn("| grabcut |", summary)
            self.assertIn("profiles/adaptive_radial_edge.svg", summary)
            self.assertIn("profiles/grabcut.svg", summary)
            self.assertTrue((root / "out" / "profiles" / "adaptive_radial_edge.svg").is_file())
            self.assertTrue((root / "out" / "profiles" / "grabcut.svg").is_file())
            self.assertIn("  - [grabcut](#detector-run-profile-grabcut)", summary)



    def test_optimizer_report_recovers_when_optimizer_index_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parallelism = {"observations": [
                {"detector_id": "adaptive_radial_edge", "optimizer_run_id": "200", "source": "execution-optimizer",
                 "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10,
                 "active_pipelines": 2, "shards": 2, "threads_per_pipeline": 1, "allocated_threads": 2,
                 "wall_clock_seconds": 10, "parameter_sets_per_second": 2.0, "execution_shape": "2p/1t",
                 "optimizer_shape_sequence": 1, "captured_at_utc": "2026-01-02T00:00:00Z",
                 "runner": {"runner_label": "e7k", "name": "host", "logical_cpus": 96}},
            ]}
            indexes = root / "indexes"
            indexes.mkdir()
            (indexes / "parallelism-index.json").write_text(json.dumps(parallelism), encoding="utf-8")
            self._write_completed_optimizer_summary(root, "adaptive_radial_edge", "200")

            paths = generate_optimizer_report(root, "adaptive_radial_edge", root / "out")
            text = paths["summary"].read_text(encoding="utf-8")
            self.assertIn("Optimizer run: **200**", text)
            self.assertIn("| 2 | 2 | 1 |", text)

    def test_optimizer_report_recovers_published_table_when_indexes_lost_run_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            indexes = root / "indexes"
            indexes.mkdir()
            (indexes / "parallelism-index.json").write_text(json.dumps({"observations": []}), encoding="utf-8")
            persisted = root / "execution-optimizer" / "adaptive_radial_edge"
            persisted.mkdir(parents=True)
            (persisted / "summary.md").write_text(
                "### Execution optimizer summary\n\n"
                "Detector: `adaptive_radial_edge`  \n"
                "Optimizer run: **200** — completed.\n\n"
                "| Runner | Pipelines | Shards | Threads / pipeline | Allocated threads | Fastest wall | Median wall | Sets/s | Speedup vs 1 pipeline | Efficiency | Runs |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
                "| e7k — host (96 vCPU) | 2 | 2 | 48 | 96 | 10s | 10s | 2.0 | 2.00× | 100% | 1 |\n",
                encoding="utf-8",
            )

            paths = generate_optimizer_report_all(root, root / "out")
            text = paths["summary"].read_text(encoding="utf-8")
            self.assertIn("adaptive_radial_edge", text)
            self.assertIn("2.00", text)

    def test_optimizer_report_all_discovers_persisted_detectors_without_optimizer_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parallelism = {"observations": [
                {"detector_id": "adaptive_radial_edge", "optimizer_run_id": "200", "source": "execution-optimizer",
                 "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10,
                 "active_pipelines": 2, "shards": 2, "threads_per_pipeline": 1, "allocated_threads": 2,
                 "wall_clock_seconds": 10, "parameter_sets_per_second": 2.0, "execution_shape": "2p/1t",
                 "optimizer_shape_sequence": 1, "captured_at_utc": "2026-01-02T00:00:00Z",
                 "runner": {"runner_label": "e7k", "name": "host", "logical_cpus": 96}},
            ]}
            indexes = root / "indexes"
            indexes.mkdir()
            (indexes / "parallelism-index.json").write_text(json.dumps(parallelism), encoding="utf-8")
            self._write_completed_optimizer_summary(root, "adaptive_radial_edge", "200")

            paths = generate_optimizer_report_all(root, root / "out")
            text = paths["summary"].read_text(encoding="utf-8")
            self.assertIn("adaptive_radial_edge", text)


if __name__ == "__main__":
    unittest.main()
