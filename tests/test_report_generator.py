from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hth.report_generator import calibration_run_dirs, generate_calibration_manifest, generate_optimizer_report


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
                    {"detector_id": "a", "calibration_status": "authoritative", "record_path": "records/a-full", "created_at_utc": "2026-01-02"},
                    {"detector_id": "b", "calibration_status": "authoritative", "record_path": "records/b-full", "created_at_utc": "2026-01-03"},
                ]
            }
            (root / "calibration-index.json").write_text(json.dumps(index), encoding="utf-8")
            runs = calibration_run_dirs(root)
            self.assertEqual([path.name for path in runs], ["a-full", "b-full"])


    def test_calibration_manifest_reads_flattened_persisted_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            record = root / "records" / "a-full"
            record.mkdir(parents=True)
            (root / "calibration-index.json").write_text(json.dumps({"entries": [{
                "detector_id": "a", "calibration_status": "authoritative",
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

    def test_optimizer_report_regenerates_legacy_completed_report_from_latest_run_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "optimizer-index.json").write_text(json.dumps({"schema_version": 1, "detectors": {}}), encoding="utf-8")
            parallelism = {"observations": [
                {"detector_id": "adaptive_radial_edge", "optimizer_run_id": "100", "source": "execution-optimizer", "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10, "active_pipelines": 1, "shards": 1, "threads_per_pipeline": 192, "allocated_threads": 192, "wall_clock_seconds": 2700, "parameter_sets_per_second": 2.4, "execution_shape": "1p/1s/192t", "optimizer_shape_sequence": 1, "captured_at_utc": "2026-01-01T00:00:00Z", "runner": {"runner_label": "unknown", "runner_name": "host", "logical_cpu_count": 96}},
                {"detector_id": "adaptive_radial_edge", "optimizer_run_id": "200", "source": "execution-optimizer", "mode": "full", "strategy": "exhaustive", "actual_parameter_sets": 10, "possible_parameter_sets": 10, "active_pipelines": 2, "shards": 2, "threads_per_pipeline": 96, "allocated_threads": 192, "wall_clock_seconds": 1200, "parameter_sets_per_second": 5.5, "execution_shape": "2p/2s/96t", "optimizer_shape_sequence": 2, "captured_at_utc": "2026-01-02T00:00:00Z", "runner": {"runner_label": "e7k", "runner_name": "host", "logical_cpu_count": 96}},
            ]}
            (root / "parallelism-index.json").write_text(json.dumps(parallelism), encoding="utf-8")
            persisted = root / "execution-optimizer" / "adaptive_radial_edge"
            persisted.mkdir(parents=True)
            (persisted / "summary.md").write_text("legacy completed optimizer report\n", encoding="utf-8")
            (persisted / "heatmap.svg").write_text("<svg>legacy</svg>\n", encoding="utf-8")
            paths = generate_optimizer_report(root, "adaptive_radial_edge", root / "out")
            summary = paths["summary"].read_text(encoding="utf-8")
            profile = paths["profile"].read_text(encoding="utf-8")
            self.assertIn("Optimizer run: **200**", summary)
            self.assertIn("| **e7k — host (96 vCPU)** | 2 | 2 | 96 |", summary)
            self.assertNotIn("unknown", summary)
            self.assertIn("detector pipelines (log₂ scale)", profile)
            self.assertIn("parameter sets / second", profile)
            self.assertNotIn("<svg>legacy</svg>", profile)

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


if __name__ == "__main__":
    unittest.main()
