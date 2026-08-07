from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hth.report_generator import calibration_run_dirs, generate_calibration_manifest, generate_optimizer_report


class ReportGeneratorTests(unittest.TestCase):
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
            paths = generate_optimizer_report(root, "adaptive_radial_edge", root / "out")
            text = paths["summary"].read_text(encoding="utf-8")
            self.assertIn("Optimizer run: **2**", text)
            self.assertIn("| 2 | 2 | 1 |", text)
            self.assertNotIn("| 1 | 1 | 1 |", text)


if __name__ == "__main__":
    unittest.main()
