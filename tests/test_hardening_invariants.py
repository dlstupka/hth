import json
import tempfile
import unittest
from pathlib import Path

from hth.domain.calibration import authoritative_record
from hth.domain.execution_shape import select_preferred_shape
from hth.domain.result_metrics import aggregate_page_metrics, calibration_metric_view
from hth.regression_shape import RunnerProfile, resolve_workflow_shape, workflow_shape_env


class HardeningInvariantTests(unittest.TestCase):
    def test_one_success_four_failures_is_canonical_everywhere(self):
        pages = [{"status": "ok", "iou": 0.9}] + [{"status": "no_candidate", "iou": 0.0} for _ in range(4)]
        metrics = aggregate_page_metrics(pages)
        self.assertAlmostEqual(metrics["mean_iou"], 0.18)
        self.assertAlmostEqual(metrics["mean_iou_success"], 0.9)
        self.assertEqual(metrics["failure_count"], 4)

        payload = {"detector_selection_intelligence": {"best_avg_iou": 0.9, "failure_count": 0}}
        summary = {"winner": {"summary": metrics}}
        view = calibration_metric_view(payload, summary)
        self.assertAlmostEqual(view["mean_iou"], 0.18)
        self.assertAlmostEqual(view["mean_iou_success"], 0.9)
        self.assertEqual(view["failure_count"], 4)

    def test_preferred_shape_tie_break_is_canonical(self):
        shapes = [
            {"pipelines": 2, "threads_per_pipeline": 192, "allocated_threads": 384, "parameter_sets_per_second": 364.50},
            {"pipelines": 7, "threads_per_pipeline": 54, "allocated_threads": 378, "parameter_sets_per_second": 364.50},
            {"pipelines": 8, "threads_per_pipeline": 48, "allocated_threads": 384, "parameter_sets_per_second": 364.50},
        ]
        best = select_preferred_shape(shapes)
        self.assertEqual((best["pipelines"], best["threads_per_pipeline"]), (7, 54))

    def test_resolved_shape_is_exactly_the_shape_exported_to_manifest_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            # Manual resolution exercises the same exact-shape environment contract
            # used after preferred/predicted selection.
            result = resolve_workflow_shape(
                shape_mode="manual", regression_mode="full", strategy="exhaustive",
                limit="", detector="gradient_vote", manual_shape="9p/42t",
                parallelism_index=root/"parallelism-index.json",
                predictions_index=root/"optimizer-predictions.json",
                detector_config_root=root, golden_set=root/"golden.json",
                max_dimension=1800,
                profile=RunnerProfile("runner", "192t", "cpu", 192, 192),
                runner_budget=384,
            )
            env = workflow_shape_env(result)
            self.assertEqual(env["DETECTOR_PIPELINES"], 9)
            self.assertEqual(env["THREADS"], 42)
            self.assertNotIn("SHARDS", env)
            self.assertEqual(env["HTH_EXACT_EXECUTION_SHAPE"], "1")

    def test_preferred_shape_scales_one_collected_shape_linearly_across_vcpu_sizes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            detector_root = root / "detectors"
            detector_root.mkdir()
            detector = detector_root / "example.json"
            golden = root / "golden.json"
            detector.write_text(json.dumps({"detector": "example"}), encoding="utf-8")
            golden.write_text(json.dumps({"pages": []}), encoding="utf-8")
            detector_sha = __import__("hashlib").sha256(detector.read_bytes()).hexdigest()
            golden_sha = __import__("hashlib").sha256(golden.read_bytes()).hexdigest()
            index = root / "parallelism-index.json"
            index.write_text(json.dumps({"observations": [{
                "source": "execution-optimizer",
                "detector_id": "example",
                "mode": "full",
                "strategy": "exhaustive",
                "detector_config_sha256": detector_sha,
                "golden_set_sha256": golden_sha,
                "possible_parameter_sets": 100,
                "actual_parameter_sets": 100,
                "max_dimension": 1800,
                "wall_clock_seconds": 10.0,
                "parameter_sets_per_second": 10.0,
                "active_pipelines": 32,
                "shards": 32,
                "threads_per_pipeline": 12,
                "allocated_threads": 384,
                "runner": {
                    "runner_name": "source-192",
                    "runner_label": "192t",
                    "cpu_model": "source cpu",
                    "physical_core_count": 192,
                    "logical_cpu_count": 192,
                },
            }]}), encoding="utf-8")
            result = resolve_workflow_shape(
                shape_mode="preferred", regression_mode="full", strategy="exhaustive",
                limit="", detector="example", manual_shape="",
                parallelism_index=index, predictions_index=root/"optimizer-predictions.json",
                detector_config_root=detector_root, golden_set=golden, max_dimension=1800,
                profile=RunnerProfile("target-32", "32t", "different cpu", 32, 32),
                runner_budget=64,
            )
            self.assertTrue(result["exact"])
            self.assertEqual(result["pipelines"], 5)
            self.assertEqual(result["threads_per_pipeline"], 12)
            self.assertEqual(result["source"], "predicted-low-linear-vcpu")

    def test_authoritative_full_provenance_gates_higher_scoring_smoke(self):
        records = [
            {"status": "authoritative", "search_type": "exhaustive", "created_at_utc": "2026-08-10T01:00:00Z", "mean_iou": 0.5},
            {"status": "provisional", "search_type": "smoke", "created_at_utc": "2026-08-11T01:00:00Z", "mean_iou": 0.99},
        ]
        selected = authoritative_record(records)
        self.assertEqual(selected["mean_iou"], 0.5)


if __name__ == "__main__":
    unittest.main()
