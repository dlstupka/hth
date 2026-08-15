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
                prediction_out=None, runner_budget=384,
            )
            env = workflow_shape_env(result)
            self.assertEqual(env["DETECTOR_PIPELINES"], 9)
            self.assertEqual(env["THREADS"], 42)
            self.assertEqual(env["SHARDS"], 9)
            self.assertEqual(env["HTH_EXACT_EXECUTION_SHAPE"], "1")

    def test_authoritative_full_provenance_gates_higher_scoring_smoke(self):
        records = [
            {"status": "authoritative", "search_type": "exhaustive", "created_at_utc": "2026-08-10T01:00:00Z", "mean_iou": 0.5},
            {"status": "provisional", "search_type": "smoke", "created_at_utc": "2026-08-11T01:00:00Z", "mean_iou": 0.99},
        ]
        selected = authoritative_record(records)
        self.assertEqual(selected["mean_iou"], 0.5)


if __name__ == "__main__":
    unittest.main()
