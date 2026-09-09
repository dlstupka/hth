import json
import tempfile
import unittest
from pathlib import Path

from hth.domain.multidetector_schedule import optimize_lpt_schedule, plan_lpt_workers, preferred_short_schedule, recommended_schedule, workload_class


class MultiDetectorScheduleTests(unittest.TestCase):
    def test_large_384_thread_host(self):
        self.assertEqual(plan_lpt_workers(39, 384), 6)

    def test_workload_classes_keep_smoke_separate_from_full_exhaustive(self):
        self.assertEqual(workload_class("smoke", "exhaustive", "10"), "short")
        self.assertEqual(workload_class("full", "exhaustive", ""), "full-exhaustive")
        self.assertEqual(workload_class("full", "moderate+", ""), "short")

    def _index(self, root: Path, **overrides) -> Path:
        row = {
            "observation_id": "smoke-e9k-6p", "observed_at_utc": "2026-08-15T20:00:00Z",
            "workload_class": "short", "detector_count": 39, "golden_set_sha256": "gold",
            "runner_label": "384t", "runner_thread_budget": 384, "worker_count": 6,
            "makespan_seconds": 452.0, "worker_utilization": 0.82, "final_tail_seconds": 80.0,
        }
        row.update(overrides)
        path = root / "multidetector-index.json"
        path.write_text(json.dumps({"schema_version": 1, "observations": [row]}), encoding="utf-8")
        return path

    def test_preferred_short_keeps_measured_shape_on_same_large_host(self):
        with tempfile.TemporaryDirectory() as td:
            result = preferred_short_schedule(index_path=self._index(Path(td)), detector_count=39, runner_thread_budget=384, runner_label="384t", golden_set_sha256="gold")
            self.assertEqual(result["pipelines"], 6)
            self.assertEqual(result["threads_per_pipeline"], 64)

    def test_preferred_short_scales_threads_per_worker_across_large_hosts(self):
        with tempfile.TemporaryDirectory() as td:
            index = self._index(Path(td))
            small = preferred_short_schedule(index_path=index, detector_count=39, runner_thread_budget=192, runner_label="192t", golden_set_sha256="gold")
            large = preferred_short_schedule(index_path=index, detector_count=39, runner_thread_budget=768, runner_label="768t", golden_set_sha256="gold")
            self.assertEqual(small["pipelines"], 3)
            self.assertEqual(large["pipelines"], 12)

    def test_low_tail_high_utilization_explores_one_more_worker(self):
        with tempfile.TemporaryDirectory() as td:
            result = preferred_short_schedule(index_path=self._index(Path(td), worker_utilization=0.94, final_tail_seconds=40.0), detector_count=39, runner_thread_budget=384, runner_label="384t", golden_set_sha256="gold")
            self.assertEqual(result["pipelines"], 7)

    def test_recommended_schedule_uses_persisted_short_occupation_for_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            result = recommended_schedule(
                index_path=self._index(Path(td)), detector_count=39,
                runner_thread_budget=384, runner_label="384t", golden_set_sha256="gold",
                mode="smoke", strategy="exhaustive", limit="10",
            )
            self.assertEqual(result["source"], "multidetector-short-occupancy")
            self.assertEqual(result["pipelines"], 6)

    def test_recommended_schedule_uses_same_canonical_lpt_fallback_as_launcher(self):
        result = recommended_schedule(
            index_path=None, detector_count=39, runner_thread_budget=384,
            runner_label="384t", golden_set_sha256="gold",
            mode="full", strategy="exhaustive", limit="",
        )
        self.assertEqual(result["source"], "canonical-lpt-planner")
        self.assertEqual(result["pipelines"], plan_lpt_workers(39, 384))
        self.assertEqual(result["threads_per_pipeline"], 64)

    def test_long_tail_returns_one_workers_budget_to_long_work(self):
        with tempfile.TemporaryDirectory() as td:
            result = preferred_short_schedule(index_path=self._index(Path(td), worker_utilization=0.62, final_tail_seconds=180.0), detector_count=39, runner_thread_budget=384, runner_label="384t", golden_set_sha256="gold")
            self.assertEqual(result["pipelines"], 5)

    def test_optimizer_scores_every_feasible_lpt_worker_count(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime-index.json"
            rows = []
            for detector, seconds in (("slow", 100.0), ("medium", 40.0), ("fast", 10.0)):
                rows.append({
                    "detector_id": detector, "mode": "smoke", "resolved_strategy": "exhaustive",
                    "configured_threads": 8, "max_dimension": 1800, "golden_set_sha256": "gold",
                    "wall_clock_seconds": seconds, "observed_at_utc": "2026-09-09T00:00:00Z",
                    "runner": {"runner_labels": ["24t"]},
                    "build": {"github_run_id": "complete-build"},
                })
            path.write_text(json.dumps({"observations": rows}), encoding="utf-8")
            result = optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=["slow", "medium", "fast"],
                runner_thread_budget=24, runner_label="24t", golden_set_sha256="gold",
                mode="smoke", strategy="exhaustive", max_dimension=1800,
            )
            self.assertIsNotNone(result)
            self.assertEqual(result["source"], "runtime-index-lpt-optimizer")
            self.assertEqual(result["evidence_detector_count"], 3)
            self.assertGreaterEqual(result["pipelines"], 2)
            self.assertEqual(result["candidate_count"], 3)
            self.assertGreaterEqual(len(result["leading_candidates"]), 2)

    def test_optimizer_uses_latest_complete_cross_golden_build_when_exact_is_partial(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime-index.json"
            rows = []
            for detector, seconds in (("a", 10.0), ("b", 20.0), ("c", 30.0)):
                rows.append({
                    "detector_id": detector, "mode": "smoke", "resolved_strategy": "exhaustive",
                    "configured_threads": 8, "max_dimension": 1800, "golden_set_sha256": "other",
                    "wall_clock_seconds": seconds, "observed_at_utc": "2026-09-09T00:00:00Z",
                    "build": {"github_run_id": "complete-other"},
                })
            rows.append({
                "detector_id": "a", "mode": "smoke", "resolved_strategy": "exhaustive",
                "configured_threads": 8, "max_dimension": 1800, "golden_set_sha256": "target",
                "wall_clock_seconds": 10000.0, "observed_at_utc": "2026-09-10T00:00:00Z",
                "build": {"github_run_id": "partial-target"},
            })
            for detector in ("a", "b", "c"):
                rows.append({
                    "detector_id": detector, "mode": "smoke", "resolved_strategy": "exhaustive",
                    "configured_threads": 8, "max_dimension": 1800, "golden_set_sha256": "target",
                    "wall_clock_seconds": 50.0, "observed_at_utc": "2026-09-08T00:00:00Z",
                    "build": {"github_run_id": "older-complete-target"},
                })
            path.write_text(json.dumps({"observations": rows}), encoding="utf-8")
            result = optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=["a", "b", "c"], runner_thread_budget=24,
                runner_label="24t", golden_set_sha256="target", mode="smoke",
                strategy="exhaustive", max_dimension=1800,
            )
            self.assertEqual(result["evidence_build_id"], "complete-other")
            self.assertEqual(result["evidence_golden_set_relation"], "latest-compatible")


if __name__ == "__main__":
    unittest.main()
