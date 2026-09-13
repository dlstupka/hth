import json
import tempfile
import unittest
from pathlib import Path

from hth.domain.multidetector_schedule import (
    materially_improves_makespan,
    normalize_pipeline_assignments,
    optimize_lpt_schedule,
    plan_capacity_shards,
    plan_golden_set_lanes,
    plan_lpt_workers,
    plan_static_lpt_tasks,
    recommended_schedule,
    workload_class,
)


class MultiDetectorScheduleTests(unittest.TestCase):
    def test_assignment_normalization_rejects_partial_duplicate_and_malformed_maps(self):
        self.assertEqual(normalize_pipeline_assignments(["a", "b"], {"a": "1", "b": 2}, 2), {"a": 1, "b": 2})
        self.assertIsNone(normalize_pipeline_assignments(["a", "b"], {"a": 1}, 2))
        self.assertIsNone(normalize_pipeline_assignments(["a", "a"], {"a": 1}, 2))
        self.assertIsNone(normalize_pipeline_assignments(["a"], {"a": "nonsense"}, 2))
        self.assertIsNone(normalize_pipeline_assignments(["a"], {"a": 3}, 2))
        self.assertIsNone(normalize_pipeline_assignments(["a"], {"a": 1}, 0))

    def test_makespan_gate_accepts_exact_threshold_and_enforces_high_water(self):
        self.assertTrue(materially_improves_makespan(100.0, 80.0, high_water_seconds=80.0))
        self.assertFalse(materially_improves_makespan(100.0, 80.0001, high_water_seconds=100.0))
        self.assertFalse(materially_improves_makespan(200.0, 100.0, high_water_seconds=99.0))
        self.assertFalse(materially_improves_makespan(200.0, 100.0, high_water_seconds=float("nan")))
        with self.assertRaisesRegex(ValueError, "minimum_improvement"):
            materially_improves_makespan(100.0, 50.0, minimum_improvement=float("nan"))

    def test_capacity_sharding_never_splits_unknown_runtime(self):
        plan = plan_capacity_shards([1200.0, None], 8)
        self.assertEqual(plan["shard_counts"][1], 1)

    def test_capacity_sharding_without_measurements_preserves_unsharded_capacity(self):
        plan = plan_capacity_shards([None, None, None], 2)
        self.assertFalse(plan["applied"])
        self.assertEqual(plan["pipelines"], 2)
        self.assertEqual(plan["task_count"], 3)

    def test_static_lpt_caps_idle_pipeline_allocation_to_runnable_tasks(self):
        schedule = plan_static_lpt_tasks([10.0], 1_000_000)
        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule[0]["pipeline"], 1)

    def test_static_lpt_rejects_invalid_estimate_floor(self):
        with self.assertRaisesRegex(ValueError, "estimate_floor_seconds"):
            plan_static_lpt_tasks([10.0], 1, estimate_floor_seconds=float("nan"))

    def test_capacity_sharding_caps_fanout_by_runnable_candidates(self):
        plan = plan_capacity_shards([2400.0, 60.0], 8, maximum_shards=[2, 1])
        self.assertTrue(plan["applied"])
        self.assertEqual(plan["shard_counts"], [2, 1])

    def test_capacity_sharding_does_not_materialize_more_shards_than_runner_capacity(self):
        plan = plan_capacity_shards([10**12], 8, maximum_shards=[10**9])
        self.assertTrue(plan["applied"])
        self.assertEqual(plan["shard_counts"], [8])
        self.assertEqual(plan["task_count"], 8)

    def test_capacity_sharding_rejects_misaligned_candidate_caps(self):
        with self.assertRaisesRegex(ValueError, "align"):
            plan_capacity_shards([1200.0], 8, maximum_shards=[2, 1])
        with self.assertRaisesRegex(ValueError, r"maximum_shards\[0\]"):
            plan_capacity_shards([1200.0], 8, maximum_shards=["bad"])
        with self.assertRaisesRegex(ValueError, "target_shard_seconds"):
            plan_capacity_shards([1200.0], 8, target_shard_seconds=float("nan"))

    def test_large_384_thread_host(self):
        self.assertEqual(plan_lpt_workers(39, 384), 39)

    def test_capacity_sharding_splits_long_detectors_to_ten_minute_jobs(self):
        estimates = [1980, 606, 769, 699, 1183, 671] + [60] * 41
        plan = plan_capacity_shards(estimates, 192)
        self.assertTrue(plan["applied"])
        self.assertEqual(plan["shard_counts"][:6], [4, 2, 2, 2, 2, 2])
        self.assertEqual(plan["task_count"], 55)
        self.assertEqual(plan["pipelines"], 55)
        self.assertLessEqual(plan["predicted_makespan_seconds"], 600)

    def test_shard_target_boundary_does_not_split_exact_ten_minute_job(self):
        plan = plan_capacity_shards([600.0, 60.0], 8)
        self.assertFalse(plan["applied"])
        self.assertEqual(plan["shard_counts"], [1, 1])

    def test_capacity_shards_do_not_divide_fixed_preparation(self):
        plan = plan_capacity_shards(
            [2400.0, 60.0], 8,
            fixed_preparation_seconds=[2300.0, 0.0],
            shardable_work_seconds=[100.0, 60.0],
        )
        self.assertFalse(plan["applied"])
        self.assertEqual(plan["shard_counts"], [1, 1])
        self.assertEqual(plan["reason"], "no-eligible-work-above-target")

    def test_capacity_shards_explain_when_fixed_preparation_makes_candidate_slower(self):
        plan = plan_capacity_shards(
            [2006.0, 1179.0], 192,
            fixed_preparation_seconds=[1326.0, 683.0],
            shardable_work_seconds=[680.0, 496.0],
        )
        self.assertFalse(plan["applied"])
        self.assertEqual(plan["shard_counts"], [1, 1])
        self.assertEqual(plan["candidate_shard_counts"], [2, 1])
        self.assertEqual(plan["reason"], "candidate-increases-makespan")
        self.assertEqual(plan["candidate_task_count"], 3)
        self.assertAlmostEqual(plan["candidate_makespan_seconds"], 2505.0)
        self.assertAlmostEqual(plan["candidate_shared_preparation_seconds"], 1326.0)

    def test_capacity_shards_include_shared_preparation_in_makespan(self):
        plan = plan_capacity_shards(
            [2400.0, 60.0], 8,
            fixed_preparation_seconds=[300.0, 0.0],
            shardable_work_seconds=[2100.0, 60.0],
        )
        self.assertTrue(plan["applied"])
        self.assertEqual(plan["shard_counts"][0], 4)
        self.assertEqual(plan["shared_preparation_seconds"], 300.0)
        self.assertAlmostEqual(plan["predicted_makespan_seconds"], 825.0)

    def test_existing_parent_shared_preparation_is_outside_fanout(self):
        plan = plan_capacity_shards(
            [1000.0, 100.0], 8,
            fixed_preparation_seconds=[800.0, 0.0],
            shardable_work_seconds=[200.0, 100.0],
            pre_fanout_preparation=[True, False],
        )
        self.assertFalse(plan["applied"])
        self.assertEqual(plan["unsharded_makespan_seconds"], 1000.0)

    def test_golden_set_lanes_split_only_measured_page_bounded_work(self):
        plan = plan_golden_set_lanes(
            [2400.0, 720.0, 60.0], 8, maximum_lanes=[18, 18, 18],
        )
        self.assertTrue(plan["applied"])
        self.assertEqual(plan["lane_counts"], [4, 2, 1])
        self.assertEqual(plan["pipelines"], 7)
        self.assertLessEqual(plan["predicted_makespan_seconds"], 600.0)

    def test_golden_set_lanes_require_pages_spare_capacity_and_twenty_percent_gain(self):
        self.assertFalse(plan_golden_set_lanes(
            [2400.0, 60.0], 8, maximum_lanes=[None, 18],
        )["applied"])
        self.assertFalse(plan_golden_set_lanes(
            [600.0, 60.0], 8, maximum_lanes=[18, 18],
        )["applied"])
        self.assertFalse(plan_golden_set_lanes(
            [2400.0, 60.0], 2, maximum_lanes=[18, 18],
        )["applied"])

    def test_golden_set_lanes_do_not_divide_fixed_preparation(self):
        plan = plan_golden_set_lanes(
            [2400.0, 60.0], 8, maximum_lanes=[18, 18],
            fixed_preparation_seconds=[2300.0, 0.0],
            lane_work_seconds=[100.0, 60.0],
        )
        self.assertFalse(plan["applied"])
        self.assertEqual(plan["lane_counts"], [1, 1])

    def test_golden_set_lanes_keep_parent_shared_preparation_before_fanout(self):
        plan = plan_golden_set_lanes(
            [2400.0, 60.0], 8, maximum_lanes=[18, 18],
            fixed_preparation_seconds=[300.0, 0.0],
            lane_work_seconds=[2100.0, 60.0],
            pre_fanout_preparation=[True, False],
        )
        self.assertTrue(plan["applied"])
        self.assertEqual(plan["lane_counts"], [4, 1])
        self.assertEqual(plan["shared_preparation_seconds"], 300.0)
        self.assertEqual(plan["predicted_makespan_seconds"], 825.0)

    def test_persisted_pre_decomposition_index_requires_reset(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime-index.json"
            path.write_text(json.dumps({"schema_version": "1.0", "observations": [{
                "schema_version": "1.0", "detector_id": "slow", "mode": "full",
                "resolved_strategy": "adaptive", "wall_clock_seconds": 600.0,
                "observed_at_utc": "2026-09-12T00:00:00Z",
                "build": {"github_run_id": "old-model"},
            }]}), encoding="utf-8")
            self.assertIsNone(optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=["slow"], runner_thread_budget=384,
                runner_label="192t", golden_set_sha256=None, mode="full",
                strategy="adaptive", max_dimension=1800,
            ))

    def test_github_sized_optimizer_does_not_change_existing_topology(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime-index.json"
            path.write_text(json.dumps({"observations": [
                {
                    "detector_id": detector, "mode": "smoke", "resolved_strategy": "exhaustive",
                    "wall_clock_seconds": seconds, "observed_at_utc": "2026-09-09T00:00:00Z",
                    "build": {"github_run_id": "complete"},
                }
                for detector, seconds in (("a", 1980), ("b", 769), ("c", 699), ("d", 60))
            ]}), encoding="utf-8")
            result = optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=["a", "b", "c", "d"],
                runner_thread_budget=8, runner_label="github-hosted", golden_set_sha256=None,
                mode="smoke", strategy="exhaustive", max_dimension=1800,
            )
            self.assertFalse(result["sharding_applied"])
            self.assertEqual(result["detector_shard_counts"], {"a": 1, "b": 1, "c": 1, "d": 1})
            self.assertLessEqual(result["pipelines"], 4)

    def test_optimizer_does_not_contract_incumbent_pipeline_count_on_floor_tie(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime-index.json"
            costs = [("slow", 40.0)] + [(letter, 10.0) for letter in "abcdef"]
            assignments = {"slow": 1, "a": 1, "b": 1, "c": 2, "d": 2, "e": 3, "f": 4}
            path.write_text(json.dumps({"observations": [
                {
                    "detector_id": detector, "mode": "smoke",
                    "resolved_strategy": "exhaustive", "wall_clock_seconds": seconds,
                    "detector_pipeline_number": assignments[detector],
                    "detector_pipelines": 4,
                    "observed_at_utc": "2026-09-12T00:00:00Z",
                    "build": {"github_run_id": "complete"},
                }
                for detector, seconds in costs
            ]}), encoding="utf-8")

            result = optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=[row[0] for row in costs],
                runner_thread_budget=8, runner_label="github-hosted",
                golden_set_sha256=None, mode="smoke", strategy="exhaustive",
                max_dimension=1800,
            )

            self.assertEqual(result["pipelines"], 4)
            self.assertEqual(result["predicted_makespan_seconds"], 40.0)

    def test_github_hosted_adaptive_never_enables_golden_set_lane_scaling(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime-index.json"
            path.write_text(json.dumps({"observations": [
                {
                    "detector_id": detector, "mode": "full", "resolved_strategy": "adaptive",
                    "wall_clock_seconds": seconds, "golden_set_pages": 18,
                    "observed_at_utc": "2026-09-11T00:00:00Z",
                    "build": {"github_run_id": "complete"},
                }
                for detector, seconds in (("a", 2400), ("b", 60))
            ]}), encoding="utf-8")
            result = optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=["a", "b"],
                runner_thread_budget=32, runner_label="github-hosted",
                golden_set_sha256=None, mode="full", strategy="adaptive",
                max_dimension=1800,
            )
            self.assertFalse(result["golden_set_lane_scaling_applied"])
            self.assertEqual(result["detector_golden_set_lane_counts"], {"a": 1, "b": 1})

    def test_self_hosted_adaptive_uses_serial_equivalent_history_for_stable_lanes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime-index.json"
            path.write_text(json.dumps({"observations": [
                {
                    "detector_id": detector, "mode": "full", "resolved_strategy": "adaptive",
                    "wall_clock_seconds": wall, "estimated_serial_runtime_seconds": serial,
                    "golden_set_pages": 18, "observed_at_utc": "2026-09-12T00:00:00Z",
                    "runner": {"runner_labels": ["192t"]},
                    "build": {"github_run_id": "coordinated"},
                }
                for detector, wall, serial in (("slow", 600, 2400), ("fast", 60, 60))
            ]}), encoding="utf-8")
            result = optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=["slow", "fast"],
                runner_thread_budget=16, runner_label="192t",
                golden_set_sha256=None, mode="full", strategy="adaptive",
                max_dimension=1800,
            )
            self.assertTrue(result["golden_set_lane_scaling_applied"])
            self.assertEqual(result["detector_golden_set_lane_counts"], {"slow": 4, "fast": 1})
            self.assertEqual(result["pipelines"], 5)
            self.assertEqual(result["predicted_makespan_seconds"], 600.0)
            self.assertEqual(result["detector_fanout_estimates"]["slow"], 600.0)

    def test_merged_shard_serial_work_prevents_next_run_whipsaw(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime-index.json"
            path.write_text(json.dumps({"observations": [{
                "detector_id": "slow", "mode": "smoke", "resolved_strategy": "exhaustive",
                "wall_clock_seconds": 495.0, "estimated_serial_runtime_seconds": 1980.0,
                "observed_at_utc": "2026-09-10T00:00:00Z",
                "build": {"github_run_id": "sharded"},
            }]}), encoding="utf-8")
            result = optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=["slow"], runner_thread_budget=384,
                runner_label="192t", golden_set_sha256=None, mode="smoke",
                strategy="exhaustive", max_dimension=1800,
            )
            self.assertTrue(result["sharding_applied"])
            self.assertEqual(result["detector_shard_counts"], {"slow": 4})
            self.assertEqual(result["detector_fanout_estimates"], {"slow": 1980.0})

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

    def test_multidetector_summary_without_runtime_rows_uses_bootstrap(self):
        with tempfile.TemporaryDirectory() as td:
            result = recommended_schedule(
                index_path=self._index(Path(td)), detector_count=39,
                runner_thread_budget=384, runner_label="384t", golden_set_sha256="gold",
                mode="smoke", strategy="exhaustive", limit="10",
            )
            self.assertEqual(result["source"], "canonical-lpt-planner")
            self.assertEqual(result["pipelines"], 39)

    def test_recommended_schedule_uses_same_canonical_lpt_fallback_as_launcher(self):
        result = recommended_schedule(
            index_path=None, detector_count=39, runner_thread_budget=384,
            runner_label="384t", golden_set_sha256="gold",
            mode="full", strategy="exhaustive", limit="",
        )
        self.assertEqual(result["source"], "canonical-lpt-planner")
        self.assertEqual(result["pipelines"], plan_lpt_workers(39, 384))
        self.assertEqual(result["threads_per_pipeline"], 9)

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

    def test_optimizer_rejects_incomplete_timing_build_instead_of_fabricating_cost(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime-index.json"
            path.write_text(json.dumps({"observations": [
                {
                    "detector_id": detector, "mode": "smoke", "resolved_strategy": "exhaustive",
                    "wall_clock_seconds": seconds, "observed_at_utc": "2026-09-09T00:00:00Z",
                    "build": {"github_run_id": "complete"},
                }
                for detector, seconds in (("a", 100.0), ("b", None))
            ]}), encoding="utf-8")
            result = optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=["a", "b"], runner_thread_budget=16,
                runner_label="8t", golden_set_sha256=None, mode="smoke",
                strategy="exhaustive", max_dimension=1800,
            )
            self.assertIsNone(result)

    def test_optimizer_treats_corrupt_index_and_duplicate_detector_ids_as_no_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime-index.json"
            path.write_text("not json", encoding="utf-8")
            self.assertIsNone(optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=["a"], runner_thread_budget=8,
                runner_label="8t", golden_set_sha256=None, mode="smoke",
                strategy="exhaustive", max_dimension=1800,
            ))
            path.write_text(json.dumps({"observations": [{
                "detector_id": "a", "mode": "smoke", "resolved_strategy": "exhaustive",
                "wall_clock_seconds": 10.0, "observed_at_utc": "2026-09-09T00:00:00Z",
                "build": {"github_run_id": "complete"},
            }]}), encoding="utf-8")
            self.assertIsNone(optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=["a", "a"], runner_thread_budget=8,
                runner_label="8t", golden_set_sha256=None, mode="smoke",
                strategy="exhaustive", max_dimension=1800,
            ))

    def test_optimizer_never_exceeds_longest_job_high_water_mark(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime-index.json"
            path.write_text(json.dumps({"observations": [
                {
                    "detector_id": detector, "mode": "smoke", "resolved_strategy": "exhaustive",
                    "wall_clock_seconds": seconds, "observed_at_utc": "2026-09-09T00:00:00Z",
                    "build": {"github_run_id": "complete"},
                }
                for detector, seconds in (("a", 100.0), ("b", 60.0), ("c", 55.0), ("d", 5.0))
            ]}), encoding="utf-8")
            result = optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=["a", "b", "c", "d"],
                runner_thread_budget=16, runner_label="8t", golden_set_sha256=None,
                mode="smoke", strategy="exhaustive", max_dimension=1800,
            )
            self.assertEqual(result["pipelines"], 3)
            self.assertEqual(result["predicted_makespan_seconds"], 100.0)
            self.assertEqual(result["longest_detector_floor_seconds"], 100.0)

    def test_optimizer_retains_incumbent_without_twenty_percent_gain(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime-index.json"
            durations = [100.0] + [10.0] * 7
            path.write_text(json.dumps({"observations": [
                {
                    "detector_id": f"d{index}", "mode": "full", "resolved_strategy": "adaptive",
                    "wall_clock_seconds": seconds, "observed_at_utc": "2026-09-10T00:00:00Z",
                    "detector_pipelines": 8, "detector_pipeline_number": index + 1,
                    "build": {"github_run_id": "complete"},
                }
                for index, seconds in enumerate(durations)
            ]}), encoding="utf-8")
            result = optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=[f"d{i}" for i in range(8)],
                runner_thread_budget=384, runner_label="192t", golden_set_sha256=None,
                mode="full", strategy="adaptive", max_dimension=1800,
            )
            self.assertEqual(result["pipelines"], 8)
            self.assertEqual(result["predicted_makespan_seconds"], 100.0)
            self.assertTrue(result["schedule_retained"])
            self.assertEqual(result["detector_pipeline_assignments"]["d0"], 1)

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
            completed = Path(td) / "multidetector-index.json"
            completed.write_text(json.dumps({"observations": [{
                "github_run_id": "complete-other", "mode": "smoke", "strategy": "exhaustive",
                "detector_count": 3, "task_count": 3,
            }]}), encoding="utf-8")
            result = optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=["a", "b", "c"], runner_thread_budget=24,
                runner_label="24t", golden_set_sha256="target", mode="smoke",
                strategy="exhaustive", max_dimension=1800, completion_index_path=completed,
            )
            self.assertEqual(result["evidence_build_id"], "complete-other")
            self.assertEqual(result["evidence_golden_set_relation"], "latest-compatible")

    def test_optimizer_prefers_valid_complete_exact_golden_build(self):
        with tempfile.TemporaryDirectory() as td:
            runtime = Path(td) / "runtime-index.json"
            rows = []
            for build_id, golden, observed in (
                ("exact", "target", "2026-09-08T00:00:00Z"),
                ("newer-other", "other", "2026-09-09T00:00:00Z"),
            ):
                for detector in ("a", "b"):
                    rows.append({
                        "detector_id": detector, "mode": "smoke", "resolved_strategy": "exhaustive",
                        "configured_threads": 8, "max_dimension": 1800, "golden_set_sha256": golden,
                        "wall_clock_seconds": 10.0, "observed_at_utc": observed,
                        "build": {"github_run_id": build_id},
                    })
            runtime.write_text(json.dumps({"observations": rows}), encoding="utf-8")
            completed = Path(td) / "multidetector-index.json"
            completed.write_text(json.dumps({"observations": [
                {"github_run_id": build_id, "mode": "smoke", "strategy": "exhaustive", "detector_count": 2, "task_count": 2}
                for build_id in ("exact", "newer-other")
            ]}), encoding="utf-8")
            result = optimize_lpt_schedule(
                runtime_index_path=runtime, completion_index_path=completed,
                detector_ids=["a", "b"], runner_thread_budget=16, runner_label="16t",
                golden_set_sha256="target", mode="smoke", strategy="exhaustive", max_dimension=1800,
            )
            self.assertEqual(result["evidence_build_id"], "exact")
            self.assertEqual(result["evidence_golden_set_relation"], "exact")

    def test_optimizer_prefers_matching_dimension_and_runner_within_golden_set(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "runtime-index.json"
            rows = []
            for build_id, dimension, label, observed, seconds in (
                ("matching", 1800, "192t", "2026-09-08T00:00:00Z", 100.0),
                ("newer-wrong-context", 900, "github-hosted", "2026-09-09T00:00:00Z", 10.0),
            ):
                for detector in ("a", "b"):
                    rows.append({
                        "detector_id": detector, "mode": "smoke", "resolved_strategy": "exhaustive",
                        "max_dimension": dimension, "golden_set_sha256": "gold",
                        "wall_clock_seconds": seconds, "observed_at_utc": observed,
                        "runner": {"runner_labels": [label]},
                        "build": {"github_run_id": build_id},
                    })
            path.write_text(json.dumps({"observations": rows}), encoding="utf-8")
            result = optimize_lpt_schedule(
                runtime_index_path=path, detector_ids=["a", "b"], runner_thread_budget=384,
                runner_label="192t", golden_set_sha256="gold", mode="smoke",
                strategy="exhaustive", max_dimension=1800,
            )
            self.assertEqual(result["evidence_build_id"], "matching")
            self.assertEqual(result["evidence_dimension_relation"], "exact")
            self.assertEqual(result["evidence_runner_relation"], "exact")

    def test_smoke_limit_does_not_cap_execution_threads(self):
        with tempfile.TemporaryDirectory() as td:
            runtime = Path(td) / "runtime-index.json"
            runtime.write_text(json.dumps({"observations": [
                {
                    "detector_id": detector, "mode": "smoke", "resolved_strategy": "exhaustive",
                    "configured_threads": 192, "actual_parameter_sets": 10,
                    "wall_clock_seconds": seconds, "observed_at_utc": "2026-09-09T00:00:00Z",
                    "build": {"github_run_id": "complete"},
                }
                for detector, seconds in (("a", 1200.0), ("b", 300.0), ("c", 100.0))
            ]}), encoding="utf-8")
            result = optimize_lpt_schedule(
                runtime_index_path=runtime, detector_ids=["a", "b", "c"],
                runner_thread_budget=384, runner_label="192t", golden_set_sha256=None,
                mode="smoke", strategy="exhaustive", max_dimension=1800,
            )
            self.assertEqual(result["pipelines"], 4)
            self.assertEqual(result["threads_per_pipeline"], 96)
            self.assertEqual(result["detector_shard_counts"], {"a": 2, "b": 1, "c": 1})


if __name__ == "__main__":
    unittest.main()
