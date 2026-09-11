import json
import tempfile
import unittest
from pathlib import Path

from hth.regression_shape import RunnerProfile, resolve_workflow_shape, workflow_shape_env


class MultiDetectorPreferredShapeTests(unittest.TestCase):
    def test_preferred_self_hosted_shape_exports_material_capacity_shards(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            configs = base / "configs"
            configs.mkdir()
            durations = [1980, 606, 769, 699, 1183, 671] + [60] * 41
            rows = []
            for index, seconds in enumerate(durations):
                detector = f"d{index:02d}"
                (configs / f"{detector}.json").write_text(
                    json.dumps({"detector": detector}), encoding="utf-8"
                )
                rows.append({
                    "detector_id": detector, "mode": "smoke", "resolved_strategy": "exhaustive",
                    "max_dimension": 1800, "wall_clock_seconds": seconds,
                    "observed_at_utc": "2026-09-10T00:00:00Z",
                    "runner": {"runner_labels": ["192t"]},
                    "build": {"github_run_id": "complete"},
                })
            runtime = base / "runtime-index.json"
            runtime.write_text(json.dumps({"observations": rows}), encoding="utf-8")
            golden = base / "golden.json"
            golden.write_text("{}", encoding="utf-8")
            profile = RunnerProfile(
                name="e9k", label="192t", cpu_model="x", physical_cores=192, logical_cpus=192
            )
            result = resolve_workflow_shape(
                shape_mode="preferred", regression_mode="smoke", strategy="exhaustive", limit="10",
                detector="all", manual_shape=None, parallelism_index=base / "parallelism.json",
                predictions_index=None, multidetector_index=None, runtime_index=runtime,
                detector_config_root=configs, golden_set=golden, max_dimension=1800,
                profile=profile, runner_budget=384,
                pre_resolved_pipelines=3, pre_resolved_threads=128,
                pre_resolved_source="preferred-dispatch",
            )
            self.assertEqual(result["pipelines"], 55)
            self.assertEqual(result["threads_per_pipeline"], 6)
            self.assertTrue(result["sharding_applied"])
            env = workflow_shape_env(result)
            counts = json.loads(env["HTH_DETECTOR_SHARD_COUNTS_JSON"])
            self.assertEqual([counts[f"d{i:02d}"] for i in range(6)], [4, 2, 2, 2, 2, 2])
            self.assertEqual(env["HTH_CAPACITY_SHARD_TARGET_SECONDS"], 600)

    def test_reset_bootstraps_at_max_pipeline_count(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            configs = base / "configs"
            configs.mkdir()
            for i in range(47):
                (configs / f"d{i}.json").write_text(json.dumps({"detector": f"d{i}"}), encoding="utf-8")
            golden = base / "golden.json"
            golden.write_text("{}", encoding="utf-8")
            profile = RunnerProfile(name="e9k", label="192t", cpu_model="x", physical_cores=192, logical_cpus=192)
            result = resolve_workflow_shape(
                shape_mode="reset", regression_mode="smoke", strategy="exhaustive", limit="10", detector="all",
                manual_shape=None, parallelism_index=base / "parallelism.json", predictions_index=None,
                multidetector_index=None, runtime_index=None, detector_config_root=configs,
                golden_set=golden, max_dimension=1800, profile=profile, runner_budget=384,
            )
            self.assertEqual(result["pipelines"], 47)
            self.assertEqual(result["threads_per_pipeline"], 8)
            self.assertEqual(result["source"], "reset-bootstrap-max-pipelines")

    def test_preferred_short_all_uses_occupancy_history_without_forcing_shards(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            configs = base / "configs"
            configs.mkdir()
            for i in range(39):
                (configs / f"d{i}.json").write_text(json.dumps({"detector": f"d{i}"}), encoding="utf-8")
            golden = base / "golden.json"
            golden.write_text("{}", encoding="utf-8")
            import hashlib
            gold = hashlib.sha256(golden.read_bytes()).hexdigest()
            index = base / "multidetector-index.json"
            index.write_text(json.dumps({"schema_version": 1, "observations": [{
                "observation_id": "obs", "observed_at_utc": "2026-08-15T20:00:00Z", "workload_class": "short",
                "detector_count": 39, "golden_set_sha256": gold, "runner_label": "384t",
                "runner_thread_budget": 384, "worker_count": 6, "makespan_seconds": 452.0,
                "worker_utilization": 0.82, "final_tail_seconds": 80.0,
            }]}), encoding="utf-8")
            profile = RunnerProfile(name="e9k", label="384t", cpu_model="x", physical_cores=192, logical_cpus=192)
            result = resolve_workflow_shape(
                shape_mode="preferred", regression_mode="smoke", strategy="exhaustive", limit="10", detector="all",
                manual_shape=None, parallelism_index=base/"parallelism.json", predictions_index=None,
                multidetector_index=index, detector_config_root=configs, golden_set=golden, max_dimension=1800,
                profile=profile, runner_budget=384,
            )
            self.assertTrue(result["exact"])
            self.assertTrue(result["multidetector"])
            self.assertEqual(result["pipelines"], 39)
            env = workflow_shape_env(result)
            self.assertEqual(env["DETECTOR_PIPELINES"], 39)
            self.assertEqual(env["THREADS"], 9)
            self.assertNotIn("SHARDS", env)
            self.assertNotIn("SHARDING", env)

    def test_full_adaptive_all_jointly_optimizes_pipeline_count_and_lpt(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            configs = base / "configs"
            configs.mkdir()
            rows = []
            for detector, seconds in (("slow", 100.0), ("medium", 40.0), ("fast", 10.0)):
                (configs / f"{detector}.json").write_text(json.dumps({"detector": detector}), encoding="utf-8")
                rows.append({
                    "detector_id": detector, "mode": "full", "resolved_strategy": "adaptive",
                    "configured_threads": 8, "max_dimension": 1800,
                    "wall_clock_seconds": seconds, "observed_at_utc": "2026-09-09T00:00:00Z",
                    "runner": {"runner_labels": ["24t"]},
                    "build": {"github_run_id": "complete-build"},
                })
            runtime = base / "runtime-index.json"
            runtime.write_text(json.dumps({"observations": rows}), encoding="utf-8")
            golden = base / "golden.json"
            golden.write_text("{}", encoding="utf-8")
            profile = RunnerProfile(name="runner", label="24t", cpu_model="x", physical_cores=12, logical_cpus=24)
            result = resolve_workflow_shape(
                shape_mode="preferred", regression_mode="full", strategy="adaptive", limit="", detector="all",
                manual_shape=None, parallelism_index=base / "parallelism.json", predictions_index=None,
                multidetector_index=None, runtime_index=runtime, detector_config_root=configs,
                golden_set=golden, max_dimension=1800, profile=profile, runner_budget=24,
            )
            self.assertTrue(result["exact"])
            self.assertEqual(result["source"], "preferred-runtime-index-lpt-optimizer")
            self.assertEqual(result["evidence_detector_count"], 3)
            self.assertGreaterEqual(result["pipelines"], 2)


if __name__ == "__main__":
    unittest.main()
