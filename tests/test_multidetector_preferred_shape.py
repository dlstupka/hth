import json
import tempfile
import unittest
from pathlib import Path

from hth.regression_shape import RunnerProfile, resolve_workflow_shape, workflow_shape_env


class MultiDetectorPreferredShapeTests(unittest.TestCase):
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
            self.assertEqual(result["pipelines"], 6)
            env = workflow_shape_env(result)
            self.assertEqual(env["DETECTOR_PIPELINES"], 6)
            self.assertEqual(env["THREADS"], 64)
            self.assertNotIn("SHARDS", env)
            self.assertNotIn("SHARDING", env)


if __name__ == "__main__":
    unittest.main()
