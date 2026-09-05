import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "tools" / "run-detector-regressions.sh").read_text(encoding="utf-8")


class AutomaticShardTopologyContractTests(unittest.TestCase):
    def test_single_detector_auto_sharding_tracks_active_pipelines(self):
        self.assertIn('elif (( detector_count == 1 )); then', DRIVER)
        self.assertIn('planned_shards="$effective_pipelines"', DRIVER)
        self.assertIn('plan_source="auto-one-shard-per-pipeline"', DRIVER)
        self.assertIn('auto-one-shard-per-pipeline-capped-to-parameter-space', DRIVER)

    def test_manual_override_remains_shards_per_pipeline(self):
        self.assertIn('planned_shards=$((shard_pipeline_count * sharding_policy))', DRIVER)
        self.assertIn('explicit-${sharding_policy}-shards-per-pipeline', DRIVER)

    def test_unshardable_and_multidetector_work_keep_safe_exceptions(self):
        self.assertIn('"$effective_strategy" == "binary-refine" || "$effective_strategy" == "adaptive"', DRIVER)
        self.assertIn('plan_source="${effective_strategy}-single-shard"', DRIVER)
        self.assertIn('plan_source="multi-detector-single-shard"', DRIVER)


if __name__ == "__main__":
    unittest.main()
