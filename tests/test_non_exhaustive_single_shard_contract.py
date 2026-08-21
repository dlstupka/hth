import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "tools" / "run-detector-regressions.sh").read_text(encoding="utf-8")


class NonExhaustiveSingleShardContractTests(unittest.TestCase):
    def test_auto_sharding_forces_non_exhaustive_detector_work_to_one_shard(self):
        self.assertIn('exhaustive_shardable=0', DRIVER)
        self.assertIn('[[ "$REGRESSION_MODE" == "full" ]]', DRIVER)
        self.assertIn('[[ -z "${effective_limit:-}" ]]', DRIVER)
        self.assertIn('[[ "$effective_strategy" == "exhaustive-with-zombies" ]]', DRIVER)
        self.assertIn('if (( exhaustive_shardable == 0 )); then', DRIVER)
        self.assertIn('planned_shards=1', DRIVER)
        self.assertIn('plan_source="non-exhaustive-single-shard"', DRIVER)

    def test_exact_shape_shard_fanout_is_limited_to_full_unlimited_exhaustive(self):
        non_exhaustive = DRIVER.index('if (( exhaustive_shardable == 0 )); then')
        exact = DRIVER.index('elif [[ "${HTH_EXACT_EXECUTION_SHAPE:-0}" == "1" ]]', non_exhaustive)
        self.assertLess(non_exhaustive, exact)
        self.assertIn('&& "${exhaustive_shardable:-0}" == "1"', DRIVER)


if __name__ == "__main__":
    unittest.main()
