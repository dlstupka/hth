import unittest

from hth.contracts import (
    adapt_calibration_index, adapt_parallelism_index,
    adapt_regression_summary, adapt_runtime_index,
)


class PersistenceContractTests(unittest.TestCase):
    def test_legacy_indexes_are_adapted_at_one_boundary(self):
        self.assertEqual(adapt_calibration_index({})["schema_version"], "legacy")
        self.assertEqual(adapt_runtime_index({})["observations"], [])
        self.assertEqual(adapt_parallelism_index({})["shard_observations"], [])

    def test_legacy_regression_metrics_are_normalized_by_summary_adapter(self):
        summary = {
            "winner": {"summary": {
                "page_count": 5, "success_count": 1, "failure_count": 4,
                "mean_iou": 0.9, "minimum_iou": 0.9, "stddev_iou": 0.0,
            }}
        }
        adapted = adapt_regression_summary(summary)
        stats = adapted["winner"]["summary"]
        self.assertAlmostEqual(stats["mean_iou"], 0.18)
        self.assertAlmostEqual(stats["mean_iou_success"], 0.9)
        self.assertEqual(stats["minimum_iou"], 0.0)


if __name__ == "__main__":
    unittest.main()
