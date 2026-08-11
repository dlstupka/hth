import unittest

from hth.regression.result_metrics import (
    aggregate_page_metrics,
    normalize_result_metrics,
    normalize_summary_metrics,
)


class ResultMetricsTests(unittest.TestCase):
    def test_live_and_legacy_metrics_match_for_failed_pages(self):
        pages = [
            {"status": "ok", "iou": 0.9638},
            {"status": "no_candidate", "iou": 0.0},
            {"status": "no_candidate", "iou": 0.0},
            {"status": "no_candidate", "iou": 0.0},
            {"status": "no_candidate", "iou": 0.0},
        ]
        live = aggregate_page_metrics(pages)
        legacy = {
            "page_count": 5,
            "success_count": 1,
            "failure_count": 4,
            "mean_iou": 0.9638,
            "minimum_iou": 0.9638,
            "stddev_iou": 0.0,
        }
        normalize_result_metrics(legacy)

        self.assertAlmostEqual(live["mean_iou"], 0.19276, places=5)
        self.assertEqual(live["mean_iou_success"], 0.9638)
        self.assertEqual(live["minimum_iou"], 0.0)
        self.assertEqual(live["failure_count"], 4)
        self.assertEqual(live["mean_iou"], legacy["mean_iou"])
        self.assertEqual(live["mean_iou_success"], legacy["mean_iou_success"])
        self.assertEqual(live["minimum_iou"], legacy["minimum_iou"])

    def test_current_metrics_are_idempotent(self):
        stats = {
            "page_count": 5,
            "success_count": 4,
            "failure_count": 1,
            "mean_iou": 0.72,
            "mean_iou_success": 0.90,
            "minimum_iou": 0.0,
            "stddev_iou": 0.36,
        }
        before = dict(stats)
        normalize_result_metrics(stats)
        self.assertEqual(stats, before)

    def test_summary_normalizes_winner_and_baseline(self):
        summary = {
            "winner": {"summary": {"page_count":5,"success_count":4,"failure_count":1,"mean_iou":0.9,"minimum_iou":0.8,"stddev_iou":0.1}},
            "baseline": {"summary": {"page_count":5,"success_count":5,"failure_count":0,"mean_iou":0.8,"minimum_iou":0.7,"stddev_iou":0.05}},
        }
        normalize_summary_metrics(summary)
        self.assertAlmostEqual(summary["winner"]["summary"]["mean_iou"], 0.72)
        self.assertEqual(summary["winner"]["summary"]["mean_iou_success"], 0.9)
        self.assertEqual(summary["baseline"]["summary"]["mean_iou"], 0.8)
        self.assertEqual(summary["baseline"]["summary"]["mean_iou_success"], 0.8)


if __name__ == "__main__":
    unittest.main()
