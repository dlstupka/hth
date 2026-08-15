import unittest

from hth.regression.authoritative_record import authoritative_record


class AuthoritativeRecordTests(unittest.TestCase):
    def test_better_authoritative_full_survives_newer_worse_full(self):
        records = [
            {
                "status": "authoritative",
                "search_type": "exhaustive",
                "created_at_utc": "2026-08-10T12:00:00+00:00",
                "build_number": 292,
                "mean_iou": 0.9250,
                "minimum_iou": 0.80,
                "stddev_iou": 0.05,
                "failures": 0,
            },
            {
                "status": "authoritative",
                "search_type": "exhaustive",
                "created_at_utc": "2026-08-10T20:00:00+00:00",
                "build_number": 300,
                "mean_iou": 0.7746,
                "minimum_iou": 0.70,
                "stddev_iou": 0.10,
                "failures": 0,
            },
        ]
        self.assertEqual(authoritative_record(records)["build_number"], 292)

    def test_authoritative_full_beats_newer_higher_scoring_smoke(self):
        records = [
            {
                "status": "authoritative",
                "search_type": "exhaustive",
                "created_at_utc": "2026-08-10T20:00:00+00:00",
                "build_number": 300,
                "mean_iou": 0.90,
            },
            {
                "status": "provisional",
                "search_type": "smoke",
                "created_at_utc": "2026-08-11T01:00:00+00:00",
                "build_number": 305,
                "mean_iou": 0.99,
            },
        ]
        self.assertEqual(authoritative_record(records)["build_number"], 300)

    def test_latest_smoke_is_fallback_when_no_full_exists(self):
        records = [
            {"status": "provisional", "search_type": "smoke", "created_at_utc": "2026-08-10T10:00:00+00:00", "build_number": 10, "mean_iou": 0.99},
            {"status": "provisional", "search_type": "smoke", "created_at_utc": "2026-08-10T11:00:00+00:00", "build_number": 11, "mean_iou": 0.50},
        ]
        self.assertEqual(authoritative_record(records)["build_number"], 11)

    def test_nested_persisted_selection_quality_is_used(self):
        records = [
            {
                "calibration_status": "authoritative",
                "search": {"exhaustive_complete": True},
                "created_at_utc": "2026-08-10T10:00:00Z",
                "build": {"github_run_number": 290},
                "selection": {"best_avg_iou": 0.9137, "minimum_iou": 0.7378, "stddev_iou": 0.0886, "failure_count": 0},
            },
            {
                "calibration_status": "authoritative",
                "search": {"exhaustive_complete": True},
                "created_at_utc": "2026-08-15T14:00:00Z",
                "build": {"github_run_number": 429},
                "selection": {"best_avg_iou": 0.9071, "minimum_iou": 0.7249, "stddev_iou": 0.0912, "failure_count": 0},
            },
        ]
        self.assertEqual(authoritative_record(records)["build"]["github_run_number"], 290)

    def test_minimum_iou_then_failures_then_stddev_break_equal_avg_ties(self):
        records = [
            {"status": "authoritative", "search_type": "exhaustive", "build_number": 1, "mean_iou": 0.95, "minimum_iou": 0.80, "failures": 0, "stddev_iou": 0.04},
            {"status": "authoritative", "search_type": "exhaustive", "build_number": 2, "mean_iou": 0.95, "minimum_iou": 0.81, "failures": 1, "stddev_iou": 0.03},
        ]
        self.assertEqual(authoritative_record(records)["build_number"], 2)


if __name__ == "__main__":
    unittest.main()
