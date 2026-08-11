import unittest

from hth.regression.authoritative_record import authoritative_record


class AuthoritativeRecordTests(unittest.TestCase):
    def test_newest_authoritative_full_wins_even_with_worse_quality(self):
        records = [
            {
                "status": "authoritative",
                "search_type": "exhaustive",
                "created_at_utc": "2026-08-10T12:00:00+00:00",
                "build_number": 292,
                "mean_iou": 0.9250,
                "failures": 0,
            },
            {
                "status": "authoritative",
                "search_type": "exhaustive",
                "created_at_utc": "2026-08-10T20:00:00+00:00",
                "build_number": 300,
                "mean_iou": 0.7746,
                "failures": 1,
            },
        ]
        self.assertEqual(authoritative_record(records)["build_number"], 300)

    def test_authoritative_full_beats_newer_smoke(self):
        records = [
            {
                "status": "authoritative",
                "search_type": "exhaustive",
                "created_at_utc": "2026-08-10T20:00:00+00:00",
                "build_number": 300,
            },
            {
                "status": "provisional",
                "search_type": "smoke",
                "created_at_utc": "2026-08-11T01:00:00+00:00",
                "build_number": 305,
            },
        ]
        self.assertEqual(authoritative_record(records)["build_number"], 300)

    def test_latest_smoke_is_fallback_when_no_full_exists(self):
        records = [
            {"status": "provisional", "search_type": "smoke", "created_at_utc": "2026-08-10T10:00:00+00:00", "build_number": 10},
            {"status": "provisional", "search_type": "smoke", "created_at_utc": "2026-08-10T11:00:00+00:00", "build_number": 11},
        ]
        self.assertEqual(authoritative_record(records)["build_number"], 11)

    def test_quality_never_selects_provenance(self):
        records = [
            {"status": "authoritative", "search_type": "exhaustive", "created_at_utc": "2026-08-10T10:00:00+00:00", "build_number": 1, "mean_iou": 0.99},
            {"status": "authoritative", "search_type": "exhaustive", "created_at_utc": "2026-08-10T11:00:00+00:00", "build_number": 2, "mean_iou": 0.50},
        ]
        self.assertEqual(authoritative_record(records)["build_number"], 2)


if __name__ == "__main__":
    unittest.main()
