import unittest

from hth.regression.runner import build_winner_page_report


class WinnerPageReportTests(unittest.TestCase):
    def test_regressions_use_the_same_threshold_as_status_and_sort_best_first(self):
        winner = {
            "parameter_set_id": "1234567890abcdef",
            "pages": [
                {"global_ordinal": 1, "status": "ok", "iou": 0.80},
                {"global_ordinal": 5, "status": "ok", "iou": 0.40},
                {"global_ordinal": 6, "status": "ok", "iou": 0.95},
            ],
        }
        baseline = {
            "pages": [
                {"global_ordinal": 1, "status": "ok", "iou": 0.82},
                {"global_ordinal": 5, "status": "ok", "iou": 0.20},
                {"global_ordinal": 6, "status": "ok", "iou": 0.95},
            ],
        }

        report = build_winner_page_report(winner, baseline)

        self.assertEqual(report["counts"]["regressions"], 1)
        self.assertEqual([page["golden_set_page"] for page in report["pages"]], [6, 1, 5])
        regressed = next(page for page in report["pages"] if page["golden_set_page"] == 1)
        self.assertEqual(regressed["status"], "Regressed")
        self.assertEqual(regressed["problem_reasons"], ["Regressed"])
        self.assertEqual(regressed["parameter_set"], "1234567890ab")


if __name__ == "__main__":
    unittest.main()
