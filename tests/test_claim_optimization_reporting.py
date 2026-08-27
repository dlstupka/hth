import unittest
from hth.write_regression_summary import _static_pipeline_schedule


class StaticScheduleReportingTests(unittest.TestCase):
    def test_report_uses_balanced_static_schedule(self):
        rows = [
            {"detector": "a", "estimate_seconds": 10.0},
            {"detector": "b", "estimate_seconds": 8.0},
            {"detector": "c", "estimate_seconds": 4.0},
        ]
        plan = _static_pipeline_schedule(rows, 2)
        self.assertEqual([r["detector"] for r in plan[0]["tasks"]], ["a"])
        self.assertEqual([r["detector"] for r in plan[1]["tasks"]], ["b", "c"])
        self.assertAlmostEqual(plan[0]["estimated_seconds"], 10.0)
        self.assertAlmostEqual(plan[1]["estimated_seconds"], 12.0)


if __name__ == "__main__":
    unittest.main()
