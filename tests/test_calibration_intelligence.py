import unittest

from hth.regression.calibration_intelligence import build_calibration_intelligence


class CalibrationIntelligenceTests(unittest.TestCase):
    def test_characterizes_parameter_influence_basin_and_pages(self):
        ranked = []
        for index, (alpha, beta, score) in enumerate([
            (1, "x", 0.90),
            (1, "y", 0.89),
            (2, "x", 0.70),
            (2, "y", 0.69),
        ]):
            ranked.append({
                "parameter_set_id": str(index),
                "parameters": {"alpha": alpha, "beta": beta},
                "summary": {"mean_iou": score, "failure_count": 0},
                "pages": [
                    {"global_ordinal": 1, "status": "ok", "iou": score},
                    {"global_ordinal": 5, "status": "ok", "iou": score - 0.1},
                ],
            })

        report = build_calibration_intelligence(
            ranked,
            detector="example",
            strategy="exhaustive",
            possible_parameter_sets=4,
        )

        self.assertTrue(report["available"])
        self.assertTrue(report["search"]["exhaustive_complete"])
        self.assertEqual(report["landscape"]["equivalent_winner_count"], 1)
        influence = {item["parameter"]: item for item in report["parameter_influence"]}
        self.assertGreater(influence["alpha"]["eta_squared"], influence["beta"]["eta_squared"])
        self.assertEqual(influence["alpha"]["classification"], "Critical")
        self.assertEqual(len(report["page_sensitivity"]), 2)
        self.assertEqual(report["page_sensitivity"][0]["global_ordinal"], 1)
        self.assertIn(report["calibration_confidence"]["rating"], {"Medium", "High"})

    def test_marks_flat_parameter_as_dormant(self):
        ranked = [
            {
                "parameters": {"unused": value},
                "summary": {"mean_iou": 0.8, "failure_count": 0},
                "pages": [],
            }
            for value in (1, 2, 3)
        ]
        report = build_calibration_intelligence(
            ranked,
            detector="flat",
            strategy="exhaustive",
            possible_parameter_sets=3,
        )
        self.assertEqual(report["parameter_influence"][0]["classification"], "Dormant")
        self.assertEqual(report["recommendations"]["dormant_parameters"], ["unused"])


if __name__ == "__main__":
    unittest.main()
