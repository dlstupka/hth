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
        self.assertEqual(report["schema_version"], "1.1")
        self.assertIn("calibration_identity", report)
        self.assertIn("regression_metadata", report)
        self.assertEqual(report["detector_evidence"]["detector_id"], "example")
        self.assertIn("parameters", report["parameter_intelligence"])
        self.assertIn("domains", report["domain_space_intelligence"])
        self.assertEqual(report["detector_selection_intelligence"]["recommended_detector_id"], "example")

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


    def test_preserves_calibration_and_regression_context(self):
        ranked = [{
            "parameter_set_id": "winner-id",
            "parameters": {"alpha": 1},
            "summary": {"mean_iou": 0.9, "minimum_iou": 0.8, "stddev_iou": 0.05, "failure_count": 0},
            "pages": [],
        }]
        calibration_context = {
            "calibration_run_id": "run-1",
            "source_document": {"title": "Example"},
            "golden_set": {"collection_id": "GS-1", "sha256": "abc"},
            "detector_configuration": {"sha256": "def"},
        }
        regression_context = {"requested_strategy": "critical", "resolved_strategy": "exhaustive"}
        report = build_calibration_intelligence(
            ranked, detector="example", strategy="exhaustive", possible_parameter_sets=1,
            calibration_context=calibration_context, regression_context=regression_context,
        )
        self.assertEqual(report["calibration_identity"], calibration_context)
        self.assertEqual(report["regression_metadata"], regression_context)
        selection = report["detector_selection_intelligence"]
        self.assertEqual(selection["recommended_parameter_set_id"], "winner-id")
        self.assertEqual(selection["applicability"]["golden_set"]["collection_id"], "GS-1")
        self.assertEqual(report["domain_space_intelligence"]["default_strategy"], "exhaustive")
        self.assertEqual(report["domain_space_intelligence"]["fallback_order"][-1], "exhaustive")



if __name__ == "__main__":
    unittest.main()
