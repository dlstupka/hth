import unittest

from hth.regression.runner import failure_diagnostics
from hth.regression.calibration_intelligence import build_calibration_intelligence


class FailureDiagnosticsTests(unittest.TestCase):
    def test_runner_summarizes_candidate_reasons_and_probability_ranges(self):
        result = {"pages": [
            {"status": "no_candidate", "candidate": {"diagnostics": {"reason": "no_learned_page_region", "probability_max": 0.12, "thresholded_fraction": 0.0}}},
            {"status": "no_candidate", "candidate": {"diagnostics": {"reason": "no_learned_page_region", "probability_max": 0.18, "thresholded_fraction": 0.0}}},
            {"status": "error", "error": {"type": "RuntimeError", "message": "boom"}},
        ]}
        diag = failure_diagnostics(result)
        self.assertEqual(diag["reason_counts"], {"no_learned_page_region": 2, "RuntimeError": 1})
        self.assertEqual(diag["diagnostic_ranges"]["probability_max"], {"min": 0.12, "max": 0.18})


    def test_runner_preserves_registry_exception_message_and_traceback(self):
        result = {"pages": [
            {
                "global_ordinal": 6,
                "status": "error",
                "candidate": {
                    "diagnostics": {
                        "reason": "detector_exception",
                        "exception_type": "ValueError",
                        "exception_message": "bad serving signature",
                        "traceback": "Traceback...bad serving signature",
                    }
                },
            }
        ]}
        diag = failure_diagnostics(result)
        self.assertEqual(diag["reason_counts"], {"detector_exception": 1})
        self.assertEqual(len(diag["exceptions"]), 1)
        self.assertEqual(diag["exceptions"][0]["type"], "ValueError")
        self.assertEqual(diag["exceptions"][0]["message"], "bad serving signature")
        self.assertEqual(diag["exceptions"][0]["example_page"], 6)
        self.assertIn("Traceback", diag["exceptions"][0]["traceback"])

    def test_calibration_measurement_state_preserves_failure_reasons(self):
        ranked = [{
            "parameter_set_id": "x", "parameters": {"threshold": 0.5},
            "summary": {"mean_iou": 0.0, "failure_count": 2},
            "pages": [
                {"status": "no_candidate", "iou": 0.0, "candidate": {"diagnostics": {"reason": "learned_mask_too_small"}}},
                {"status": "no_candidate", "iou": 0.0, "candidate": {"diagnostics": {"reason": "learned_mask_too_small"}}},
            ],
        }]
        report = build_calibration_intelligence(ranked, detector="learned_page_mask", strategy="exhaustive", possible_parameter_sets=1)
        self.assertFalse(report["measurement_state"]["informative"])
        self.assertEqual(report["measurement_state"]["failure_reason_counts"], {"learned_mask_too_small": 2})


if __name__ == "__main__":
    unittest.main()
