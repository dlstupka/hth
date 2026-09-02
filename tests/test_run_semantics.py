import unittest

from hth.regression.run_semantics import (
    evidence_tier_for,
    legacy_run_semantics,
    validate_run_semantics,
)


class RunSemanticsTests(unittest.TestCase):
    def test_execution_mode_and_completeness_define_tier_once(self):
        self.assertEqual(evidence_tier_for("smoke", exhaustive_complete=False), "provisional")
        self.assertEqual(evidence_tier_for("smoke", exhaustive_complete=True), "provisional")
        self.assertEqual(evidence_tier_for("full", exhaustive_complete=False), "partial")
        self.assertEqual(evidence_tier_for("full", exhaustive_complete=True), "authoritative")

    def test_invalid_mode_tier_combinations_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Smoke runs must be provisional"):
            validate_run_semantics("smoke", "authoritative")
        with self.assertRaisesRegex(ValueError, "Full runs cannot carry provisional"):
            validate_run_semantics("full", "provisional")

    def test_explicit_fields_win_over_legacy_hints(self):
        self.assertEqual(
            legacy_run_semantics({
                "run_mode": "full",
                "evidence_tier": "partial",
                "calibration_status": "provisional",
                "search": {"exhaustive_complete": True},
            }),
            ("full", "partial"),
        )


if __name__ == "__main__":
    unittest.main()
