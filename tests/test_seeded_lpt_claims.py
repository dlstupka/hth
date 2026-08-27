import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "tools" / "run-detector-regressions.sh").read_text(encoding="utf-8")


class StaticLptShellContractTests(unittest.TestCase):
    def test_multidetector_lpt_builds_static_plan(self):
        self.assertIn('initial_claim_strategy="static-lpt-plan"', DRIVER)
        self.assertIn("plan_static_lpt_tasks", DRIVER)
        self.assertIn("Static schedule pipeline=", DRIVER)

    def test_schedule_is_built_before_worker_fanout(self):
        plan = DRIVER.index("Static schedule pipeline=")
        fanout = DRIVER.index('detector_worker "$pipeline_index"')
        self.assertLess(plan, fanout)

    def test_telemetry_records_static_scheduler_strategy(self):
        self.assertIn('--claim-strategy "$initial_claim_strategy"', DRIVER)


if __name__ == "__main__":
    unittest.main()
