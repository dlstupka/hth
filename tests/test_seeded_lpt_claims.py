import unittest
from pathlib import Path

from hth.domain.execution_dispatch import plan_static_dispatch

ROOT = Path(__file__).resolve().parents[1]
DRIVER = (ROOT / "tools" / "run-detector-regressions.sh").read_text(encoding="utf-8")


class StaticLptShellContractTests(unittest.TestCase):
    def test_multidetector_lpt_builds_static_plan(self):
        self.assertIn('initial_claim_strategy="static-lpt-plan"', DRIVER)
        self.assertIn("plan_static_dispatch", DRIVER)
        self.assertIn("Static schedule pipeline=", DRIVER)
        single = plan_static_dispatch(task_count=9, pipeline_count=9, multidetector=False)
        self.assertEqual([row["task_indexes"] for row in single], [[i] for i in range(9)])

    def test_schedule_is_built_before_worker_fanout(self):
        plan = DRIVER.index("Static schedule pipeline=")
        fanout = DRIVER.index('detector_worker "$pipeline_index"')
        self.assertLess(plan, fanout)
        self.assertNotIn('static_pipeline_tasks[0]="$(seq -s, 0', DRIVER)
        self.assertNotIn('task_index % effective_pipelines', DRIVER)

    def test_telemetry_records_static_scheduler_strategy(self):
        self.assertIn('--claim-strategy "$initial_claim_strategy"', DRIVER)


if __name__ == "__main__":
    unittest.main()
