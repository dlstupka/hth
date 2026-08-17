import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/regress-detector.yml"


class ManualRegressionConcurrencyTests(unittest.TestCase):
    def test_manual_runs_receive_unique_workflow_concurrency_groups(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        block = text.split("concurrency:", 1)[1].split("jobs:", 1)[0]
        self.assertIn("github.event_name == 'workflow_dispatch'", block)
        self.assertIn("format('manual-{0}', github.run_id)", block)
        self.assertNotIn("inputs.runner", block)
        self.assertNotIn("inputs.algorithm", block)

    def test_automatic_runs_remain_serialized_per_workflow_and_ref(self):
        text = WORKFLOW.read_text(encoding="utf-8")
        block = text.split("concurrency:", 1)[1].split("jobs:", 1)[0]
        self.assertIn("|| 'automatic'", block)
        self.assertIn("cancel-in-progress: false", block)


if __name__ == "__main__":
    unittest.main()
