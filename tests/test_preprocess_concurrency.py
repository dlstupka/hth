import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github/workflows/preprocess.yml",
    ROOT / ".github/workflows/preprocess-test.yml",
)


class PreprocessConcurrencyTests(unittest.TestCase):
    def _concurrency_block(self, workflow: Path) -> str:
        text = workflow.read_text(encoding="utf-8")
        return text.split("concurrency:", 1)[1].split("jobs:", 1)[0]

    def test_manual_preprocess_runs_receive_unique_groups(self):
        for workflow in WORKFLOWS:
            with self.subTest(workflow=workflow.name):
                block = self._concurrency_block(workflow)
                self.assertIn("github.event_name == 'workflow_dispatch'", block)
                self.assertIn("format('manual-{0}', github.run_id)", block)
                self.assertNotIn("inputs.runner", block)
                self.assertNotIn("inputs.source_repository", block)

    def test_automatic_preprocess_runs_remain_serialized(self):
        for workflow in WORKFLOWS:
            with self.subTest(workflow=workflow.name):
                block = self._concurrency_block(workflow)
                self.assertIn("|| 'automatic'", block)
                self.assertIn("cancel-in-progress: false", block)


if __name__ == "__main__":
    unittest.main()
