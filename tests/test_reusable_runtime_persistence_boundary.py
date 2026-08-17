import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github/workflows/regress-detector.yml",
    ROOT / ".github/workflows/execution-optimizer.yml",
)


class ReusableRuntimePersistenceBoundaryTests(unittest.TestCase):
    def test_self_hosted_runtime_is_outside_checkout_clean_tree(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn('runtime_root="/tmp/.ar/.hth-runtime"', text, workflow.name)
            self.assertNotIn('runtime_root="$GITHUB_WORKSPACE/.hth-runtime"', text, workflow.name)

    def test_manual_wipe_removes_workspace_and_external_runtime(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn('find "$GITHUB_WORKSPACE" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +', text)
            self.assertIn('rm -rf "/tmp/.ar/.hth-runtime"', text)

    def test_successful_verification_explicitly_reuses_previous_install(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("Base dependencies verified — using previous install; no install required.", text)
            self.assertIn("dhSegment TensorFlow runtime verified — using previous install; no install required.", text)
            self.assertIn("Kraken runtime verified — using previous install; no install required.", text)


if __name__ == "__main__":
    unittest.main()
