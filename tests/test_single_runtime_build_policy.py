import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANAGER = (ROOT / "tools" / "ensure-managed-runtime.sh").read_text(encoding="utf-8")
WORKFLOWS = (
    ROOT / ".github/workflows/regress-detector.yml",
    ROOT / ".github/workflows/execution-optimizer.yml",
)


class SingleRuntimeBuildPolicyTests(unittest.TestCase):
    def test_pip_is_not_unconditionally_upgraded(self):
        self.assertNotIn("pip install --upgrade pip", MANAGER)
        self.assertIn("python -m pip --version", MANAGER)
        self.assertIn("python -m ensurepip --upgrade", MANAGER)

    def test_specialized_workflow_steps_do_not_install_or_rebuild(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            dh = text.index("- name: Verify dhSegment TensorFlow runtime")
            kr = text.index("- name: Verify Kraken historical-document segmentation runtime")
            after_kr = text.find("\n      - name:", kr + 1)
            dh_block = text[dh:kr]
            kr_block = text[kr:after_kr]
            self.assertNotIn("pip install", dh_block, workflow.name)
            self.assertNotIn("pip install", kr_block, workflow.name)
            self.assertNotIn("atomic_rebuild", dh_block, workflow.name)
            self.assertNotIn("atomic_rebuild", kr_block, workflow.name)


if __name__ == "__main__":
    unittest.main()
