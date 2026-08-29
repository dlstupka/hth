import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANAGER = (ROOT / "tools" / "ensure-managed-runtime.sh").read_text(encoding="utf-8")
MANAGED_ACTION = (ROOT / ".github/actions/setup-hth-managed-runtime/action.yml").read_text(encoding="utf-8")
WORKFLOWS = (
    ROOT / ".github/workflows/regress-detector.yml",
    ROOT / ".github/workflows/execution-optimizer.yml",
)


class SingleRuntimeBuildPolicyTests(unittest.TestCase):
    def test_pip_is_not_unconditionally_upgraded(self):
        self.assertNotIn("pip install --upgrade pip", MANAGER)
        self.assertIn("python -m pip --version", MANAGER)
        self.assertIn("python -m ensurepip --upgrade", MANAGER)

    def test_specialized_runtime_steps_do_not_install_or_rebuild(self):
        dh = MANAGED_ACTION.index("- name: Verify dhSegment TensorFlow runtime")
        kr = MANAGED_ACTION.index("- name: Verify Kraken historical-document segmentation runtime")
        after_kr = MANAGED_ACTION.find("\n    - name:", kr + 1)
        dh_block = MANAGED_ACTION[dh:kr]
        kr_block = MANAGED_ACTION[kr:after_kr]
        self.assertNotIn("pip install", dh_block)
        self.assertNotIn("pip install", kr_block)
        self.assertNotIn("atomic_rebuild", dh_block)
        self.assertNotIn("atomic_rebuild", kr_block)
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("uses: ./hth-pipeline/.github/actions/setup-hth-managed-runtime", text, workflow.name)


if __name__ == "__main__":
    unittest.main()
