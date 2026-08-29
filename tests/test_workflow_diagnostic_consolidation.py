import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github/workflows/regress-detector.yml",
    ROOT / ".github/workflows/execution-optimizer.yml",
)
PYTHON_ACTION = ROOT / ".github/actions/setup-hth-python/action.yml"
MANAGED_ACTION = ROOT / ".github/actions/setup-hth-managed-runtime/action.yml"


class WorkflowDiagnosticConsolidationTests(unittest.TestCase):
    def test_bootstrap_python_validation_is_owned_by_canonical_action(self):
        action = PYTHON_ACTION.read_text(encoding="utf-8")
        self.assertNotIn("- name: Verify Python — self-hosted Linux", action)
        self.assertIn("- name: Verify / Create reusable Python environment", action)
        self.assertIn("Expected bootstrap Python 3.12", action)
        self.assertIn("Bootstrap Python executable:", action)
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("uses: ./hth-pipeline/.github/actions/setup-hth-python", text, workflow.name)

    def test_opencv_build_and_benchmark_are_one_canonical_step(self):
        action = MANAGED_ACTION.read_text(encoding="utf-8")
        self.assertIn("- name: Verify / Benchmark OpenCV runtime", action)
        self.assertNotIn("- name: Show OpenCV build", action)
        self.assertNotIn("- name: Benchmark OpenCV", action)
        start = action.index("- name: Verify / Benchmark OpenCV runtime")
        block = action[start:]
        self.assertIn("cv2.getBuildInformation()", block)
        self.assertIn("GaussianBlur 100 iterations", block)
        self.assertIn("HTH_OPENCV_BENCHMARK_SECONDS", block)
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn('benchmark-opencv: "true"', text, workflow.name)


if __name__ == "__main__":
    unittest.main()
