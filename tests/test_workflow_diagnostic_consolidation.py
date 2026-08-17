import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = (
    ROOT / ".github/workflows/regress-detector.yml",
    ROOT / ".github/workflows/execution-optimizer.yml",
)


class WorkflowDiagnosticConsolidationTests(unittest.TestCase):
    def test_bootstrap_python_validation_is_owned_by_reusable_environment_step(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertNotIn("- name: Verify Python — self-hosted Linux", text, workflow.name)
            self.assertIn("- name: Verify / Create reusable Python environment", text, workflow.name)
            self.assertIn("Expected bootstrap Python 3.12", text, workflow.name)
            self.assertIn("Bootstrap Python executable:", text, workflow.name)

    def test_opencv_build_and_benchmark_are_one_step(self):
        for workflow in WORKFLOWS:
            text = workflow.read_text(encoding="utf-8")
            self.assertIn("- name: Verify / Benchmark OpenCV runtime", text, workflow.name)
            self.assertNotIn("- name: Show OpenCV build", text, workflow.name)
            self.assertNotIn("- name: Benchmark OpenCV", text, workflow.name)
            start = text.index("- name: Verify / Benchmark OpenCV runtime")
            end = text.find("\n      - name:", start + 1)
            block = text[start:end]
            self.assertIn("cv2.getBuildInformation()", block, workflow.name)
            self.assertIn("GaussianBlur 100 iterations", block, workflow.name)
            self.assertIn("HTH_OPENCV_BENCHMARK_SECONDS", block, workflow.name)


if __name__ == "__main__":
    unittest.main()
