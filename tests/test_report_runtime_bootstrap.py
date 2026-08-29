import pathlib
import unittest


ROOT = pathlib.Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "_core-hth.yml"
PYTHON_ACTION = ROOT / ".github" / "actions" / "setup-hth-python" / "action.yml"


class ReportRuntimeBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")
        cls.report = cls.text.split("  generate-report:", 1)[1]
        cls.action = PYTHON_ACTION.read_text(encoding="utf-8")

    def test_report_uses_canonical_python_bootstrap(self):
        self.assertIn("uses: ./hth-pipeline/.github/actions/setup-hth-python", self.report)
        self.assertIn("- name: Set up Python — GitHub-hosted Linux", self.action)
        self.assertIn("runner.environment == 'github-hosted'", self.action)
        self.assertNotIn("if: ${{ runner.os != 'Windows' }}", self.action)

    def test_report_reuses_canonical_runtime_environment_contract(self):
        self.assertIn("- name: Use repaired Python tool cache — Windows", self.action)
        self.assertIn("- name: Verify / Create reusable Python environment", self.action)
        self.assertIn('runtime_root="/tmp/.ar/.hth-runtime"', self.action)


if __name__ == "__main__":
    unittest.main()
