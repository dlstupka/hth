import pathlib
import unittest


WORKFLOW = pathlib.Path(__file__).parents[1] / ".github" / "workflows" / "_core-hth.yml"


class ReportRuntimeBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")
        cls.report = cls.text.split("  generate-report:", 1)[1]

    def test_report_uses_setup_python_only_on_github_hosted_linux(self):
        self.assertIn("- name: Set up Python — GitHub-hosted Linux", self.report)
        self.assertIn("runner.environment == 'github-hosted'", self.report)
        self.assertNotIn("if: ${{ runner.os != 'Windows' }}", self.report)

    def test_report_uses_reusable_runtime_bootstrap(self):
        self.assertIn("- name: Use repaired Python tool cache — Windows", self.report)
        self.assertIn("- name: Verify / Create reusable Python environment", self.report)
        self.assertIn('runtime_root="/tmp/.ar/.hth-runtime"', self.report)


if __name__ == "__main__":
    unittest.main()
