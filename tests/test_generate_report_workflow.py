from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "generate-report.yml"
CORE = ROOT / ".github" / "workflows" / "_core-hth.yml"


class GenerateReportWorkflowTests(unittest.TestCase):
    def test_report_workflow_is_manual_with_report_and_runner_choices(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", text)
        self.assertNotIn("push:", text)
        self.assertIn("detector-calibration-manifest", text)
        self.assertIn("execution-optimizer", text)
        self.assertIn("default: all", text)
        self.assertIn("          - all", text)
        self.assertIn("default: github-hosted", text)
        self.assertIn("uses: ./.github/workflows/_core-hth.yml", text)
        self.assertIn("mode: report", text)
        for runner in ("self-hosted-hth", "self-hosted-windows", "self-hosted-rhel8", "self-hosted-e7k", "self-hosted-e9k"):
            self.assertIn(runner, text)

    def test_core_exposes_common_manual_runner_selection(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        self.assertIn('runner:\n        description: "Execution runner"', text)
        self.assertIn("inputs.runner == 'self-hosted-e7k'", text)
        self.assertIn("inputs.runner == 'self-hosted-e9k'", text)

    def test_core_optimizer_summary_is_appended_after_report_publish(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        generate_step = text.split("- name: Generate selected report", 1)[1].split("- name: Publish regenerated report", 1)[0]
        self.assertNotIn("report-run={run_id}", generate_step)
        self.assertIn("- name: Publish optimizer report summary", text)
        publish_pos = text.index("- name: Publish regenerated report")
        summary_pos = text.index("- name: Publish optimizer report summary")
        self.assertLess(publish_pos, summary_pos)
        summary_step = text[summary_pos:]
        self.assertIn("python -c", summary_step)
        self.assertNotIn("PYSUMMARY", summary_step)
        self.assertIn("report-run={run_id}", summary_step)
        self.assertIn("re.sub", summary_step)

    def test_core_publishes_optimizer_report_directory_recursively(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        publish_step = text.split("- name: Publish regenerated report", 1)[1]
        self.assertIn('cp -a "generated-report/execution-optimizer/${{ inputs.report_algorithm }}/."', publish_step)
        self.assertIn('git -C results-repo add "execution-optimizer/${{ inputs.report_algorithm }}"', publish_step)


if __name__ == "__main__":
    unittest.main()
