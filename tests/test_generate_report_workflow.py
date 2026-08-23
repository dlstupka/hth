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


    def test_core_report_results_checkout_is_shallow_main_only(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        report_job = text.split("generate-report:", 1)[1]
        checkout = report_job.split("- name: Checkout results repository", 1)[1].split("- name: Set up Python", 1)[0]
        self.assertIn("ref: main", checkout)
        self.assertIn("fetch-depth: 1", checkout)
        self.assertNotIn("fetch-depth: 0", checkout)

    def test_core_report_summary_is_appended_after_successful_publish(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        generate_step = text.split("- name: Generate selected report", 1)[1].split("- name: Publish regenerated report", 1)[0]
        self.assertNotIn("GITHUB_STEP_SUMMARY", generate_step)
        self.assertIn("- name: Publish regenerated report summary", text)
        publish_pos = text.index("- name: Publish regenerated report")
        summary_pos = text.index("- name: Publish regenerated report summary")
        self.assertLess(publish_pos, summary_pos)
        summary_step = text[summary_pos:]
        self.assertIn("detector-calibration-manifest.md", summary_step)
        self.assertIn("python -c", summary_step)
        self.assertIn("report-run={run_id}", summary_step)
        self.assertIn("re.sub", summary_step)
        optimizer_summary_tail = summary_step.split('"$GITHUB_RUN_ID"', 1)[1]
        self.assertTrue(
            optimizer_summary_tail.lstrip().startswith("fi"),
            "report-summary detector/optimizer conditional must be closed",
        )

    def test_core_report_publish_retries_concurrent_results_updates_with_regeneration(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        publish_step = text.split("- name: Publish regenerated report", 1)[1].split("- name: Publish regenerated report summary", 1)[0]
        self.assertIn("source hth-pipeline/tools/hardened-persistence.sh", publish_step)
        self.assertIn("hth_hardened_persist", publish_step)
        self.assertIn("regenerate_and_stage", publish_step)

    def test_core_publishes_optimizer_report_directory_recursively(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        publish_step = text.split("- name: Publish regenerated report", 1)[1]
        self.assertIn('cp -a "generated-report/execution-optimizer/${{ inputs.report_algorithm }}/."', publish_step)
        self.assertIn('git -C results-repo add -A -- "execution-optimizer/${{ inputs.report_algorithm }}"', publish_step)


if __name__ == "__main__":
    unittest.main()
