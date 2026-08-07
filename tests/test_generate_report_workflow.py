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


if __name__ == "__main__":
    unittest.main()
