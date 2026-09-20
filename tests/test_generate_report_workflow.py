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
        self.assertIn("full-normalization-summary", text)
        self.assertIn("default: all", text)
        self.assertIn("          - all", text)
        self.assertIn("default: github-hosted", text)
        self.assertIn("golden_release_tag:", text)
        self.assertIn("default: HTH-GOLDEN-0002", text)
        self.assertIn("          - HTH-GOLDEN-0001", text)
        self.assertIn("          - HTH-GOLDEN-0002", text)
        self.assertNotIn("default: config/golden_set.json", text)
        self.assertIn("report_golden_set: ${{ inputs.golden_release_tag }}", text)
        self.assertIn("uses: ./.github/workflows/_core-hth.yml", text)
        self.assertIn("mode: report", text)
        for runner in ("self-hosted-linux", "self-hosted-windows", "hth", "rhel8", "e7k", "e9k", "192t", "96t", "32t"):
            self.assertIn(runner, text)

    def test_core_exposes_common_manual_runner_selection(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        self.assertIn('runner_target:\n        description: "Canonical execution runner target"', text)
        self.assertIn("inputs.runner_target == 'e7k'", text)
        self.assertIn("inputs.runner_target == 'e9k'", text)

    def test_core_resolves_report_release_tag_to_validated_canonical_file(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        self.assertIn("- name: Resolve report Golden Set release", text)
        self.assertIn("python -m hth.golden_set_catalog", text)
        self.assertIn('--release-tag "${{ inputs.report_golden_set }}"', text)
        self.assertIn('--golden-set "${{ steps.report_golden_set.outputs.golden_set_path }}"', text)
        self.assertIn('FREEZE_SOURCE="${{ steps.report_golden_set.outputs.freeze_path }}"', text)


    def test_core_report_results_checkouts_are_explicit_and_main_only(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        report_job = text.split("generate-report:", 1)[1]
        setup = report_job.split("- name: Set up canonical HTH Python runtime", 1)[0]
        compact = setup.split("- name: Checkout compact normalization audit evidence", 1)[1].split(
            "- name: Checkout detector calibration report evidence", 1
        )[0]
        detector = setup.split("- name: Checkout detector calibration report evidence", 1)[1].split(
            "- name: Checkout execution optimizer report evidence", 1
        )[0]
        optimizer = setup.split("- name: Checkout execution optimizer report evidence", 1)[1]
        for checkout in (compact, detector, optimizer):
            self.assertIn("ref: main", checkout)
            self.assertNotIn("fetch-depth: 0", checkout)
            self.assertIn("sparse-checkout:", checkout)
            self.assertIn("sparse-checkout-cone-mode: false", checkout)
        self.assertIn("fetch-depth: 1", compact)
        self.assertIn("fetch-depth: 1", detector)
        self.assertIn("fetch-depth: 100", optimizer)

    def test_normalization_report_uses_compact_canonical_audit_checkout(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        checkout = text.split("- name: Checkout compact normalization audit evidence", 1)[1].split(
            "- name: Checkout detector calibration report evidence", 1
        )[0]
        self.assertIn("inputs.report_type == 'full-normalization-summary'", checkout)
        self.assertIn("/metadata/resource-lifecycle.json", checkout)
        self.assertIn("/normalization/", checkout)
        self.assertIn("/reports/full-normalization-summary.md", checkout)
        self.assertIn("sparse-checkout-cone-mode: false", checkout)

    def test_detector_report_checkout_declares_indexes_records_and_publish_surface(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        checkout = text.split("- name: Checkout detector calibration report evidence", 1)[1].split(
            "- name: Checkout execution optimizer report evidence", 1
        )[0]
        self.assertIn("inputs.report_type == 'detector-calibration-manifest'", checkout)
        self.assertIn("/indexes/calibration-index.json", checkout)
        self.assertIn("/indexes/runtime-index.json", checkout)
        self.assertIn("/source-documents/", checkout)
        self.assertIn("/reports/", checkout)

    def test_optimizer_report_checkout_declares_indexes_history_and_publish_surface(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        checkout = text.split("- name: Checkout execution optimizer report evidence", 1)[1].split(
            "- name: Set up canonical HTH Python runtime", 1
        )[0]
        self.assertIn("inputs.report_type == 'execution-optimizer'", checkout)
        self.assertIn("/indexes/parallelism-index.json", checkout)
        self.assertIn("/indexes/optimizer-index.json", checkout)
        self.assertIn("/indexes/optimizer-predictions.json", checkout)
        self.assertIn("/execution-optimizer/", checkout)
        self.assertNotIn("source-documents", checkout)

    def test_normalization_research_artifact_excludes_detector_calibration_tree(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        assemble = text.split("- name: Assemble report research artifact", 1)[1].split(
            "- name: Upload report research artifact", 1
        )[0]
        normalization = assemble.split(
            'if [[ "${{ inputs.report_type }}" == "full-normalization-summary" ]]', 1
        )[1].split("else", 1)[0]
        self.assertIn("results-repo/metadata/resource-lifecycle.json", normalization)
        self.assertIn("cp -a results-repo/normalization", normalization)
        self.assertNotIn("source-documents", normalization)

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
        self.assertIn("full-normalization-summary.md", summary_step)
        self.assertIn("python -c", summary_step)
        self.assertIn("RESULTS_COMMIT=", summary_step)
        self.assertIn("/{results_commit}/execution-optimizer/", summary_step)
        self.assertNotIn("report-run={run_id}", summary_step)
        self.assertIn("re.sub", summary_step)
        optimizer_summary_tail = summary_step.split('"$RESULTS_COMMIT"', 1)[1]
        self.assertTrue(
            optimizer_summary_tail.lstrip().startswith('elif [[ "${{ inputs.report_type }}" == "full-normalization-summary" ]]'),
            "report-summary optimizer branch must flow into the normalization-summary branch",
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
        self.assertIn('hth_results_stage results-repo "execution-optimizer/${{ inputs.report_algorithm }}"', publish_step)

    def test_core_generates_and_publishes_full_normalization_summary(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        self.assertIn("python -m hth.report_generator full-normalization-summary", text)
        self.assertIn("--pipeline-commit \"${{ github.sha }}\"", text)
        self.assertIn("reports/full-normalization-summary.md", text)
        self.assertIn("generated-report/full-normalization-summary.md \"$GITHUB_STEP_SUMMARY\"", text)


if __name__ == "__main__":
    unittest.main()
