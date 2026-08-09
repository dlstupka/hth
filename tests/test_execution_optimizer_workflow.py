from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "execution-optimizer.yml"
CORE = ROOT / ".github" / "workflows" / "_core-hth.yml"
REGRESSION = ROOT / ".github" / "workflows" / "regress-detector.yml"
DRIVER = ROOT / "tools" / "run-detector-regressions.sh"


class ExecutionOptimizerWorkflowTests(unittest.TestCase):
    def test_execution_optimizer_is_manual_and_supports_exhaustive_powers_of_two_or_adaptive_enumeration(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("name: HTH execution optimizer", text)
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("pipeline_enumeration:", text)
        self.assertIn("default: exhaustive", text)
        self.assertIn("- exhaustive", text)
        self.assertIn("- powers-of-2", text)
        self.assertIn("- adaptive", text)
        self.assertIn("pipeline_min:", text)
        self.assertIn("pipeline_max:", text)
        self.assertIn("thread_min:", text)
        self.assertIn("thread_max:", text)
        self.assertIn("early_stop:", text)
        self.assertIn("default: true", text)
        self.assertIn("resume:", text)
        self.assertIn("default: auto", text)

    def test_execution_optimizer_is_one_direct_job_on_selected_runner(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("jobs:\n  optimize:", text)
        self.assertIn("name: Optimize detector execution shapes", text)
        self.assertNotIn("uses: ./.github/workflows/_core-hth.yml", text)
        self.assertNotIn("gh workflow run", text)
        self.assertNotIn("gh run watch", text)
        self.assertIn("inputs.runner == 'self-hosted-e7k'", text)
        self.assertIn("inputs.runner == 'self-hosted-e9k'", text)

    def test_execution_optimizer_reuses_normal_regression_setup_sequence(self) -> None:
        optimizer = WORKFLOW.read_text(encoding="utf-8")
        regression = REGRESSION.read_text(encoding="utf-8")
        for step in (
            "Checkout HTH pipeline", "Checkout results repository", "Runner diagnostics",
            "Set up Python — GitHub-hosted Linux", "Verify Python — self-hosted Linux",
            "Use repaired Python tool cache — Windows", "Create isolated Python environment",
            "Install dependencies", "Verify Python dependency ABI", "Show toolchain environment",
            "Show OpenCV build", "Benchmark OpenCV",
        ):
            marker = f"- name: {step}"
            self.assertIn(marker, regression)
            self.assertIn(marker, optimizer)

    def test_execution_optimizer_serially_reuses_normal_regression_driver(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("while true; do", text)
        self.assertIn("python -m hth.optimizer_search", text)
        self.assertIn("--mode adaptive", text)
        self.assertIn('export DETECTOR_PIPELINES="$pipelines"', text)
        self.assertIn('export SHARDS="$pipelines"', text)
        self.assertIn('export THREADS="$effective_threads"', text)
        self.assertIn('bash hth-pipeline/tools/run-detector-regressions.sh', text)
        self.assertIn("Execution optimizer shape $shape_number/$total", text)
        self.assertTrue(DRIVER.is_file())

    def test_execution_optimizer_honors_runner_and_manual_thread_bounds(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('effective_threads=$((RUNNER_BUDGET / pipelines))', text)
        self.assertIn('effective_threads > THREAD_MAX', text)
        self.assertIn('effective_threads < THREAD_MIN', text)
        self.assertIn('feasible_pipeline_max=$((budget / THREAD_MIN))', text)

    def test_execution_optimizer_records_shards_and_current_run_only_reports(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("optimizer-work/shards.jsonl", text)
        self.assertIn("HTH_OPTIMIZER_SHARD_LOG", text)
        self.assertIn("--replay-shard-log", text)
        self.assertIn('--optimizer-run-id "$GITHUB_RUN_ID"', text)
        self.assertIn("parallelism-index.json", text)
        self.assertIn("optimizer-index.json", text)
        self.assertIn("execution-optimizer/$DETECTOR_ALGORITHM/summary.md", text)
        self.assertIn("execution-optimizer/$DETECTOR_ALGORITHM/heatmap.svg", text)
        self.assertNotIn("upload-artifact", text)

    def test_execution_optimizer_resume_reuses_only_completed_compatible_shapes(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("python -m hth.optimizer_resume prepare", text)
        self.assertIn("python -m hth.optimizer_resume shape-completed", text)
        self.assertIn("resumed_from_optimizer_run_id", text)
        self.assertIn("completed shape pipelines=$pipelines threads=$effective_threads is already checkpointed", text)

    def test_execution_optimizer_collects_proc_metrics_only_with_heartbeat(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("Start execution optimizer heartbeat", text)
        self.assertIn("[execution optimizer heartbeat]", text)
        self.assertIn("python -m hth.runner_metrics", text)
        self.assertIn("sleep 60", text)
        self.assertIn("Stop execution optimizer heartbeat", text)

    def test_execution_optimizer_has_default_three_shape_one_percent_early_stop(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("--threshold-pct 1.0", text)
        self.assertIn("--consecutive 3", text)
        self.assertIn("throughput_plateau", text)
        self.assertIn("3 consecutive completed shapes improved throughput by <1%", text)

    def test_core_has_no_execution_optimizer_orchestration(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        self.assertNotIn("prepare-execution-optimizer", text)
        self.assertNotIn("optimizer_algorithm", text)
        self.assertNotIn("prepare-execution-optimizer", text)
        self.assertNotIn("optimizer_algorithm", text)


if __name__ == "__main__":
    unittest.main()
