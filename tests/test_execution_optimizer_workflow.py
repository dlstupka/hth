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
        self.assertIn("default: adaptive", text)
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
        self.assertIn("allow_thread_oversubscription:", text)
        self.assertIn("Manual exception — allow requested shapes to exceed the detected runner thread budget", text)

    def test_execution_optimizer_supports_single_or_dispatcher_target_modes(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("- all", text)
        self.assertIn("- all-without-preference", text)
        self.assertIn("dispatch-detectors:", text)
        self.assertIn("name: Dispatch detector optimizer runs", text)
        self.assertIn("inputs.algorithm != 'all'", text)
        self.assertIn("inputs.algorithm != 'all-without-preference'", text)
        self.assertIn("actions: write", text)
        self.assertIn("python -m hth.optimizer_dispatch", text)
        dispatch = (ROOT / "hth" / "optimizer_dispatch.py").read_text(encoding="utf-8")
        self.assertIn("preferred_executor_configurations", dispatch)
        self.assertIn("/dispatches", dispatch)
        self.assertIn("name: Optimize detector execution shapes", text)
        self.assertNotIn("uses: ./.github/workflows/_core-hth.yml", text)
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
        self.assertIn('export HTH_EXACT_EXECUTION_SHAPE=1', text)
        self.assertIn('export HTH_EXECUTION_THREAD_BUDGET="$RUNNER_BUDGET"', text)
        self.assertIn('export HTH_ALLOW_THREAD_OVERSUBSCRIPTION="$ALLOW_THREAD_OVERSUBSCRIPTION"', text)
        self.assertIn('bash hth-pipeline/tools/run-detector-regressions.sh', text)
        self.assertIn("Execution optimizer shape $shape_number/$total", text)
        self.assertTrue(DRIVER.is_file())

    def test_execution_optimizer_honors_runner_and_manual_thread_bounds(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('effective_threads=$((RUNNER_BUDGET / pipelines))', text)
        self.assertIn('effective_threads > THREAD_MAX', text)
        self.assertIn('effective_threads < THREAD_MIN', text)
        self.assertIn('feasible_pipeline_max=$((budget / THREAD_MIN))', text)
        self.assertIn('echo "pipeline_max=$PIPELINE_MAX"', text)
        self.assertIn('echo "thread_max=$THREAD_MAX"', text)
        self.assertIn('ALLOW_THREAD_OVERSUBSCRIPTION', text)
        self.assertIn('effective_threads="$THREAD_MIN"', text)
        self.assertIn('OVERSUBSCRIBED execution shape', text)


    def test_regression_driver_honors_optimizer_owned_execution_shape(self) -> None:
        text = DRIVER.read_text(encoding="utf-8")
        self.assertIn('HTH_EXACT_EXECUTION_SHAPE', text)
        self.assertIn('HTH_EXECUTION_THREAD_BUDGET', text)
        self.assertIn('planned_threads="$THREADS"', text)
        self.assertIn('effective_threads_per_pipeline="$THREADS"', text)
        self.assertIn('Execution shape    : optimizer-exact', text)
        self.assertIn('Optimizer requested ${THREADS} threads/pipeline but regression executor resolved', text)

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
        # Reused checkpoint rows must be replayed before the run-local summary is rendered,
        # not only later in the publish step.
        first_store = text.index("python -m hth.optimizer_store")
        first_replay = text.index("--replay-log", text.index("completed shape pipelines=$pipelines"))
        self.assertLess(first_replay, first_store)

    def test_execution_optimizer_collects_proc_metrics_only_with_heartbeat(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("Start execution optimizer heartbeat", text)
        self.assertIn("[execution optimizer heartbeat]", text)
        self.assertIn("python -m hth.runner_metrics", text)
        self.assertIn("sleep 60", text)
        self.assertIn("Stop execution optimizer heartbeat", text)

    def test_execution_optimizer_has_default_three_shape_two_percent_early_stop(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("--threshold-pct 2.0", text)
        self.assertIn("--consecutive 3", text)
        self.assertIn("throughput_plateau", text)
        self.assertIn("3 consecutive completed shapes improved throughput by <=2%", text)
        self.assertIn('ENUMERATION" != "adaptive"', text)
        self.assertIn("generic <=2% plateau stop deferred until the <=2% preferred-shape boundaries are resolved", text)
        self.assertIn('"optimization_wall_seconds"', text)
        self.assertIn('"pipeline_enumeration"', text)

    def test_core_has_no_execution_optimizer_orchestration(self) -> None:
        text = CORE.read_text(encoding="utf-8")
        self.assertNotIn("prepare-execution-optimizer", text)
        self.assertNotIn("optimizer_algorithm", text)
        self.assertNotIn("prepare-execution-optimizer", text)
        self.assertNotIn("optimizer_algorithm", text)


    def test_manual_workflows_offer_specific_runner_selection(self) -> None:
        workflow_dir = ROOT / ".github" / "workflows"
        for name in (
            "execution-optimizer.yml",
            "regress-detector.yml",
            "calibrate-geometry.yml",
            "preprocess.yml",
            "preprocess-test.yml",
            "generate-report.yml",
        ):
            text = (workflow_dir / name).read_text(encoding="utf-8")
            self.assertIn("specific_runner:", text, name)
            self.assertIn("- any", text, name)
            self.assertIn("- custom", text, name)
            self.assertIn("custom_runner_label:", text, name)
            self.assertNotIn("- rh8-al320", text, name)
            self.assertNotIn("- rh8-al97", text, name)
            self.assertNotIn("- rh8-s32", text, name)

    def test_custom_runner_uses_exact_label_and_detected_budget(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("inputs.specific_runner == 'custom'", text)
        self.assertIn("inputs.custom_runner_label", text)
        self.assertIn('fromJSON(format(\'["self-hosted","{0}"]\', inputs.custom_runner_label))', text)
        self.assertIn('budget="$(( $(nproc) * 2 ))"', text)
        self.assertNotIn("rh8-al320", text)
        self.assertNotIn("rh8-al97", text)
        self.assertNotIn("rh8-s32", text)



    def test_execution_optimizer_initializes_optimization_timer_before_metadata(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        initialization = 'optimization_started_epoch="$(date +%s)"'
        metadata_use = '"$optimization_started_epoch" <<\'PYMETA\''
        self.assertIn(initialization, text)
        self.assertIn(metadata_use, text)
        self.assertLess(text.index(initialization), text.index(metadata_use))

if __name__ == "__main__":
    unittest.main()
