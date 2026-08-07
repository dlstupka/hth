from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "execution-optimizer.yml"
CORE = ROOT / ".github" / "workflows" / "_core-hth.yml"
REGRESSION = ROOT / ".github" / "workflows" / "regress-detector.yml"
DRIVER = ROOT / "tools" / "run-detector-regressions.sh"


def test_execution_optimizer_is_manual_and_supports_auto_or_manual_ranges() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "name: HTH execution optimizer" in text
    assert "workflow_dispatch:" in text
    assert "pipeline_range:" in text
    assert "default: auto" in text
    assert "pipeline_min:" in text
    assert "pipeline_max:" in text


def test_execution_optimizer_is_one_direct_job_on_selected_runner() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "jobs:\n  optimize:" in text
    assert "name: Optimize detector execution shapes" in text
    assert "uses: ./.github/workflows/_core-hth.yml" not in text
    assert "gh workflow run" not in text
    assert "gh run watch" not in text
    assert "inputs.runner == 'self-hosted-e7k'" in text
    assert "inputs.runner == 'self-hosted-e9k'" in text


def test_execution_optimizer_reuses_normal_regression_setup_sequence() -> None:
    optimizer = WORKFLOW.read_text(encoding="utf-8")
    regression = REGRESSION.read_text(encoding="utf-8")
    setup_steps = (
        "Checkout HTH pipeline",
        "Checkout results repository",
        "Runner diagnostics",
        "Set up Python — GitHub-hosted Linux",
        "Verify Python — self-hosted Linux",
        "Use repaired Python tool cache — Windows",
        "Create isolated Python environment",
        "Install dependencies",
        "Verify Python dependency ABI",
        "Show toolchain environment",
        "Show OpenCV build",
        "Benchmark OpenCV",
    )
    for step in setup_steps:
        marker = f"- name: {step}"
        assert marker in regression
        assert marker in optimizer


def test_execution_optimizer_serially_reuses_normal_regression_driver() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'for pipelines in "${candidates[@]}"; do' in text
    assert 'export DETECTOR_PIPELINES="$pipelines"' in text
    assert 'export SHARDS="$pipelines"' in text
    assert 'export THREADS="auto"' in text
    assert 'bash hth-pipeline/tools/run-detector-regressions.sh' in text
    assert "Execution optimizer shape $shape_number/$total" in text
    assert DRIVER.is_file()


def test_execution_optimizer_records_only_optimizer_intelligence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m hth.optimizer_capture" in text
    assert "python -m hth.optimizer_store" in text
    assert "parallelism-index.json" in text
    assert "optimizer-index.json" in text
    assert "execution-optimizer/$DETECTOR_ALGORITHM/summary.md" in text
    assert "execution-optimizer/$DETECTOR_ALGORITHM/heatmap.svg" in text
    assert "hth.calibration_store publish" not in text
    assert "Write Regression Manifest" not in text
    assert "upload-artifact" not in text


def test_execution_optimizer_rebases_observations_before_publish() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "optimizer-work/observations.jsonl" in text
    assert "git -C results-repo fetch origin main" in text
    assert "git -C results-repo reset --hard origin/main" in text
    assert "--replay-log" in text
    assert "max_publish_attempts=5" in text


def test_execution_optimizer_has_one_aggregate_heartbeat() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "Start execution optimizer heartbeat" in text
    assert "[execution optimizer heartbeat]" in text
    assert "Stop execution optimizer heartbeat" in text


def test_core_has_no_execution_optimizer_orchestration() -> None:
    text = CORE.read_text(encoding="utf-8")
    assert "prepare-execution-optimizer" not in text
    assert "optimizer_algorithm" not in text
    assert "execution-optimizer" not in text
