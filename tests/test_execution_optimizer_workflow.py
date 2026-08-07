from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "execution-optimizer.yml"
CORE = ROOT / ".github" / "workflows" / "_core-hth.yml"
DRIVER = ROOT / "tools" / "run-detector-regressions.sh"


def test_execution_optimizer_is_manual_and_supports_auto_or_manual_ranges() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "name: HTH execution optimizer" in text
    assert "workflow_dispatch:" in text
    assert "pipeline_range:" in text
    assert "default: auto" in text
    assert "pipeline_min:" in text
    assert "pipeline_max:" in text


def test_execution_optimizer_is_one_core_job_on_the_selected_runner() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "uses: ./.github/workflows/_core-hth.yml" in text
    assert "mode: execution-optimizer" in text
    assert "optimizer_runner: ${{ inputs.runner }}" in text
    assert "gh workflow run" not in text
    assert "gh run watch" not in text
    assert "runs-on: ubuntu-latest" not in text


def test_execution_optimizer_serially_reuses_normal_regression_driver() -> None:
    core = CORE.read_text(encoding="utf-8")
    assert "Run execution optimizer experiments" in core
    assert 'for pipelines in "${candidates[@]}"; do' in core
    assert 'export DETECTOR_PIPELINES="$pipelines"' in core
    assert 'export SHARDS="$pipelines"' in core
    assert 'export THREADS="auto"' in core
    assert 'bash hth-pipeline/tools/run-detector-regressions.sh' in core
    assert "Execution optimizer shape $shape_number/$total" in core
    assert DRIVER.is_file()


def test_execution_optimizer_records_only_optimizer_intelligence() -> None:
    core = CORE.read_text(encoding="utf-8")
    optimizer = core.split("  prepare-execution-optimizer:", 1)[1]
    assert "python -m hth.optimizer_capture" in optimizer
    assert "python -m hth.optimizer_store" in optimizer
    assert "parallelism-index.json" in optimizer
    assert "optimizer-index.json" in optimizer
    assert "execution-optimizer/$DETECTOR_ALGORITHM/summary.md" in optimizer
    assert "execution-optimizer/$DETECTOR_ALGORITHM/heatmap.svg" in optimizer
    assert "hth.calibration_store publish" not in optimizer
    assert "Write Regression Manifest" not in optimizer
    assert "upload-artifact" not in optimizer


def test_execution_optimizer_rebases_optimizer_observations_on_latest_results_state() -> None:
    core = CORE.read_text(encoding="utf-8")
    optimizer = core.split("  prepare-execution-optimizer:", 1)[1]
    assert "optimizer-work/observations.jsonl" in optimizer
    assert "git -C results-repo fetch origin main" in optimizer
    assert "git -C results-repo reset --hard origin/main" in optimizer
    assert "--replay-log" in optimizer
    assert "max_publish_attempts=5" in optimizer


def test_execution_optimizer_uses_core_environment_checks_and_heartbeat() -> None:
    core = CORE.read_text(encoding="utf-8")
    for step in (
        "Runner diagnostics",
        "Set up Python — GitHub-hosted Linux",
        "Verify Python — self-hosted Linux",
        "Create isolated Python environment",
        "Install dependencies",
        "Verify Python dependency ABI",
        "Show toolchain environment",
        "Show OpenCV build",
        "Benchmark OpenCV",
        "Build pipeline enumeration",
    ):
        assert f"- name: {step}" in core
    assert "[hth core heartbeat]" in core
