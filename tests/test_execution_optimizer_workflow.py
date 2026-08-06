from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "execution-optimizer.yml"


def test_execution_optimizer_is_manual_and_supports_auto_or_manual_ranges() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "name: HTH execution optimizer" in text
    assert "workflow_dispatch:" in text
    assert "pipeline_range:" in text
    assert "default: auto" in text
    assert "pipeline_min:" in text
    assert "pipeline_max:" in text


def test_execution_optimizer_dispatches_full_regressions_with_auto_threads() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "gh workflow run regress-detector.yml" in text
    assert "-f threads=auto" in text
    assert '-f shards="$pipelines"' in text
    assert '-f detector_pipelines="$pipelines"' in text
    assert "-f mode=full" in text
    assert 'gh run watch "$run_id" --exit-status' in text


def test_execution_optimizer_uses_runner_budget_candidates() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "from hth.regression.sharding import runner_max_threads" in text
    assert "preferred = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192]" in text
    assert "Manual pipeline range must satisfy" in text


def test_execution_optimizer_publishes_optimizer_table_heatmap_and_index() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m hth.optimizer_store" in text
    assert "optimizer-index.json" in text
    assert 'execution-optimizer/$ALGORITHM/summary.md' in text
    assert 'execution-optimizer/$ALGORITHM/heatmap.svg' in text
    assert "Execution optimizer heat map" in text


def test_execution_optimizer_validates_selected_runner_with_regression_setup() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "name: Validate selected runner and enumerate shapes" in text
    assert "inputs.runner == 'self-hosted-e7k'" in text
    assert "fromJSON('[\"self-hosted\",\"Linux\",\"X64\",\"e7k\"]')" in text
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
    ):
        assert f"- name: {step}" in text


def test_execution_optimizer_controller_does_not_hold_selected_runner_and_emits_heartbeat() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "name: Dispatch and monitor execution shapes" in text
    assert "needs: prepare" in text
    assert "runs-on: ubuntu-latest" in text
    assert "[optimizer heartbeat]" in text
    assert "gh run watch \"$run_id\" --exit-status" in text
