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


def test_execution_optimizer_uses_runner_budget_candidates_from_core() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    core = (WORKFLOW.parent / "_core-hth.yml").read_text(encoding="utf-8")
    assert "uses: ./.github/workflows/_core-hth.yml" in text
    assert "from hth.regression.sharding import runner_max_threads" in core
    assert "preferred = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192]" in core
    assert "Manual pipeline range must satisfy" in core


def test_execution_optimizer_publishes_optimizer_table_heatmap_and_index() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m hth.optimizer_store" in text
    assert "optimizer-index.json" in text
    assert 'execution-optimizer/$ALGORITHM/summary.md' in text
    assert 'execution-optimizer/$ALGORITHM/heatmap.svg' in text
    assert "Execution optimizer heat map" in text


def test_execution_optimizer_uses_reusable_hth_core_for_runner_validation() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "name: Validate selected runner and enumerate shapes" in text
    assert "uses: ./.github/workflows/_core-hth.yml" in text
    assert "mode: execution-optimizer" in text
    assert "optimizer_runner: ${{ inputs.runner }}" in text
    assert "optimizer_pipeline_range: ${{ inputs.pipeline_range }}" in text
    assert "optimizer_pipeline_min: ${{ inputs.pipeline_min }}" in text
    assert "optimizer_pipeline_max: ${{ inputs.pipeline_max }}" in text
    assert "Runner diagnostics" not in text
    assert "Benchmark OpenCV" not in text


def test_execution_optimizer_controller_does_not_hold_selected_runner_and_emits_heartbeat() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "name: Dispatch and monitor execution shapes" in text
    assert "needs: prepare" in text
    assert "runs-on: ubuntu-latest" in text
    assert "[optimizer heartbeat]" in text
    assert "gh run watch \"$run_id\" --exit-status" in text
