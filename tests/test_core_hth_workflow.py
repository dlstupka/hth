from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "_core-hth.yml"


def test_core_supports_serial_execution_optimizer() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "prepare-execution-optimizer:" in text
    assert "if: ${{ inputs.mode == 'execution-optimizer' }}" in text
    assert "inputs.optimizer_runner == 'self-hosted-e7k'" in text
    assert "from hth.regression.sharding import runner_max_threads" in text
    assert "Run execution optimizer experiments" in text
    assert "one job, one runner, serial execution shapes" in text


def test_core_exposes_execution_optimizer_plan_outputs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "optimizer_runner_label:" in text
    assert "optimizer_runner_budget:" in text
    assert "optimizer_pipelines:" in text
    assert "jobs.prepare-execution-optimizer.outputs.runner_budget" in text


def test_core_accepts_execution_optimizer_workload_inputs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    for name in (
        "optimizer_algorithm:",
        "optimizer_golden_set:",
        "optimizer_image_root:",
        "optimizer_max_dimension:",
        "optimizer_strategy:",
        "optimizer_debug_level:",
    ):
        assert name in text
