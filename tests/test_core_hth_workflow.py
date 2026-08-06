from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "_core-hth.yml"



def test_core_supports_execution_optimizer_runner_validation() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "prepare-execution-optimizer:" in text
    assert "if: ${{ inputs.mode == 'execution-optimizer' }}" in text
    assert "inputs.optimizer_runner == 'self-hosted-e7k'" in text
    assert "from hth.regression.sharding import runner_max_threads" in text
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
        assert f"- name: {step}" in text
    assert "[hth core heartbeat]" in text


def test_core_exposes_execution_optimizer_plan_outputs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "optimizer_runner_label:" in text
    assert "optimizer_runner_budget:" in text
    assert "optimizer_pipelines:" in text
    assert "jobs.prepare-execution-optimizer.outputs.runner_budget" in text
