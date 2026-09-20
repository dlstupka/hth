from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_regression_shape_policy_is_not_reimplemented_in_yaml():
    text = (ROOT / ".github/workflows/regress-detector.yml").read_text(encoding="utf-8")
    assert "python -m hth.regression_shape workflow-resolve" in text
    assert "use_exact_shape()" not in text
    assert 'case "$shape_mode"' not in text


def test_historical_artifact_policy_is_not_reimplemented_in_yaml():
    text = (ROOT / ".github/workflows/rebuild-historical-regression.yml").read_text(encoding="utf-8")
    assert "python -m hth.historical_rebuild_workflow" in text
    assert "gh run download" not in text
    assert "no space left on device" not in text


def test_immutable_release_creation_is_not_reimplemented_in_yaml():
    helper = (ROOT / "tools/hardened-release.sh").read_text(encoding="utf-8")
    assert "gh release create" in helper
    for workflow in (ROOT / ".github/workflows").glob("*.yml"):
        text = workflow.read_text(encoding="utf-8")
        assert "gh release create" not in text, workflow.name


def test_every_executable_job_has_a_bounded_timeout():
    for workflow in (ROOT / ".github/workflows").glob("*.yml"):
        lines = workflow.read_text(encoding="utf-8").splitlines()
        jobs: list[tuple[int, str]] = []
        in_jobs = False
        for index, line in enumerate(lines):
            if line == "jobs:":
                in_jobs = True
                continue
            if in_jobs and line and not line.startswith(" "):
                in_jobs = False
            if (
                in_jobs
                and len(line) - len(line.lstrip()) == 2
                and line.rstrip().endswith(":")
            ):
                jobs.append((index, line.strip()[:-1]))
        for position, (start, name) in enumerate(jobs):
            end = jobs[position + 1][0] if position + 1 < len(jobs) else len(lines)
            block = "\n".join(lines[start:end])
            if "runs-on:" in block:
                assert "timeout-minutes:" in block, f"{workflow.name}:{name}"
