from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "rebuild-historical-regression.yml"
REGRESSION = ROOT / ".github" / "workflows" / "regress-detector.yml"


def test_historical_workflow_delegates_resolution_and_streaming_to_python():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "python -m hth.historical_rebuild_workflow resolve" in text
    assert "python -m hth.historical_rebuild_workflow stream" in text
    assert "gh run download" not in text
    assert "merge-base --is-ancestor" not in text


def test_historical_floor_remains_explicit():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "ddbd063fdfd72319d42266cd1b2e02f078d9e7c3" in text


def test_core_workflow_change_triggers_detector_smoke():
    text = REGRESSION.read_text(encoding="utf-8")
    assert text.count('".github/workflows/_core-hth.yml"') >= 2
