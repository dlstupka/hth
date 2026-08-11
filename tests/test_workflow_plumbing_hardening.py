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
