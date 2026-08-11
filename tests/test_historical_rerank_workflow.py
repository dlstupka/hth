from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "rebuild-historical-regression.yml"
REGRESSION = ROOT / ".github" / "workflows" / "regress-detector.yml"


def test_historical_rerank_workflow_is_manual_and_uses_raw_artifacts():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "single-build" in text
    assert "all-available-artifacts" in text
    assert "gh run download" in text
    assert "raw/results.csv" in text
    assert "python -m hth.historical_rerank" in text
    assert "--results-root results-repo" in text


def test_core_workflow_change_triggers_detector_smoke():
    text = REGRESSION.read_text(encoding="utf-8")
    assert text.count('".github/workflows/_core-hth.yml"') >= 2


    def test_artifact_download_uses_explicit_repository(self):
        workflow = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "rebuild-historical-regression-metrics.yml"
        text = workflow.read_text(encoding="utf-8")
        self.assertIn('gh run download "$run_id" --repo "$GITHUB_REPOSITORY"', text)


def test_historical_rerank_filters_pre_intelligence_runs_before_download():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "ddbd063fdfd72319d42266cd1b2e02f078d9e7c3" in text
    assert "merge-base --is-ancestor" in text
    assert ".head_sha" in text
    assert "Skipping pre-calibration-intelligence regression build" in text
    assert '"$eligible_runs" > "$run_ids"' in text
