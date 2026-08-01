from __future__ import annotations

from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "regress-detector.yml"


def test_multidetector_pipeline_inputs_and_defaults_are_declared() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "detector_pipelines:" in text
    assert "default: \"4\"" in text
    for option in ("1", "2", "4", "8"):
        assert f'          - "{option}"' in text
    assert "pipeline_stagger_minutes:" in text
    assert 'default: "0"' in text


def test_detector_queue_is_dynamic_and_records_pipeline_metadata() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'mkdir "$queue_dir/claims/$task_index"' in text
    assert "Completed; pipeline is taking the next queued detector." in text
    assert 'HTH_DETECTOR_PIPELINES="$effective_pipelines"' in text
    assert 'HTH_DETECTOR_PIPELINE_NUMBER="$pipeline_number"' in text
    assert 'HTH_PIPELINE_STAGGER_MINUTES="$PIPELINE_STAGGER_MINUTES"' in text
    assert 'HTH_DETECTOR_QUEUE_POSITION="$((task_index + 1))"' in text
    assert 'HTH_DETECTOR_RANKED_QUALITY="$detector_ranked_quality"' in text


def test_single_detector_runs_force_one_pipeline() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'if [[ "${DETECTOR_ALGORITHM,,}" != "all" ]]; then' in text
    assert "effective_pipelines=1" in text


def test_loading_strategies_runtime_index_and_announcements_are_wired() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "detector_loading_strategy:" in text
    assert "default: lpt" in text
    for strategy in ("lpt", "fifo", "ranked"):
        assert f"          - {strategy}" in text
    assert "python -m hth.runtime_store order" in text
    assert "results-repo/runtime-index.json" in text
    assert "LOAD detector=$detector_name" in text
    assert "UNLOAD detector=$detector_name status=complete" in text
    assert "git -C results-repo add calibration-index.json runtime-index.json source-documents/" in text
