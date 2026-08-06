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
    assert 'mkdir "$claim_dir"' in text
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


def test_intelligence_publisher_rebuilds_from_latest_results_state_on_retry() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'max_publish_attempts=5' in text
    assert 'git -C results-repo fetch origin main' in text
    assert 'git -C results-repo reset --hard origin/main' in text
    assert 'git -C results-repo clean -fd -- calibration-index.json runtime-index.json source-documents/' in text
    assert 'python -m hth.calibration_store publish' in text
    assert 'git -C results-repo push origin HEAD:main' in text
    assert 'git -C results-repo pull --rebase origin main' not in text
    assert 'retry_delay=$((5 * (attempt - 1)))' in text
    assert 'Publish collision detected; waiting ${retry_delay}s for calibration intelligence to free up before retrying...' in text


def test_manual_debug_level_choices_default_to_none_and_are_forwarded() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "debug_level:" in text
    assert "description: Debug image detail for manual builds" in text
    assert "default: none" in text
    for level in ("none", "basic", "verbose"):
        assert f"          - {level}" in text
    assert "DEBUG_LEVEL:" in text
    assert '--debug-level "$DEBUG_LEVEL"' in text
    assert '"Debug level" "$DEBUG_LEVEL"' in text


def test_auto_threads_shards_and_expiring_leases_are_wired() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "default: auto" in text
    assert "shard_target_minutes:" in text
    assert "shard_lease_minutes:" in text
    assert "from hth.regression.sharding import best_smoke_observation" in text
    assert '--shard-index "$shard_index"' in text
    assert '--shard-count "$shard_count"' in text
    assert "Reclaiming expired shard lease" in text
    assert "python -m hth.regression.merge_shards" in text
    assert 'runner_label=runner_label' in text


def test_execution_summary_and_merge_use_canonical_detector_shard_and_budget_counts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'detector_count=${#detector_configs[@]}' in text
    assert 'echo "Detectors          : $detector_count"' in text
    assert 'echo "Shards             : ${#detector_configs[@]}"' in text
    assert 'from hth.regression.sharding import budgeted_threads' in text
    assert 'rm -rf "$queue_dir" regression-output/.shards' in text
    assert '--expected-shard-count "$expected_detector_shards"' in text
    assert 'Missing completed shard' in text
