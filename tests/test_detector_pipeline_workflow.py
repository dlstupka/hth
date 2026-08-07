from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "regress-detector.yml"
DRIVER = ROOT / "tools" / "run-detector-regressions.sh"


def test_multidetector_pipeline_inputs_and_defaults_are_declared() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "detector_pipelines:" in text
    assert "default: \"4\"" in text
    assert "type: string" in text
    assert "any integer from 1 through 1024" in text
    assert "pipeline_stagger_minutes:" in text
    assert 'default: "0"' in text


def test_detector_queue_is_dynamic_and_records_pipeline_metadata() -> None:
    text = DRIVER.read_text(encoding="utf-8")
    assert 'mkdir "$claim_dir"' in text
    assert "Completed; pipeline is taking the next queued detector." in text
    assert 'HTH_DETECTOR_PIPELINES="$effective_pipelines"' in text
    assert 'HTH_DETECTOR_PIPELINE_NUMBER="$pipeline_number"' in text
    assert 'HTH_PIPELINE_STAGGER_MINUTES="$PIPELINE_STAGGER_MINUTES"' in text
    assert 'HTH_DETECTOR_QUEUE_POSITION="$((task_index + 1))"' in text
    assert 'HTH_DETECTOR_RANKED_QUALITY="$detector_ranked_quality"' in text


def test_single_detector_runs_begin_with_one_pipeline_before_shard_expansion() -> None:
    text = DRIVER.read_text(encoding="utf-8")
    assert 'if [[ "${DETECTOR_ALGORITHM,,}" != "all" ]]; then' in text
    assert "effective_pipelines=1" in text


def test_loading_strategies_runtime_index_and_announcements_are_wired() -> None:
    text = DRIVER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "detector_loading_strategy:" in workflow
    assert "default: lpt" in workflow
    for strategy in ("lpt", "fifo", "ranked"):
        assert f"          - {strategy}" in workflow
    assert "python -m hth.runtime_store order" in text
    assert "results-repo/runtime-index.json" in text
    assert "LOAD detector=$detector_name shard=$((shard_index + 1))/$shard_count" in text
    assert "START detector=$detector_name shard=$((shard_index + 1))/$shard_count" in text
    assert "UNLOAD detector=$detector_name shard=$((shard_index + 1))/$shard_count status=complete" in text
    assert "time=$lifecycle_time" in text
    assert "git -C results-repo add calibration-index.json runtime-index.json parallelism-index.json source-documents/" in workflow


def test_intelligence_publisher_rebuilds_from_latest_results_state_on_retry() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'max_publish_attempts=5' in text
    assert 'git -C results-repo fetch origin main' in text
    assert 'git -C results-repo reset --hard origin/main' in text
    assert 'git -C results-repo clean -fd -- calibration-index.json runtime-index.json parallelism-index.json source-documents/' in text
    assert 'python -m hth.calibration_store publish' in text
    assert 'git -C results-repo push origin HEAD:main' in text
    assert 'git -C results-repo pull --rebase origin main' not in text
    assert 'retry_delay=$((5 * (attempt - 1)))' in text
    assert 'Publish collision detected; waiting ${retry_delay}s for calibration intelligence to free up before retrying...' in text


def test_manual_debug_level_choices_default_to_none_and_are_forwarded() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    driver = DRIVER.read_text(encoding="utf-8")
    assert "debug_level:" in text
    assert "description: Debug image detail for manual builds" in text
    assert "default: none" in text
    for level in ("none", "basic", "verbose"):
        assert f"          - {level}" in text
    assert "DEBUG_LEVEL:" in text
    assert '--debug-level "$DEBUG_LEVEL"' in driver
    assert '"Debug level" "$DEBUG_LEVEL"' in driver


def test_auto_threads_shards_and_expiring_leases_are_wired() -> None:
    text = DRIVER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "default: auto" in workflow
    assert "shards:" in workflow
    assert "explicit shard count" in workflow
    assert "SHARDS:" in workflow
    assert "shard_target_minutes:" in workflow
    assert "shard_lease_minutes:" in workflow
    assert "from hth.regression.sharding import best_smoke_observation" in text
    assert '--shard-index "$shard_index"' in text
    assert '--shard-count "$shard_count"' in text
    assert "Reclaiming expired shard lease" in text
    assert "python -m hth.regression.merge_shards" in text
    assert 'runner_label=runner_label' in text
    assert 'requested_shards=int(requested_shards) if requested_shards else None' in text
    assert 'possible_parameter_sets=possible' in text


def test_execution_summary_and_merge_use_canonical_detector_shard_and_budget_counts() -> None:
    text = DRIVER.read_text(encoding="utf-8")
    assert 'detector_count=${#detector_configs[@]}' in text
    assert 'echo "Detectors          : $detector_count"' in text
    assert 'echo "Shards             : ${#detector_configs[@]}${SHARDS:+ (explicit request $SHARDS; capped at one parameter set per shard)}"' in text
    assert 'from hth.regression.sharding import plan_execution' in text
    assert 'task_threads[$task_index]="$effective_threads_per_pipeline"' in text
    assert 'rm -rf "$queue_dir"' in text
    assert '--expected-shard-count "$expected_detector_shards"' in text
    assert 'Missing completed shard' in text


def test_runner_thread_budget_profile_survives_runner_diagnostics() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "HTH_RUNNER_LABEL: >-" in text
    assert "|| 'github-hosted' }}" in text
    assert 'echo "HTH_RUNNER_LABEL=${HTH_RUNNER_LABEL}" >> "$GITHUB_ENV"' in text
    assert 'echo "HTH_RUNNER_LABEL=${{ inputs.execution_target }}" >> "$GITHUB_ENV"' not in text


def test_detector_pipeline_validation_accepts_any_integer_within_runner_budget() -> None:
    text = DRIVER.read_text(encoding="utf-8")
    assert "Detector pipelines must be one of 1, 2, 4, or 8" not in text
    assert 'if [[ ! "$requested_pipelines" =~ ^[0-9]+$ ]] || (( requested_pipelines < 1 || requested_pipelines > runner_pipeline_max )); then' in text
    assert 'Detector pipelines must be an integer from 1 through runner budget $runner_pipeline_max' in text
