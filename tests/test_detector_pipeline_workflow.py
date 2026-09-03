from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "regress-detector.yml"
DRIVER = ROOT / "tools" / "run-detector-regressions.sh"


def test_execution_shape_inputs_replace_raw_pipeline_and_thread_knobs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "execution_shape:" in text
    assert "default: preferred" in text
    for mode in ("preferred", "auto", "manual"):
        assert f"          - {mode}" in text
    assert "manual_execution_shape:" in text
    assert "Manual execution shape, e.g. 8p/48t" in text
    assert "detector_pipelines:" not in text
    assert "threads:" not in text
    assert "pipeline_stagger_minutes:" in text
    assert 'default: "0"' in text


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
    assert "results-repo/indexes/runtime-index.json" in text
    assert "LOAD detector=$detector_name shard=$((shard_index + 1))/$shard_count" in text
    assert "START detector=$detector_name shard=$((shard_index + 1))/$shard_count" in text
    assert "UNLOAD detector=$detector_name shard=$((shard_index + 1))/$shard_count status=complete" in text
    assert "time=$lifecycle_time" in text
    assert "calibration-index.json" in workflow
    assert "parameter-provenance-index.json" in workflow
    assert "runtime-index.json" in workflow
    assert "parallelism-index.json" in workflow
    assert "source-documents/" in workflow
    assert "git -C results-repo add models/" not in workflow


def test_intelligence_publisher_uses_shared_hardened_persistence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "source hth-pipeline/tools/hardened-persistence.sh" in text
    assert "hth_hardened_persist" in text
    assert "apply_calibration_intelligence" in text
    assert "python -m hth.calibration_store publish" in text


def test_manual_debug_level_choices_default_to_none_and_are_forwarded() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    driver = DRIVER.read_text(encoding="utf-8")
    assert "debug_level:" in text
    assert 'description: "Advanced — debug image detail for manual builds"' in text
    assert "default: none" in text
    for level in ("none", "basic", "verbose"):
        assert f"          - {level}" in text
    assert "DEBUG_LEVEL:" in text
    assert '--debug-level "$DEBUG_LEVEL"' in driver
    assert '"Debug level" "$DEBUG_LEVEL"' in driver


def test_auto_threads_and_runtime_shard_planning_are_wired() -> None:
    text = DRIVER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "default: preferred" in workflow
    assert "THREADS: auto" in workflow
    assert "shard_target_minutes:" in workflow
    assert "shard_lease_minutes:" in workflow
    assert "from hth.regression.sharding import best_smoke_observation" in text
    assert '--shard-index "$shard_index"' in text
    assert '--shard-count "$shard_count"' in text
    assert "python -m hth.regression.merge_shards" in text
    assert 'runner_label=runner_label' in text
    assert 'requested_shards=None' in text
    assert 'possible_parameter_sets=possible' in text


def test_execution_summary_and_merge_use_canonical_detector_shard_and_budget_counts() -> None:
    text = DRIVER.read_text(encoding="utf-8")
    assert 'detector_count=${#detector_configs[@]}' in text
    assert 'echo "Detectors          : $detector_count"' in text
    assert 'echo "Sharding           : auto (runtime target ${SHARD_TARGET_MINUTES}m)"' in text
    assert 'echo "Sharding           : ${sharding_policy} shard(s) / active pipeline"' in text
    assert 'from hth.regression.sharding import plan_execution' in text
    assert 'task_threads[$task_index]="$effective_threads_per_pipeline"' in text
    assert 'rm -rf "$queue_dir"' in text
    assert '--expected-shard-count "$expected_detector_shards"' in text
    assert 'Missing completed shard' in text


def test_preferred_shape_resolution_falls_back_to_auto_and_exact_shape_is_explicit() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "Resolve regression execution shape" in text
    assert "python -m hth.regression_shape workflow-resolve" in text
    assert "optimizer-predictions.json" in text
    assert "--github-env" in text
    assert "--parallelism-index" in text
    assert "--predictions-index" in text

def test_all_without_exhaustive_is_first_and_dispatches_missing_authoritative_runs() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "default: all-without-exhaustive" in text
    assert text.index("          - all-without-exhaustive\n") < text.index("          - all\n")
    assert "dispatch-missing-exhaustive:" in text
    assert "python -m hth.regression_dispatch" in text
    assert "inputs.algorithm != 'all-without-exhaustive'" in text
    assert "actions: write" in text
    concurrency = text.split("concurrency:", 1)[1].split("jobs:", 1)[0]
    assert "github.event_name == 'workflow_dispatch'" in concurrency
    assert "format('manual-{0}', github.run_id)" in concurrency
    assert "|| 'automatic'" in concurrency
    assert "inputs.algorithm" not in concurrency



def test_regression_output_is_cleaned_before_each_matrix_run():
    text = WORKFLOW.read_text(encoding="utf-8")
    cleanup = text.index("- name: Reset regression output workspace")
    execute = text.index("- name: Run detector regressions")
    assert cleanup < execute
    block = text[cleanup:execute]
    assert "rm -rf regression-output" in block
    assert "rm -f regression-summary.md" in block




def test_single_and_multi_shard_paths_share_one_finalizer():
    script = (ROOT / "tools" / "run-detector-regressions.sh").read_text(encoding="utf-8")
    assert script.count("python -m hth.regression.finalize_run") == 1
    assert '--output "$finalization_root"' in script
    assert 'staging_root="$(dirname "$(dirname "$canonical_run")")"' in script
    assert '--staging-root "$staging_root"' in script
    assert '--output "$OUTPUT_DIR"' in script


def test_zero_valid_measurement_failures_are_aggregated_after_finalization():
    script = (ROOT / "tools" / "run-detector-regressions.sh").read_text(encoding="utf-8")
    collect = script.index('invalid_detectors+=("$detector_name")')
    publish_output = script.index('echo "run_dirs_file=$OUTPUT_DIR/run-directories.txt" >> "$GITHUB_OUTPUT"')
    terminal_failure = script.index('Regression failed: no valid measurements for detector(s)')
    assert collect < publish_output < terminal_failure
    assert 'state.get("status", "unknown")' in script

def test_parallel_shards_share_one_run_local_baseline_cache() -> None:
    text = DRIVER.read_text(encoding="utf-8")
    assert 'shared_baseline="$OUTPUT_DIR/.baselines/$detector_name.json"' in text
    assert 'if (( shard_count > 1 )); then' in text
    assert 'args+=(--shared-baseline "$shared_baseline")' in text


def test_results_repository_checkout_is_shallow_and_sparse() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "fetch-depth: 1" in text
    assert "sparse-checkout-cone-mode: false" in text
    assert "            calibration-index.json\n" in text
    assert "            parallelism-index.json\n" in text
    assert "            ${{ env.GOLDEN_RELEASE_TAG == '' && env.IMAGE_ROOT || '' }}\n" in text
    assert "Materialize immutable Golden Set images" in text
    assert "python -m hth.golden_set_release" in text
    assert "            source-documents\n" in text
    assert "            learned-evidence/orli_page_mask\n" in text
