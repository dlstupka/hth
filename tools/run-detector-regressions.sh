#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${HTH_REGRESSION_OUTPUT:-regression-output}"
mkdir -p "$OUTPUT_DIR"
export HTH_SOURCE_COMMIT="$(git -C results-repo rev-parse HEAD)"

effective_limit="${LIMIT:-}"
if [[ -n "$effective_limit" ]]; then
  effective_limit_source="user specified"
elif [[ "$REGRESSION_MODE" == "smoke" ]]; then
  effective_limit="10"
  effective_limit_source="smoke default"
else
  effective_limit_source="unlimited"
fi

effective_strategy="$STRATEGY"
if [[ -n "${effective_limit:-}" ]]; then
  effective_strategy="exhaustive"
fi

if [[ "${DETECTOR_ALGORITHM,,}" == "all" ]]; then
  mapfile -t detector_configs < <(
    find hth-pipeline/config/detectors \
      -maxdepth 1 \
      -type f \
      -name '*.json' \
      -print | sort
  )
else
  detector_configs=("hth-pipeline/config/detectors/${DETECTOR_ALGORITHM,,}.json")
fi

if (( ${#detector_configs[@]} == 0 )); then
  echo "::error::No detector configurations were selected"
  exit 1
fi

if [[ ! "${PIPELINE_STAGGER_MINUTES:-0}" =~ ^[0-9]+$ ]]; then
  echo "::error::Pipeline stagger must be a non-negative whole number of minutes: ${PIPELINE_STAGGER_MINUTES:-}"
  exit 1
fi
if [[ ! "${DETECTOR_LOADING_STRATEGY:-lpt}" =~ ^(lpt|fifo|ranked)$ ]]; then
  echo "::error::Detector loading strategy must be lpt, fifo, or ranked: ${DETECTOR_LOADING_STRATEGY:-}"
  exit 1
fi

requested_pipelines="$DETECTOR_PIPELINES"
runner_pipeline_max="$(python - "${HTH_RUNNER_LABEL:-}" <<'PYMAX'
import sys
from hth.regression.sharding import runner_max_threads
print(runner_max_threads(sys.argv[1]))
PYMAX
)"
if [[ ! "$requested_pipelines" =~ ^[0-9]+$ ]] || (( requested_pipelines < 1 || requested_pipelines > runner_pipeline_max )); then
  echo "::error::Detector pipelines must be an integer from 1 through runner budget $runner_pipeline_max: $requested_pipelines"
  exit 1
fi
if [[ "$THREADS" != "auto" ]] && { [[ ! "$THREADS" =~ ^[0-9]+$ ]] || (( THREADS < 1 || THREADS > 1024 )); }; then
  echo "::error::Threads must be auto or an integer from 1 through 1024: $THREADS"
  exit 1
fi
if [[ "${DETECTOR_ALGORITHM,,}" != "all" ]]; then
  effective_pipelines=1
elif (( requested_pipelines > ${#detector_configs[@]} )); then
  effective_pipelines=${#detector_configs[@]}
else
  effective_pipelines=$requested_pipelines
fi

golden_set_sha256="$(python - "hth-pipeline/$GOLDEN_SET" <<'PY'
import hashlib
import sys
from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)"

order_threads="$THREADS"
[[ "$order_threads" == "auto" ]] && order_threads="1"
declare -a ordered_configs=() detector_estimates=() detector_estimate_sources=() detector_quality=()
if [[ "${DETECTOR_ALGORITHM,,}" == "all" ]]; then
  order_args=()
  for detector_config in "${detector_configs[@]}"; do
    order_args+=(--config "$detector_config")
  done
  while IFS=$'\t' read -r ordered_config estimate estimate_source quality; do
    [[ -n "$ordered_config" ]] || continue
    ordered_configs+=("$ordered_config")
    detector_estimates+=("$estimate")
    detector_estimate_sources+=("$estimate_source")
    detector_quality+=("$quality")
  done < <(python -m hth.runtime_store order \
    "${order_args[@]}" \
    --loading-strategy "$DETECTOR_LOADING_STRATEGY" \
    --runtime-index results-repo/runtime-index.json \
    --calibration-index results-repo/calibration-index.json \
    --mode "$REGRESSION_MODE" \
    --search-strategy "$effective_strategy" \
    --threads "$order_threads" \
    --max-dimension "$MAX_DIMENSION" \
    --golden-set-sha256 "$golden_set_sha256" \
    --runner-label "${HTH_RUNNER_LABEL:-}")
  detector_configs=("${ordered_configs[@]}")
else
  detector_estimates=("unknown")
  detector_estimate_sources=("single-detector")
  detector_quality=("unknown")
fi

if [[ -n "${SHARDS:-}" && ! "${SHARDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::Shards must be a positive whole number or blank for wall-clock planning"
  exit 1
fi
if [[ ! "${SHARD_TARGET_MINUTES:-30}" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::Shard target must be a positive whole number of minutes"
  exit 1
fi
if [[ ! "${SHARD_LEASE_MINUTES:-5}" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::Shard lease must be a positive whole number of minutes"
  exit 1
fi

detector_count=${#detector_configs[@]}
declare -a task_configs=() task_estimates=() task_estimate_sources=() task_quality=()
declare -a task_shard_indexes=() task_shard_counts=() task_threads=() task_detectors=()
for ((detector_index = 0; detector_index < ${#detector_configs[@]}; detector_index++)); do
  detector_config="${detector_configs[$detector_index]}"
  detector_name="$(python - "$detector_config" <<'PYPLAN'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["detector"])
PYPLAN
  )"
  read -r planned_threads planned_shards serial_estimate plan_source < <(
    python - "$detector_config" "results-repo/runtime-index.json" "$detector_name" "${HTH_RUNNER_LABEL:-}" "$THREADS" "$SHARD_TARGET_MINUTES" "${SHARDS:-}" <<'PYPLAN'
import json, sys
from pathlib import Path
from hth.regression.sharding import best_smoke_observation, estimate_serial_runtime, plan_shards
from hth.regression.strategies.cartesian import generate
config_path, index_path, detector, runner_label, requested_threads, target_minutes, requested_shards = sys.argv[1:]
config = json.loads(Path(config_path).read_text(encoding="utf-8"))
possible = len(generate(config))
observation = best_smoke_observation(Path(index_path), detector)
serial = estimate_serial_runtime(observation, possible) if observation else None
plan = plan_shards(serial, runner_label=runner_label, requested_threads=requested_threads, target_shard_seconds=int(target_minutes) * 60, maximum_shards=possible, requested_shards=int(requested_shards) if requested_shards else None, possible_parameter_sets=possible, estimate_source="smoke-runtime-index" if observation else "no-smoke-history")
print(plan.threads, plan.shard_count, "unknown" if serial is None else f"{serial:.3f}", plan.estimate_source)
PYPLAN
  )
  # User limits and smoke runs are already bounded; do not shard them.
  if [[ "$REGRESSION_MODE" != "full" || -n "${effective_limit:-}" || "$effective_strategy" != "exhaustive" ]]; then
    planned_shards=1
  fi
  for ((shard_index = 0; shard_index < planned_shards; shard_index++)); do
    task_configs+=("$detector_config")
    task_estimates+=("${detector_estimates[$detector_index]:-$serial_estimate}")
    task_estimate_sources+=("${detector_estimate_sources[$detector_index]:-$plan_source}")
    task_quality+=("${detector_quality[$detector_index]:-unknown}")
    task_shard_indexes+=("$shard_index")
    task_shard_counts+=("$planned_shards")
    task_threads+=("$planned_threads")
    task_detectors+=("$detector_name")
  done
done
detector_configs=("${task_configs[@]}")
detector_estimates=("${task_estimates[@]}")
detector_estimate_sources=("${task_estimate_sources[@]}")
detector_quality=("${task_quality[@]}")
if (( requested_pipelines > ${#detector_configs[@]} )); then
  effective_pipelines=${#detector_configs[@]}
else
  effective_pipelines=$requested_pipelines
fi

read -r runner_thread_budget effective_threads_per_pipeline allocated_threads unused_threads < <(
  python - "$THREADS" "${HTH_RUNNER_LABEL:-}" "$effective_pipelines" <<'PYBUDGET'
import sys
from hth.regression.sharding import plan_execution
plan = plan_execution(
    sys.argv[1],
    runner_label=sys.argv[2],
    active_pipelines=int(sys.argv[3]),
)
print(
    plan.runner_thread_budget,
    plan.threads_per_pipeline,
    plan.allocated_threads,
    plan.unused_threads,
)
PYBUDGET
)
for ((task_index = 0; task_index < ${#task_threads[@]}; task_index++)); do
  task_threads[$task_index]="$effective_threads_per_pipeline"
done

echo "Regression mode    : $REGRESSION_MODE"
echo "Algorithm          : '$DETECTOR_ALGORITHM'"
echo "Effective limit    : '${effective_limit:-unlimited}' ($effective_limit_source)"
echo "Detectors          : $detector_count"
echo "Detector pipelines : $effective_pipelines (requested $requested_pipelines)"
echo "Shards             : ${#detector_configs[@]}${SHARDS:+ (explicit request $SHARDS; capped at one parameter set per shard)}"
echo "Thread budget      : $runner_thread_budget total; $effective_threads_per_pipeline per active pipeline; $allocated_threads allocated; $unused_threads free"
echo "Loading strategy   : ${DETECTOR_LOADING_STRATEGY}"
echo "Pipeline stagger   : ${PIPELINE_STAGGER_MINUTES} minute(s)"
echo
echo "Detector queue"
echo "=============="
for ((queue_index = 0; queue_index < ${#detector_configs[@]}; queue_index++)); do
  queue_detector="$(python - "${detector_configs[$queue_index]}" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("detector", "unknown"))
PY
  )"
  printf '%2d. %-24s shard=%s/%s threads=%s estimate=%-10s source=%s quality=%s\n' \
    "$((queue_index + 1))" "$queue_detector" \
    "$((task_shard_indexes[$queue_index] + 1))" "${task_shard_counts[$queue_index]}" "${task_threads[$queue_index]}" \
    "${detector_estimates[$queue_index]:-unknown}s" \
    "${detector_estimate_sources[$queue_index]:-unknown}" \
    "${detector_quality[$queue_index]:-unknown}"
done

queue_dir="$OUTPUT_DIR/.detector-queue"
rm -rf "$queue_dir"
mkdir -p "$queue_dir/claims" "$queue_dir/done" "$queue_dir/failed" \
  "$queue_dir/run-dirs" "$queue_dir/logs"
: > "$OUTPUT_DIR/run-directories.txt"
: > "$OUTPUT_DIR/runner-output.log"

run_detector_config() {
  local task_index="$1"
  local pipeline_number="$2"
  local start_delay_seconds="${3:-0}"
  local detector_config="${detector_configs[$task_index]}"

  if [[ ! -f "$detector_config" ]]; then
    echo "::error::Detector configuration not found: $detector_config"
    return 1
  fi

  local detector_name detector_estimate detector_estimate_source detector_ranked_quality
  local shard_index shard_count detector_threads shard_output
  shard_index="${task_shard_indexes[$task_index]}"
  shard_count="${task_shard_counts[$task_index]}"
  detector_threads="${task_threads[$task_index]}"
  detector_name="$(python - "$detector_config" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("detector", "unknown"))
PY
  )"
  shard_output="$OUTPUT_DIR/.shards/$detector_name/shard-$(printf '%04d' "$shard_index")"
  mkdir -p "$shard_output"
  detector_estimate="${detector_estimates[$task_index]:-unknown}"
  detector_estimate_source="${detector_estimate_sources[$task_index]:-unknown}"
  detector_ranked_quality="${detector_quality[$task_index]:-unknown}"

  local -a args=(
    python -m hth.regress_detector
    --detector-config "$detector_config"
    --golden-set "hth-pipeline/$GOLDEN_SET"
    --image-root "results-repo/$IMAGE_ROOT"
    --output "$shard_output"
    --max-dimension "$MAX_DIMENSION"
    --top "$TOP_COUNT"
    --threads "$detector_threads"
    --shard-index "$shard_index"
    --shard-count "$shard_count"
    --debug-level "$DEBUG_LEVEL"
  )

  if [[ -n "${effective_limit:-}" ]]; then
    args+=(--strategy exhaustive --limit "$effective_limit")
  else
    args+=(--strategy "$STRATEGY")
    if [[ "$STRATEGY" =~ ^(non-dormant|low\+|moderate\+|important\+|critical)$ ]]; then
      local golden_set_sha256 detector_config_sha256 calibration_file
      golden_set_sha256="$(python - "hth-pipeline/$GOLDEN_SET" <<'PY'
import hashlib
import sys
from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
      )"
      detector_config_sha256="$(python - "$detector_config" <<'PY'
import hashlib
import sys
from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
      )"
      calibration_file=""
      if [[ -f results-repo/calibration-index.json ]]; then
        calibration_file="$(python -m hth.calibration_store resolve \
          --index results-repo/calibration-index.json \
          --detector "$detector_name" \
          --golden-set-sha256 "$golden_set_sha256" \
          --detector-config-sha256 "$detector_config_sha256" 2>/dev/null || true)"
      fi
      if [[ -n "$calibration_file" && -f "$calibration_file" ]]; then
        echo "[pipeline $pipeline_number][$detector_name] Using indexed calibration intelligence: $calibration_file"
        args+=(--calibration-intelligence "$calibration_file")
      else
        echo "[pipeline $pipeline_number][$detector_name] No compatible indexed calibration intelligence found; runner will fall back to exhaustive."
      fi
    fi
  fi

  local detector_loaded_epoch detector_started_epoch detector_finished_epoch detector_wall_seconds
  local lifecycle_time
  detector_loaded_epoch="$(date +%s)"
  lifecycle_time="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo
  echo "======================================================================"
  echo "[pipeline $pipeline_number] LOAD detector=$detector_name shard=$((shard_index + 1))/$shard_count threads=$detector_threads time=$lifecycle_time"
  echo "[pipeline $pipeline_number] estimate=${detector_estimate}s source=${detector_estimate_source}"
  echo "[pipeline $pipeline_number] ranked_quality=${detector_ranked_quality}"
  if (( start_delay_seconds > 0 )); then
    echo "[pipeline $pipeline_number] WAIT detector=$detector_name shard=$((shard_index + 1))/$shard_count stagger=${start_delay_seconds}s"
    sleep "$start_delay_seconds"
  fi
  detector_started_epoch="$(date +%s)"
  lifecycle_time="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "[pipeline $pipeline_number] START detector=$detector_name shard=$((shard_index + 1))/$shard_count threads=$detector_threads time=$lifecycle_time"
  echo "======================================================================"
  echo "[pipeline $pipeline_number][$detector_name] Regression command:"
  printf '[pipeline %s][%s]' "$pipeline_number" "$detector_name"
  printf ' %q' "${args[@]}"
  printf '\n'

  echo "[pipeline $pipeline_number][$detector_name] Regression Invocation"
  printf '[pipeline %s][%s] %-22s: %s\n' "$pipeline_number" "$detector_name" "Mode" "$REGRESSION_MODE"
  printf '[pipeline %s][%s] %-22s: %s\n' "$pipeline_number" "$detector_name" "Detector" "$detector_name"
  printf '[pipeline %s][%s] %-22s: %s\n' "$pipeline_number" "$detector_name" "Detector config" "$detector_config"
  printf '[pipeline %s][%s] %-22s: %s\n' "$pipeline_number" "$detector_name" "Strategy" "$effective_strategy"
  printf '[pipeline %s][%s] %-22s: %s\n' "$pipeline_number" "$detector_name" "Threads" "$detector_threads"
  printf '[pipeline %s][%s] %-22s: %s\n' "$pipeline_number" "$detector_name" "Debug level" "$DEBUG_LEVEL"
  printf '[pipeline %s][%s] %-22s: %s\n' "$pipeline_number" "$detector_name" "Detector pipeline" "$pipeline_number of $effective_pipelines"
  if [[ -n "$effective_limit" ]]; then
    printf '[pipeline %s][%s] %-22s: %s (%s)\n' \
      "$pipeline_number" "$detector_name" "Parameter-set limit" "$effective_limit" "$effective_limit_source"
  else
    printf '[pipeline %s][%s] %-22s: %s\n' "$pipeline_number" "$detector_name" "Parameter-set limit" "unlimited"
  fi

  local before_runs task_log run_dir lease_file lease_pid
  before_runs="$(find "$shard_output/$detector_name" -mindepth 1 -maxdepth 1 -type d -name 'run-*' 2>/dev/null | sort || true)"
  task_log="$queue_dir/logs/$(printf '%04d' "$task_index")-$detector_name-shard-$shard_index.log"
  lease_file="$queue_dir/claims/$task_index/lease.json"
  renew_lease() {
    while true; do
      python - "$lease_file" "$pipeline_number" "$SHARD_LEASE_MINUTES" "$shard_index/$shard_count" <<'PYLEASE'
from pathlib import Path
import sys
from hth.regression.sharding import write_lease
write_lease(Path(sys.argv[1]), owner=f"pipeline-{sys.argv[2]}", ttl_seconds=int(sys.argv[3]) * 60, shard=sys.argv[4])
PYLEASE
      sleep 60
    done
  }
  renew_lease &
  lease_pid=$!

  if ! HTH_DETECTOR_PIPELINES="$effective_pipelines" \
    HTH_DETECTOR_PIPELINE_NUMBER="$pipeline_number" \
    HTH_PIPELINE_STAGGER_MINUTES="$PIPELINE_STAGGER_MINUTES" \
    HTH_DETECTOR_LOADING_STRATEGY="$DETECTOR_LOADING_STRATEGY" \
    HTH_DETECTOR_RUNTIME_ESTIMATE_SECONDS="$detector_estimate" \
    HTH_DETECTOR_RUNTIME_ESTIMATE_SOURCE="$detector_estimate_source" \
    HTH_DETECTOR_QUEUE_POSITION="$((task_index + 1))" \
    HTH_DETECTOR_RANKED_QUALITY="$detector_ranked_quality" \
    "${args[@]}" 2>&1 \
      | sed -u -E 's/^(Machine[[:space:]]*:).*/\1 [obfuscated]/' \
      | sed -u "s/^/[pipeline $pipeline_number][$detector_name] /" \
      | tee "$task_log"; then
    kill "$lease_pid" 2>/dev/null || true
    wait "$lease_pid" 2>/dev/null || true
    detector_finished_epoch="$(date +%s)"
    detector_wall_seconds="$((detector_finished_epoch - detector_started_epoch))"
    lifecycle_time="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "[pipeline $pipeline_number] UNLOAD detector=$detector_name shard=$((shard_index + 1))/$shard_count status=failed wall=${detector_wall_seconds}s time=$lifecycle_time"
    echo "::error::$detector_name regression command failed in pipeline $pipeline_number"
    return 1
  fi
  kill "$lease_pid" 2>/dev/null || true
  wait "$lease_pid" 2>/dev/null || true

  run_dir="$(
    comm -13 \
      <(printf '%s\n' "$before_runs" | sed '/^$/d' | sort) \
      <(find "$shard_output/$detector_name" -mindepth 1 -maxdepth 1 -type d -name 'run-*' | sort) \
    | tail -n 1
  )"

  if [[ -z "$run_dir" ]]; then
    echo "::error::$detector_name regression did not create a canonical run directory"
    return 1
  fi

  printf '%s\n' "$run_dir" > "$queue_dir/run-dirs/$(printf '%04d' "$task_index")"
  detector_finished_epoch="$(date +%s)"
  detector_wall_seconds="$((detector_finished_epoch - detector_started_epoch))"
  lifecycle_time="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "======================================================================"
  echo "[pipeline $pipeline_number] UNLOAD detector=$detector_name shard=$((shard_index + 1))/$shard_count status=complete wall=${detector_wall_seconds}s time=$lifecycle_time"
  echo "[pipeline $pipeline_number] Completed; pipeline is taking the next queued detector."
  echo "======================================================================"
}

detector_worker() {
  local pipeline_index="$1"
  local pipeline_number=$((pipeline_index + 1))
  local delay_seconds=$((pipeline_index * PIPELINE_STAGGER_MINUTES * 60))
  local task_index claimed task_start_delay

  local first_task=1
  echo "[pipeline $pipeline_number] Started."

  while true; do
    claimed=""
    for ((task_index = 0; task_index < ${#detector_configs[@]}; task_index++)); do
      [[ -f "$queue_dir/done/$task_index" ]] && continue
      claim_dir="$queue_dir/claims/$task_index"
      if mkdir "$claim_dir" 2>/dev/null; then
        claimed="$task_index"
        break
      fi
      if [[ -f "$claim_dir/lease.json" ]] && python - "$claim_dir/lease.json" <<'PYLEASE'
from pathlib import Path
import sys
from hth.regression.sharding import lease_expired
raise SystemExit(0 if lease_expired(Path(sys.argv[1])) else 1)
PYLEASE
      then
        echo "[pipeline $pipeline_number] Reclaiming expired shard lease $task_index."
        rm -rf "$claim_dir"
        if mkdir "$claim_dir" 2>/dev/null; then
          claimed="$task_index"
          break
        fi
      fi
    done

    if [[ -z "$claimed" ]]; then
      echo "[pipeline $pipeline_number] Detector queue empty."
      return 0
    fi

    task_start_delay=0
    if (( first_task == 1 )); then
      task_start_delay="$delay_seconds"
      first_task=0
    fi
    if run_detector_config "$claimed" "$pipeline_number" "$task_start_delay"; then
      : > "$queue_dir/done/$claimed"
    else
      : > "$queue_dir/failed/$claimed"
      echo "::error::Pipeline $pipeline_number failed detector config ${detector_configs[$claimed]}"
      return 1
    fi
  done
}

worker_pids=()
for ((pipeline_index = 0; pipeline_index < effective_pipelines; pipeline_index++)); do
  detector_worker "$pipeline_index" &
  worker_pids+=("$!")
done

queue_failed=0
for worker_pid in "${worker_pids[@]}"; do
  if ! wait "$worker_pid"; then
    queue_failed=1
  fi
done

: > "$OUTPUT_DIR/run-directories.txt"
mapfile -t unique_detectors < <(printf '%s\n' "${task_detectors[@]}" | sort -u)
for detector_name in "${unique_detectors[@]}"; do
  mapfile -t detector_task_indexes < <(
    for ((task_index = 0; task_index < ${#task_detectors[@]}; task_index++)); do
      [[ "${task_detectors[$task_index]}" == "$detector_name" ]] && printf '%s\n' "$task_index"
    done
  )
  first_detector_task_index="${detector_task_indexes[0]}"
  expected_detector_shards="${task_shard_counts[$first_detector_task_index]}"
  detector_shard_dirs=()
  for ((shard_index = 0; shard_index < expected_detector_shards; shard_index++)); do
    shard_root="$OUTPUT_DIR/.shards/$detector_name/shard-$(printf '%04d' "$shard_index")/$detector_name"
    shard_run_dir="$(find "$shard_root" -mindepth 1 -maxdepth 1 -type d -name 'run-*' | sort | tail -n 1)"
    if [[ -z "$shard_run_dir" ]]; then
      echo "::error::Missing completed shard $((shard_index + 1))/$expected_detector_shards for $detector_name"
      exit 1
    fi
    detector_shard_dirs+=("$shard_run_dir")
  done
  detector_config="hth-pipeline/config/detectors/$detector_name.json"
  if (( ${#detector_shard_dirs[@]} == 1 )); then
    source_dir="${detector_shard_dirs[0]}"
    mkdir -p "$OUTPUT_DIR/$detector_name"
    target_dir="$OUTPUT_DIR/$detector_name/$(basename "$source_dir")"
    rm -rf "$target_dir"
    mv "$source_dir" "$target_dir"
    printf '%s\n' "$target_dir" >> "$OUTPUT_DIR/run-directories.txt"
  else
    merge_args=()
    for shard_dir in "${detector_shard_dirs[@]}"; do merge_args+=(--shard-dir "$shard_dir"); done
    merged_dir="$(python -m hth.regression.merge_shards "${merge_args[@]}" --expected-shard-count "$expected_detector_shards" --output "$OUTPUT_DIR" --detector-config "$detector_config" --golden-set "hth-pipeline/$GOLDEN_SET" --image-root "results-repo/$IMAGE_ROOT" --max-dimension "$MAX_DIMENSION" --debug-level "$DEBUG_LEVEL" --top "$TOP_COUNT")"
    printf '%s\n' "$merged_dir" >> "$OUTPUT_DIR/run-directories.txt"
  fi
done
while IFS= read -r queue_file; do
  cat "$queue_file"
done < <(find "$queue_dir/logs" -maxdepth 1 -type f -name '*.log' | sort) \
  > "$OUTPUT_DIR/runner-output.log"

completed_count="$(find "$queue_dir/done" -maxdepth 1 -type f | wc -l | tr -d ' ')"
failed_count="$(find "$queue_dir/failed" -maxdepth 1 -type f | wc -l | tr -d ' ')"
echo "Detector queue complete: $completed_count completed, $failed_count failed."

if (( queue_failed != 0 || failed_count != 0 || completed_count != ${#detector_configs[@]} )); then
  echo "::error::Detector pipeline queue did not complete successfully"
  exit 1
fi

rm -rf "$queue_dir"
echo "run_dirs_file=$OUTPUT_DIR/run-directories.txt" >> "$GITHUB_OUTPUT"
