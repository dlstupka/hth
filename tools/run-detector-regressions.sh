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
    PYTHONPATH=hth-pipeline python -m hth.detector_catalog list       --dir hth-pipeline/config/detectors       --automatic-only
  )
else
  detector_configs=("hth-pipeline/config/detectors/${DETECTOR_ALGORITHM,,}.json")
fi

if (( ${#detector_configs[@]} == 0 )); then
  echo "::error::No detector configurations were selected"
  exit 1
fi

# Detector lifecycle is part of the canonical detector executor, not workflow YAML.
# Prepare each unique detector exactly once before sharding/pipeline workers begin.
lifecycle_env="$OUTPUT_DIR/.detector-lifecycle.env"
: > "$lifecycle_env"
declare -A lifecycle_configs_seen=()
declare -a lifecycle_configs=()
for detector_config in "${detector_configs[@]}"; do
  detector_key="$(python - "$detector_config" <<'PYLIFECYCLE'
import json,sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("detector",""))
PYLIFECYCLE
  )"
  if [[ -z "$detector_key" ]]; then
    echo "::error::Detector configuration has no detector id: $detector_config"
    exit 1
  fi
  if [[ -z "${lifecycle_configs_seen[$detector_key]+x}" ]]; then
    lifecycle_configs_seen[$detector_key]=1
    lifecycle_configs+=("$detector_config")
    python -m hth.detector_lifecycle prepare-config \
      --config "$detector_config" \
      --results-root results-repo \
      --env-file "$lifecycle_env"
  fi
done
if [[ -s "$lifecycle_env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$lifecycle_env"
  set +a
fi

# Prove that Kraken PREPARE state is visible in the actual regression shell
# before any worker launches. During this debug cycle also load the bundled
# BLLA model here so import/model failures produce an ordinary traceback.
if [[ " ${detector_configs[*]} " == *"kraken_page_mask.json"* ]]; then
  echo "Kraken Page-Mask worker preflight"
  echo "================================="
  PYTHONFAULTHANDLER=1 python -m hth.kraken_page_mask_preflight --load-model
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
if [[ "${HTH_EXACT_EXECUTION_SHAPE:-0}" == "1" ]]; then
  runner_pipeline_max="${HTH_EXECUTION_THREAD_BUDGET:?HTH_EXECUTION_THREAD_BUDGET is required for exact execution shapes}"
else
  runner_pipeline_max="$(python - "${HTH_RUNNER_LABEL:-}" <<'PYMAX'
import sys
from hth.regression.sharding import runner_max_threads
print(runner_max_threads(sys.argv[1]))
PYMAX
  )"
fi
if [[ "$requested_pipelines" != "auto" ]] && { [[ ! "$requested_pipelines" =~ ^[0-9]+$ ]] || (( requested_pipelines < 1 || requested_pipelines > runner_pipeline_max )); }; then
  echo "::error::Detector pipelines must be auto or an integer from 1 through runner budget $runner_pipeline_max: $requested_pipelines"
  exit 1
fi
if [[ "$THREADS" != "auto" ]] && { [[ ! "$THREADS" =~ ^[0-9]+$ ]] || (( THREADS < 1 || THREADS > 1024 )); }; then
  echo "::error::Threads must be auto or an integer from 1 through 1024: $THREADS"
  exit 1
fi
if [[ "${DETECTOR_ALGORITHM,,}" != "all" ]]; then
  effective_pipelines=1
elif [[ "$requested_pipelines" == "auto" ]]; then
  effective_pipelines="$(python - "${#detector_configs[@]}" "$runner_pipeline_max" <<'PYAUTO'
import sys
from hth.domain.multidetector_schedule import plan_lpt_workers
print(plan_lpt_workers(int(sys.argv[1]), int(sys.argv[2])))
PYAUTO
  )"
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
  if [[ "${HTH_EXACT_EXECUTION_SHAPE:-0}" == "1" ]]; then
    planned_threads="$THREADS"
    planned_shards="${SHARDS:-1}"
    serial_estimate="unknown"
    plan_source="${HTH_EXACT_EXECUTION_SHAPE_SOURCE:-optimizer}-exact-shape"
  else
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
  fi
  # User limits and smoke runs are already bounded; do not shard them.
  if [[ "${HTH_EXACT_EXECUTION_SHAPE:-0}" != "1" ]] && { [[ "$REGRESSION_MODE" != "full" ]] || [[ -n "${effective_limit:-}" ]] || [[ "$effective_strategy" != "exhaustive" ]]; }; then
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
# Shard expansion can change the task count. For auto mode, the
# literal request has already been resolved into numeric effective_pipelines;
# clamp that resolved value rather than re-entering "auto" into arithmetic.
if [[ "$requested_pipelines" == "auto" ]]; then
  if (( effective_pipelines > ${#detector_configs[@]} )); then
    effective_pipelines=${#detector_configs[@]}
  fi
elif (( requested_pipelines > ${#detector_configs[@]} )); then
  effective_pipelines=${#detector_configs[@]}
else
  effective_pipelines=$requested_pipelines
fi

if [[ "${HTH_EXACT_EXECUTION_SHAPE:-0}" == "1" ]]; then
  runner_thread_budget="${HTH_EXECUTION_THREAD_BUDGET:?HTH_EXECUTION_THREAD_BUDGET is required for exact execution shapes}"
  effective_threads_per_pipeline="$THREADS"
  allocated_threads=$((effective_pipelines * effective_threads_per_pipeline))
  if (( allocated_threads > runner_thread_budget )) && [[ "${HTH_ALLOW_THREAD_OVERSUBSCRIPTION:-false}" != "true" ]]; then
    echo "::error::Exact optimizer shape requests ${allocated_threads} threads (${effective_pipelines}p x ${effective_threads_per_pipeline}t) against runner budget ${runner_thread_budget} without oversubscription enabled"
    exit 1
  fi
  if (( allocated_threads > runner_thread_budget )); then
    unused_threads=0
  else
    unused_threads=$((runner_thread_budget - allocated_threads))
  fi
else
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
fi
for ((task_index = 0; task_index < ${#task_threads[@]}; task_index++)); do
  task_threads[$task_index]="$effective_threads_per_pipeline"
done

echo "Regression mode    : $REGRESSION_MODE"
echo "Algorithm          : '$DETECTOR_ALGORITHM'"
echo "Effective limit    : '${effective_limit:-unlimited}' ($effective_limit_source)"
echo "Detectors          : $detector_count"
if [[ "$requested_pipelines" == "auto" ]]; then
  echo "Detector pipelines : $effective_pipelines (auto LPT; ${detector_count} detectors, ${runner_pipeline_max}t runner budget)"
else
  echo "Detector pipelines : $effective_pipelines (requested $requested_pipelines)"
fi
echo "Shards             : ${#detector_configs[@]}${SHARDS:+ (explicit request $SHARDS; capped at one parameter set per shard)}"
echo "Thread budget      : $allocated_threads allocated / $runner_thread_budget max; $unused_threads free; $effective_threads_per_pipeline per active pipeline"
if [[ "${HTH_EXACT_EXECUTION_SHAPE:-0}" == "1" ]]; then
  echo "Execution shape    : ${HTH_EXACT_EXECUTION_SHAPE_SOURCE:-optimizer}-exact (${effective_pipelines}p/${effective_threads_per_pipeline}t)"
  if [[ "$effective_threads_per_pipeline" != "$THREADS" ]]; then
    echo "::error::Exact shape requested ${THREADS} threads/pipeline but regression executor resolved ${effective_threads_per_pipeline}"
    exit 1
  fi
fi
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

CLAIM_BATCH_TARGET_DECISECONDS=100
CLAIM_ESTIMATE_FLOOR_DECISECONDS=1
initial_claim_strategy="dynamic-lpt"
batch_claims_enabled=0
if [[ "${DETECTOR_ALGORITHM,,}" == "all" ]] \
  && [[ "${DETECTOR_LOADING_STRATEGY,,}" == "lpt" ]] \
  && { [[ "$REGRESSION_MODE" != "full" ]] || [[ -n "${effective_limit:-}" ]] || [[ "$effective_strategy" != "exhaustive" ]]; } \
  && (( effective_pipelines > 1 )); then
  initial_claim_strategy="lpt-batches-10s"
  batch_claims_enabled=1
fi
echo "Claim strategy     : $initial_claim_strategy"

queue_dir="$OUTPUT_DIR/.detector-queue"
rm -rf "$queue_dir"
mkdir -p "$queue_dir/claims" "$queue_dir/done" "$queue_dir/failed" \
  "$queue_dir/run-dirs" "$queue_dir/logs" "$queue_dir/telemetry/workers" \
  "$queue_dir/telemetry/tasks" "$queue_dir/telemetry/claim-batches"
telemetry_root="$queue_dir/telemetry"
printf 'start\t%s\n' "$(date +%s.%N)" > "$telemetry_root/batch.tsv"

# Scheduling only needs decisecond precision for these short workloads.
mapfile -t task_claim_units < <(
  python - "${detector_estimates[@]}" <<'PYCLAIMUNITS'
import math
import sys
for raw in sys.argv[1:]:
    try:
        seconds=float(raw)
    except (TypeError,ValueError):
        seconds=0.1
    print(max(1,int(math.ceil(max(0.1,seconds)*10.0))))
PYCLAIMUNITS
)
: > "$OUTPUT_DIR/run-directories.txt"
: > "$OUTPUT_DIR/runner-output.log"

run_detector_config() {
  local task_index="$1"
  local pipeline_number="$2"
  local start_delay_seconds="${3:-0}"
  local claim_batch_id="${4:-unbatched}"
  local detector_config="${detector_configs[$task_index]}"

  if [[ ! -f "$detector_config" ]]; then
    echo "::error::Detector configuration not found: $detector_config"
    return 1
  fi

  local detector_name detector_estimate detector_estimate_source detector_ranked_quality
  local shard_index shard_count detector_threads shard_output shared_baseline
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
  shared_baseline="$OUTPUT_DIR/.baselines/$detector_name.json"
  mkdir -p "$shard_output" "$(dirname "$shared_baseline")"
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
  if (( shard_count > 1 )); then
    args+=(--shared-baseline "$shared_baseline")
  fi

  # Every regression evaluates the detector default baseline and the strongest
  # historic exact parameter set known before this run, independent of search.
  local historic_best_file
  historic_best_file="$OUTPUT_DIR/.references/$detector_name-task-$task_index-best.json"
  mkdir -p "$(dirname "$historic_best_file")"
  if [[ -f results-repo/calibration-index.json ]] && python -m hth.calibration_store resolve-best-parameter \
      --index results-repo/calibration-index.json \
      --detector "$detector_name" \
      --golden-set-sha256 "$golden_set_sha256" \
      > "$historic_best_file" 2>/dev/null; then
    echo "[pipeline $pipeline_number][$detector_name] Historic best reference: $historic_best_file"
    args+=(--historic-best "$historic_best_file")
  else
    rm -f "$historic_best_file"
    echo "[pipeline $pipeline_number][$detector_name] No reconstructable historic best reference found."
  fi

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
  printf 'start\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date +%s.%N)" "$pipeline_number" "$detector_name" "$shard_index" "$shard_count" "$detector_threads" "$claim_batch_id" \
    >> "$telemetry_root/tasks/$task_index.tsv"
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

  if [[ "$detector_name" == "kraken_page_mask" ]]; then
    echo "[pipeline $pipeline_number][$detector_name] Worker model path: ${HTH_KRAKEN_PAGE_MODEL:-<unset>}"
    echo "[pipeline $pipeline_number][$detector_name] Worker model exists: $([[ -f "${HTH_KRAKEN_PAGE_MODEL:-}" ]] && echo yes || echo no)"
    echo "[pipeline $pipeline_number][$detector_name] Worker provenance path: ${HTH_KRAKEN_PAGE_PROVENANCE:-<unset>}"
    echo "[pipeline $pipeline_number][$detector_name] Worker provenance exists: $([[ -f "${HTH_KRAKEN_PAGE_PROVENANCE:-}" ]] && echo yes || echo no)"
  fi

  if ! HTH_DETECTOR_PIPELINES="$effective_pipelines" \
    HTH_DETECTOR_PIPELINE_NUMBER="$pipeline_number" \
    HTH_PIPELINE_STAGGER_MINUTES="$PIPELINE_STAGGER_MINUTES" \
    HTH_DETECTOR_LOADING_STRATEGY="$DETECTOR_LOADING_STRATEGY" \
    HTH_DETECTOR_RUNTIME_ESTIMATE_SECONDS="$detector_estimate" \
    HTH_DETECTOR_RUNTIME_ESTIMATE_SOURCE="$detector_estimate_source" \
    HTH_DETECTOR_QUEUE_POSITION="$((task_index + 1))" \
    HTH_DETECTOR_RANKED_QUALITY="$detector_ranked_quality" \
    PYTHONFAULTHANDLER=1 \
    "${args[@]}" 2>&1 \
      | sed -u -E 's/^(Machine[[:space:]]*:).*/\1 [obfuscated]/' \
      | sed -u "s/^/[pipeline $pipeline_number][$detector_name] /" \
      | tee "$task_log"; then
    kill "$lease_pid" 2>/dev/null || true
    wait "$lease_pid" 2>/dev/null || true
    detector_finished_epoch="$(date +%s)"
    printf 'finish\t%s\tfailed\n' "$(date +%s.%N)" >> "$telemetry_root/tasks/$task_index.tsv"
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
  printf 'finish\t%s\tcomplete\n' "$(date +%s.%N)" >> "$telemetry_root/tasks/$task_index.tsv"
  detector_wall_seconds="$((detector_finished_epoch - detector_started_epoch))"

  if [[ -n "${HTH_OPTIMIZER_SHARD_LOG:-}" && -n "${HTH_OPTIMIZER_RUN_ID:-}" && -n "${HTH_OPTIMIZER_SHAPE_SEQUENCE:-}" ]]; then
    optimizer_shard_args=(
      --results-root results-repo
      --shard-run-dir "$run_dir"
      --shard-wall-clock-seconds "$detector_wall_seconds"
      --runner-label "${HTH_RUNNER_LABEL:-unknown}"
      --github-run-id "$HTH_OPTIMIZER_RUN_ID"
      --shape-sequence "$HTH_OPTIMIZER_SHAPE_SEQUENCE"
      --pipeline-number "$pipeline_number"
      --shard-index "$shard_index"
      --shard-count "$shard_count"
      --threads "$detector_threads"
      --shard-log "$HTH_OPTIMIZER_SHARD_LOG"
    )
    if [[ -n "${HTH_OPTIMIZER_RUNNER_METRICS_LOG:-}" ]]; then
      optimizer_shard_args+=(--runner-metrics-log "$HTH_OPTIMIZER_RUNNER_METRICS_LOG")
    fi
    python -m hth.optimizer_capture "${optimizer_shard_args[@]}"
  fi

  lifecycle_time="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "======================================================================"
  echo "[pipeline $pipeline_number] UNLOAD detector=$detector_name shard=$((shard_index + 1))/$shard_count status=complete wall=${detector_wall_seconds}s time=$lifecycle_time"
  echo "[pipeline $pipeline_number] Completed; pipeline is taking the next queued detector."
  echo "======================================================================"
}

claim_batch_from_queue() {
  claimed_batch=()
  claimed_batch_units=0
  local task_index claim_dir units
  for ((task_index=0; task_index<${#detector_configs[@]}; task_index++)); do
    [[ -f "$queue_dir/done/$task_index" ]] && continue
    [[ -f "$queue_dir/failed/$task_index" ]] && continue
    claim_dir="$queue_dir/claims/$task_index"
    if mkdir "$claim_dir" 2>/dev/null; then
      claimed_batch+=("$task_index")
      units="${task_claim_units[$task_index]:-$CLAIM_ESTIMATE_FLOOR_DECISECONDS}"
      claimed_batch_units=$((claimed_batch_units + units))
      if (( batch_claims_enabled == 0 || claimed_batch_units >= CLAIM_BATCH_TARGET_DECISECONDS )); then
        break
      fi
    fi
  done
}

reclaim_expired_task() {
  claimed_batch=()
  claimed_batch_units=0
  local task_index claim_dir
  for ((task_index=0; task_index<${#detector_configs[@]}; task_index++)); do
    [[ -f "$queue_dir/done/$task_index" ]] && continue
    [[ -f "$queue_dir/failed/$task_index" ]] && continue
    claim_dir="$queue_dir/claims/$task_index"
    [[ -f "$claim_dir/lease.json" ]] || continue
    if python - "$claim_dir/lease.json" <<'PYLEASE'
from pathlib import Path
import sys
from hth.regression.sharding import lease_expired
raise SystemExit(0 if lease_expired(Path(sys.argv[1])) else 1)
PYLEASE
    then
      echo "[pipeline recovery] Reclaiming expired shard lease $task_index."
      rm -rf "$claim_dir"
      if mkdir "$claim_dir" 2>/dev/null; then
        claimed_batch=("$task_index")
        claimed_batch_units="${task_claim_units[$task_index]:-$CLAIM_ESTIMATE_FLOOR_DECISECONDS}"
        break
      fi
    fi
  done
}

write_claim_batch_telemetry() {
  local pipeline_number="$1" batch_sequence="$2" claimed_at="$3"
  shift 3
  local -a batch_tasks=("$@")
  local batch_id task_csv first_task
  batch_id="$(printf 'p%03d-b%05d' "$pipeline_number" "$batch_sequence")"
  task_csv="$(IFS=,; echo "${batch_tasks[*]}")"
  first_task="${batch_tasks[0]}"
  printf 'claim_batch\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$claimed_at" "$pipeline_number" "${task_threads[$first_task]}" \
    "$((claimed_batch_units/10)).$((claimed_batch_units%10))" "$task_csv" "$batch_id" \
    > "$telemetry_root/claim-batches/$batch_id.tsv"
  printf '%s\n' "$batch_id"
}

detector_worker() {
  local pipeline_index="$1" seeded_batch_csv="${2:-}"
  local pipeline_number=$((pipeline_index+1))
  local delay_seconds=$((pipeline_index*PIPELINE_STAGGER_MINUTES*60))
  local task_index task_start_delay claim_batch_id claimed_at
  local batch_sequence=0
  local claim_lock_fd="" claim_lock_dir="$queue_dir/.claim-lock"
  local -a claimed_batch=()
  local claimed_batch_units=0

  # Every background worker opens the lock independently so flock owns a
  # distinct open-file description per process. Only batch assignment is
  # serialized; detector loading and execution remain fully parallel.
  if command -v flock >/dev/null 2>&1; then
    exec {claim_lock_fd}>"$queue_dir/claim.lock"
  fi

  acquire_claim_lock() {
    if [[ -n "$claim_lock_fd" ]]; then flock -x "$claim_lock_fd"; return; fi
    while ! mkdir "$claim_lock_dir" 2>/dev/null; do sleep 0.01; done
  }
  release_claim_lock() {
    if [[ -n "$claim_lock_fd" ]]; then flock -u "$claim_lock_fd"; else rmdir "$claim_lock_dir"; fi
  }

  local first_task=1
  printf 'start\t%s\n' "$(date +%s.%N)" > "$telemetry_root/workers/$pipeline_number.tsv"
  echo "[pipeline $pipeline_number] Started."

  while true; do
    claimed_batch=()
    claimed_batch_units=0
    if [[ -n "$seeded_batch_csv" ]]; then
      IFS=',' read -r -a claimed_batch <<< "$seeded_batch_csv"
      seeded_batch_csv=""
      for task_index in "${claimed_batch[@]}"; do
        claimed_batch_units=$((claimed_batch_units + ${task_claim_units[$task_index]:-$CLAIM_ESTIMATE_FLOOR_DECISECONDS}))
      done
      echo "[pipeline $pipeline_number] Seeded initial LPT batch tasks=$(IFS=,; echo "${claimed_batch[*]}")"
    else
      acquire_claim_lock
      claim_batch_from_queue
      (( ${#claimed_batch[@]} == 0 )) && reclaim_expired_task
      release_claim_lock
    fi

    if (( ${#claimed_batch[@]} == 0 )); then
      printf 'end\t%s\n' "$(date +%s.%N)" >> "$telemetry_root/workers/$pipeline_number.tsv"
      echo "[pipeline $pipeline_number] Detector queue empty."
      return 0
    fi

    batch_sequence=$((batch_sequence+1))
    claimed_at="$(date +%s.%N)"
    claim_batch_id="$(write_claim_batch_telemetry "$pipeline_number" "$batch_sequence" "$claimed_at" "${claimed_batch[@]}")"
    echo "[pipeline $pipeline_number] CLAIM-BATCH id=$claim_batch_id tasks=${#claimed_batch[@]} estimate=$((claimed_batch_units/10)).$((claimed_batch_units%10))s"

    for task_index in "${claimed_batch[@]}"; do
      task_start_delay=0
      if (( first_task == 1 )); then task_start_delay="$delay_seconds"; first_task=0; fi
      if run_detector_config "$task_index" "$pipeline_number" "$task_start_delay" "$claim_batch_id"; then
        : > "$queue_dir/done/$task_index"
      else
        : > "$queue_dir/failed/$task_index"
        printf 'end\t%s\n' "$(date +%s.%N)" >> "$telemetry_root/workers/$pipeline_number.tsv"
        echo "::error::Pipeline $pipeline_number failed detector config ${detector_configs[$task_index]}"
        return 1
      fi
    done
  done
}

worker_pids=()
declare -a initial_seed_batches=()
if (( batch_claims_enabled == 1 )); then
  echo "Initial LPT claim batches"
  echo "========================="
  for ((pipeline_index=0; pipeline_index<effective_pipelines; pipeline_index++)); do
    claim_batch_from_queue
    (( ${#claimed_batch[@]} == 0 )) && break
    initial_seed_batches[$pipeline_index]="$(IFS=,; echo "${claimed_batch[*]}")"
    echo "pipeline=$((pipeline_index+1)) tasks=${initial_seed_batches[$pipeline_index]} estimate=$((claimed_batch_units/10)).$((claimed_batch_units%10))s"
  done
fi

for ((pipeline_index=0; pipeline_index<effective_pipelines; pipeline_index++)); do
  detector_worker "$pipeline_index" "${initial_seed_batches[$pipeline_index]:-}" &
  worker_pids+=("$!")
done

queue_failed=0
for worker_pid in "${worker_pids[@]}"; do
  if ! wait "$worker_pid"; then
    queue_failed=1
  fi
done
printf 'end\t%s\n' "$(date +%s.%N)" >> "$telemetry_root/batch.tsv"

if [[ "${DETECTOR_ALGORITHM,,}" == "all" ]]; then
  multidetector_observation_id="${GITHUB_RUN_ID:-local}:${GITHUB_RUN_ATTEMPT:-1}:${REGRESSION_MODE}:${effective_strategy}:${detector_count}"
  python -m hth.multidetector_store finalize \
    --telemetry-root "$telemetry_root" \
    --output "$OUTPUT_DIR/multidetector-execution.json" \
    --observation-id "$multidetector_observation_id" \
    --github-run-id "${GITHUB_RUN_ID:-}" \
    --github-run-number "${GITHUB_RUN_NUMBER:-}" \
    --mode "$REGRESSION_MODE" \
    --strategy "$effective_strategy" \
    --limit "${effective_limit:-}" \
    --detector-count "$detector_count" \
    --golden-set-sha256 "$golden_set_sha256" \
    --runner-label "${HTH_RUNNER_LABEL:-unknown}" \
    --runner-name "${HTH_RUNNER_NAME:-${RUNNER_NAME:-unknown}}" \
    --runner-thread-budget "$runner_thread_budget" \
    --threads-per-worker "$effective_threads_per_pipeline" \
    --allocated-threads "$allocated_threads" \
    --loading-strategy "$DETECTOR_LOADING_STRATEGY" \
    --claim-strategy "$initial_claim_strategy" \
    --scheduler-source "${HTH_EXACT_EXECUTION_SHAPE_SOURCE:-${requested_pipelines}}"
fi

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
    shard_run_dir="$(find "$shard_root" -mindepth 1 -maxdepth 1 -type d -name 'run-*' 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -z "$shard_run_dir" ]]; then
      echo "::error::Missing completed shard $((shard_index + 1))/$expected_detector_shards for $detector_name"
      exit 1
    fi
    detector_shard_dirs+=("$shard_run_dir")
  done
  detector_config="hth-pipeline/config/detectors/$detector_name.json"
  finalization_root="$OUTPUT_DIR/.finalize/$detector_name"
  rm -rf "$finalization_root"
  mkdir -p "$finalization_root"

  if (( ${#detector_shard_dirs[@]} == 1 )); then
    canonical_run="${detector_shard_dirs[0]}"
    staging_root="$(dirname "$(dirname "$canonical_run")")"
  else
    merge_args=()
    for shard_dir in "${detector_shard_dirs[@]}"; do
      merge_args+=(--shard-dir "$shard_dir")
    done
    canonical_run="$(python -m hth.regression.merge_shards \
      "${merge_args[@]}" \
      --expected-shard-count "$expected_detector_shards" \
      --output "$finalization_root" \
      --detector-config "$detector_config" \
      --golden-set "hth-pipeline/$GOLDEN_SET" \
      --image-root "results-repo/$IMAGE_ROOT" \
      --max-dimension "$MAX_DIMENSION" \
      --debug-level "$DEBUG_LEVEL" \
      --top "$TOP_COUNT")"
    staging_root="$finalization_root"
  fi

  finalized_dir="$(python -m hth.regression.finalize_run \
    --canonical-run "$canonical_run" \
    --staging-root "$staging_root" \
    --output "$OUTPUT_DIR" \
    --detector "$detector_name")"
  printf '%s\n' "$finalized_dir" >> "$OUTPUT_DIR/run-directories.txt"
  rm -rf "$finalization_root"
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

# Finalize each detector exactly once after all of its shards/runs are complete.
for detector_config in "${lifecycle_configs[@]}"; do
  python -m hth.detector_lifecycle finalize-config \
    --config "$detector_config" \
    --results-root results-repo
done

rm -rf "$queue_dir"
echo "run_dirs_file=$OUTPUT_DIR/run-directories.txt" >> "$GITHUB_OUTPUT"
