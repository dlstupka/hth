#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${HTH_REGRESSION_OUTPUT:-regression-output}"
mkdir -p "$OUTPUT_DIR"
executor_started_epoch="$(date +%s.%N)"
export HTH_SOURCE_COMMIT="$(git -C results-repo rev-parse HEAD)"

# Golden Set releases are materialized outside the results checkout, while the
# explicit legacy fallback remains relative to that checkout. Resolve this once
# so every image consumer sees the same canonical directory.
if [[ "$IMAGE_ROOT" == /* || "$IMAGE_ROOT" =~ ^[A-Za-z]:[\\/] ]]; then
  regression_image_root="$IMAGE_ROOT"
else
  regression_image_root="results-repo/$IMAGE_ROOT"
fi
if [[ ! -d "$regression_image_root" ]]; then
  echo "::error::Golden Set image root does not exist: $regression_image_root"
  exit 1
fi
echo "Golden Set image root: $regression_image_root"

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
if [[ -n "${effective_limit:-}" && "$STRATEGY" != "adaptive" && "${HTH_BOUNDED_WORKLOAD:-0}" != "1" ]]; then
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
    lifecycle_prepare_started="$(date +%s.%N)"
    python -m hth.detector_lifecycle prepare-config \
      --config "$detector_config" \
      --results-root results-repo \
      --env-file "$lifecycle_env"
    lifecycle_prepare_finished="$(date +%s.%N)"
    lifecycle_prepare_elapsed="$(python - "$lifecycle_prepare_started" "$lifecycle_prepare_finished" <<'PYLIFECYCLETIME'
import sys
print(f"{max(0.0, float(sys.argv[2]) - float(sys.argv[1])):.3f}")
PYLIFECYCLETIME
    )"
    echo "Detector lifecycle prepare timing detector=${detector_key} elapsed=${lifecycle_prepare_elapsed}s"
  fi
done
if [[ -s "$lifecycle_env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$lifecycle_env"
  set +a
fi

# Prove that Kraken PREPARE state is visible in the actual regression shell
# before any worker launches. Model inference itself is deliberately deferred
# to the one parent shared-evidence producer when a learned detector is sharded.
if [[ " ${detector_configs[*]} " == *"kraken_page_mask.json"* ]]; then
  echo "Kraken Page-Mask worker preflight"
  echo "================================="
  PYTHONFAULTHANDLER=1 python -m hth.kraken_page_mask_preflight
fi
if [[ " ${detector_configs[*]} " == *"doc_ufcn_page_mask.json"* ]]; then
  echo "Doc-UFCN Page-Mask worker preflight"
  echo "==================================="
  PYTHONFAULTHANDLER=1 python -m hth.doc_ufcn_page_mask_preflight
fi
if [[ " ${detector_configs[*]} " == *"mask_rcnn_page_mask.json"* ]]; then
  echo "Mask R-CNN Page-Mask worker preflight"
  echo "====================================="
  PYTHONFAULTHANDLER=1 python -m hth.mask_rcnn_page_mask_preflight
fi
if [[ " ${detector_configs[*]} " == *"eynollah_page_mask.json"* ]]; then
  echo "Eynollah Page-Mask worker preflight"
  echo "==================================="
  PYTHONFAULTHANDLER=1 python -m hth.eynollah_page_mask_preflight
fi
if [[ " ${detector_configs[*]} " == *"docextractor_page_mask.json"* ]]; then
  echo "docExtractor Page-Mask worker preflight"
  echo "======================================="
  PYTHONFAULTHANDLER=1 python -m hth.docextractor_page_mask_preflight
fi
if [[ " ${detector_configs[*]} " == *"pagenet_page_mask.json"* ]]; then
  echo "PageNet Page-Mask worker preflight"
  echo "=================================="
  PYTHONFAULTHANDLER=1 python -m hth.pagenet_page_mask_preflight
fi
if [[ " ${detector_configs[*]} " == *"orli_page_mask.json"* ]]; then
  echo "Running Orli Page-Mask worker preflight after lifecycle environment load."
  orli_preflight_started="$(date +%s.%N)"
  PYTHONFAULTHANDLER=1 python -m hth.orli_page_mask_preflight
  orli_preflight_finished="$(date +%s.%N)"
  orli_preflight_elapsed="$(python - "$orli_preflight_started" "$orli_preflight_finished" <<'PYORLIPREFLIGHTTIME'
import sys
print(f"{max(0.0, float(sys.argv[2]) - float(sys.argv[1])):.3f}")
PYORLIPREFLIGHTTIME
  )"
  echo "Orli Page-Mask preflight timing: ${orli_preflight_elapsed}s"
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
if [[ "${HTH_EXACT_EXECUTION_SHAPE:-0}" == "1" && "$requested_pipelines" != "auto" ]]; then
  # Exact optimizer/preferred shapes are an atomic pipelines x threads contract.
  # A single detector can and must fan out across multiple shards/pipelines when
  # replaying an optimizer-selected execution shape.
  effective_pipelines="$requested_pipelines"
elif [[ "${DETECTOR_ALGORITHM,,}" != "all" ]]; then
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
    --runtime-index results-repo/indexes/runtime-index.json \
    --calibration-index results-repo/indexes/calibration-index.json \
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

sharding_policy="${SHARDING:-auto}"
if [[ "$sharding_policy" != "auto" && ! "$sharding_policy" =~ ^[1-9][0-9]*$ ]]; then
  echo "::error::Sharding must be 'auto' or a positive whole number of shards per active pipeline: $sharding_policy"
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
    serial_estimate="unknown"
    plan_source="${HTH_EXACT_EXECUTION_SHAPE_SOURCE:-optimizer}-exact-shape"
  else
    read -r planned_threads auto_planned_shards serial_estimate plan_source < <(
      python - "$detector_config" "results-repo/indexes/runtime-index.json" "$detector_name" "${HTH_RUNNER_LABEL:-}" "$THREADS" "$SHARD_TARGET_MINUTES" <<'PYPLAN'
import json, sys
from pathlib import Path
from hth.regression.sharding import best_smoke_observation, estimate_serial_runtime, plan_shards
from hth.regression.strategies.cartesian import generate
config_path, index_path, detector, runner_label, requested_threads, target_minutes = sys.argv[1:]
config = json.loads(Path(config_path).read_text(encoding="utf-8"))
possible = len(generate(config))
observation = best_smoke_observation(Path(index_path), detector)
serial = estimate_serial_runtime(observation, possible) if observation else None
plan = plan_shards(serial, runner_label=runner_label, requested_threads=requested_threads, target_shard_seconds=int(target_minutes)*60, maximum_shards=possible, requested_shards=None, possible_parameter_sets=possible, estimate_source="smoke-runtime-index" if observation else "no-smoke-history")
print(plan.threads, plan.shard_count, "unknown" if serial is None else f"{serial:.3f}", plan.estimate_source)
PYPLAN
    )
  fi

  if [[ "$sharding_policy" == "auto" ]]; then
    if [[ "$effective_strategy" == "binary-refine" || "$effective_strategy" == "adaptive" ]]; then
      # Feedback-directed searches cannot be partitioned into independent shards.
      planned_shards=1
      plan_source="${effective_strategy}-single-shard"
    elif (( detector_count == 1 )); then
      # A single-detector run uses one work shard per active pipeline by
      # default.  This keeps the work topology aligned with the execution
      # shape chosen by the optimizer/planner instead of deriving an unrelated
      # shard count from a coarse runtime estimate.
      planned_shards="$effective_pipelines"
      possible_shards="$(python - "$detector_config" <<'PYSHARDCAP'
import json, sys
from pathlib import Path
from hth.regression.strategies.cartesian import generate
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(max(1, len(generate(config))))
PYSHARDCAP
      )"
      if (( planned_shards > possible_shards )); then
        planned_shards="$possible_shards"
        plan_source="auto-one-shard-per-pipeline-capped-to-parameter-space"
      else
        plan_source="auto-one-shard-per-pipeline"
      fi
      if [[ "${HTH_EXACT_EXECUTION_SHAPE:-0}" == "1" ]]; then
        serial_estimate="unknown"
        plan_source="${HTH_EXACT_EXECUTION_SHAPE_SOURCE:-optimizer}-${plan_source}"
      fi
    else
      # In multi-detector runs the active pipelines are detector workers, so
      # each detector remains one task; multiplying every detector by the full
      # worker count would overshard the aggregate queue.
      planned_shards=1
      plan_source="multi-detector-single-shard"
    fi
  else
    shard_pipeline_count="$effective_pipelines"
    if [[ "$requested_pipelines" != "auto" ]]; then
      shard_pipeline_count="$requested_pipelines"
    fi
    planned_shards=$((shard_pipeline_count * sharding_policy))
    possible_shards="$(python - "$detector_config" <<'PYSHARDCAP'
import json, sys
from pathlib import Path
from hth.regression.strategies.cartesian import generate
config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(max(1, len(generate(config))))
PYSHARDCAP
    )"
    if (( planned_shards > possible_shards )); then
      planned_shards="$possible_shards"
      plan_source="explicit-${sharding_policy}-shards-per-pipeline-capped-to-parameter-space"
    else
      plan_source="explicit-${sharding_policy}-shards-per-pipeline"
    fi
  fi
  if [[ "$effective_strategy" == "adaptive" && "$planned_shards" != "1" ]]; then
    planned_shards=1
    plan_source="adaptive-single-shard"
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

if [[ "${HTH_EXACT_EXECUTION_SHAPE:-0}" == "1" \
  && "${exhaustive_shardable:-0}" == "1" \
  && "$requested_pipelines" != "auto" \
  && "$effective_pipelines" != "$requested_pipelines" ]]; then
  echo "::error::Exact execution shape requested ${requested_pipelines} pipelines but executor resolved ${effective_pipelines} after shard expansion"
  exit 1
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
if [[ "$sharding_policy" == "auto" ]]; then
  echo "Sharding           : auto (runtime target ${SHARD_TARGET_MINUTES}m)"
else
  echo "Sharding           : ${sharding_policy} shard(s) / active pipeline"
fi
echo "Shards             : ${#detector_configs[@]}"
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

SCHEDULE_ESTIMATE_FLOOR_DECISECONDS=1
initial_claim_strategy="static-schedule"
if [[ "${DETECTOR_ALGORITHM,,}" == "all" ]] \
  && [[ "${DETECTOR_LOADING_STRATEGY,,}" == "lpt" ]] \
  && (( effective_pipelines > 1 )); then
  initial_claim_strategy="static-lpt-plan"
fi
echo "Scheduling strategy: $initial_claim_strategy"

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
    --image-root "$regression_image_root"
    --output "$shard_output"
    --run-mode "$REGRESSION_MODE"
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

  local shared_evidence_dir
  shared_evidence_dir="$OUTPUT_DIR/.learned-evidence/$detector_name"
  if [[ -f "$shared_evidence_dir/manifest.json" ]]; then
    echo "[pipeline $pipeline_number][$detector_name] Shared learned evidence: $shared_evidence_dir"
    args+=(--precomputed-evidence "$shared_evidence_dir")
  fi

  # Every regression evaluates the detector default baseline and the strongest
  # historic exact parameter set known before this run, independent of search.
  local historic_best_file
  historic_best_file="$OUTPUT_DIR/.references/$detector_name-task-$task_index-best.json"
  mkdir -p "$(dirname "$historic_best_file")"
  local -a historic_best_args=(
    --index results-repo/indexes/calibration-index.json
    --detector "$detector_name"
    --golden-set-sha256 "$golden_set_sha256"
  )
  local model_variant_env_name model_variant_value
  model_variant_env_name="HTH_MODEL_VARIANT_${detector_name^^}"
  model_variant_value="${!model_variant_env_name:-}"
  if [[ -n "$model_variant_value" ]]; then
    historic_best_args+=(--model-variant "$model_variant_value")
  fi
  if [[ -f results-repo/indexes/calibration-index.json ]] && python -m hth.calibration_store resolve-best-parameter \
      "${historic_best_args[@]}" \
      > "$historic_best_file" 2>/dev/null; then
    echo "[pipeline $pipeline_number][$detector_name] Historic best reference: $historic_best_file"
    args+=(--historic-best "$historic_best_file")
  else
    rm -f "$historic_best_file"
    echo "[pipeline $pipeline_number][$detector_name] No reconstructable historic best reference found."
  fi

  if [[ -n "${effective_limit:-}" ]]; then
    args+=(--strategy "$effective_strategy" --limit "$effective_limit")
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
      if [[ -f results-repo/indexes/calibration-index.json ]]; then
        calibration_file="$(python -m hth.calibration_store resolve \
          --index results-repo/indexes/calibration-index.json \
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
  if (( start_delay_seconds > 0 )); then
    echo "[pipeline $pipeline_number] WAIT detector=$detector_name shard=$((shard_index + 1))/$shard_count stagger=${start_delay_seconds}s"
    sleep "$start_delay_seconds"
  fi
  detector_loaded_epoch="$(date +%s)"
  detector_started_epoch="$detector_loaded_epoch"
  printf 'start\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date +%s.%N)" "$pipeline_number" "$detector_name" "$shard_index" "$shard_count" "$detector_threads" "$claim_batch_id" \
    >> "$telemetry_root/tasks/$task_index.tsv"
  lifecycle_time="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo
  echo "======================================================================"
  echo "[pipeline $pipeline_number] LOAD detector=$detector_name shard=$((shard_index + 1))/$shard_count threads=$detector_threads time=$lifecycle_time"
  echo "[pipeline $pipeline_number] estimate=${detector_estimate}s source=${detector_estimate_source}"
  echo "[pipeline $pipeline_number] ranked_quality=${detector_ranked_quality}"
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
  if [[ "$detector_name" == "orli_page_mask" ]]; then
    echo "[pipeline $pipeline_number][$detector_name] Worker model path: ${HTH_ORLI_PAGE_MODEL:-<unset>}"
    echo "[pipeline $pipeline_number][$detector_name] Worker model exists: $([[ -f "${HTH_ORLI_PAGE_MODEL:-}" ]] && echo yes || echo no)"
    echo "[pipeline $pipeline_number][$detector_name] Worker provenance path: ${HTH_ORLI_PAGE_PROVENANCE:-<unset>}"
    echo "[pipeline $pipeline_number][$detector_name] Worker provenance exists: $([[ -f "${HTH_ORLI_PAGE_PROVENANCE:-}" ]] && echo yes || echo no)"
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
    detector_wall_seconds="$((detector_finished_epoch - detector_started_epoch))"
    lifecycle_time="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "[pipeline $pipeline_number] UNLOAD detector=$detector_name shard=$((shard_index + 1))/$shard_count status=failed wall=${detector_wall_seconds}s time=$lifecycle_time"
    printf 'finish\t%s\tfailed\n' "$(date +%s.%N)" >> "$telemetry_root/tasks/$task_index.tsv"
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
  printf 'finish\t%s\tcomplete\n' "$(date +%s.%N)" >> "$telemetry_root/tasks/$task_index.tsv"
  echo "[pipeline $pipeline_number] Completed scheduled detector; continuing its fixed pipeline schedule."
  echo "======================================================================"
}

detector_worker() {
  local pipeline_index="$1" scheduled_tasks_csv="${2:-}"
  local pipeline_number=$((pipeline_index+1))
  local delay_seconds=$((pipeline_index*PIPELINE_STAGGER_MINUTES*60))
  local task_index task_start_delay schedule_batch_id
  local -a scheduled_tasks=()
  local scheduled_units=0

  IFS=',' read -r -a scheduled_tasks <<< "$scheduled_tasks_csv"
  if (( ${#scheduled_tasks[@]} == 0 )); then
    return 0
  fi
  for task_index in "${scheduled_tasks[@]}"; do
    scheduled_units=$((scheduled_units + ${task_claim_units[$task_index]:-$SCHEDULE_ESTIMATE_FLOOR_DECISECONDS}))
  done

  schedule_batch_id="$(printf 'p%03d-static' "$pipeline_number")"
  printf 'claim_batch\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date +%s.%N)" "$pipeline_number" "${task_threads[${scheduled_tasks[0]}]}" \
    "$((scheduled_units/10)).$((scheduled_units%10))" \
    "$(IFS=,; echo "${scheduled_tasks[*]}")" "$schedule_batch_id" \
    > "$telemetry_root/claim-batches/$schedule_batch_id.tsv"

  printf 'start\t%s\n' "$(date +%s.%N)" > "$telemetry_root/workers/$pipeline_number.tsv"
  echo "[pipeline $pipeline_number] Started fixed schedule: tasks=$(IFS=,; echo "${scheduled_tasks[*]}") estimate=$((scheduled_units/10)).$((scheduled_units%10))s"

  local first_task=1
  for task_index in "${scheduled_tasks[@]}"; do
    mkdir -p "$queue_dir/claims/$task_index"
    task_start_delay=0
    if (( first_task == 1 )); then task_start_delay="$delay_seconds"; first_task=0; fi
    if run_detector_config "$task_index" "$pipeline_number" "$task_start_delay" "$schedule_batch_id"; then
      : > "$queue_dir/done/$task_index"
    else
      : > "$queue_dir/failed/$task_index"
      printf 'end\t%s\n' "$(date +%s.%N)" >> "$telemetry_root/workers/$pipeline_number.tsv"
      echo "::error::Pipeline $pipeline_number failed detector config ${detector_configs[$task_index]}"
      return 1
    fi
  done

  printf 'end\t%s\n' "$(date +%s.%N)" >> "$telemetry_root/workers/$pipeline_number.tsv"
  echo "[pipeline $pipeline_number] Fixed schedule complete; persistence remains batched for post-run publication."
}

# Learned inference evidence is parameter-invariant. When a learned detector
# expands into multiple shard tasks, compute its Golden Set evidence exactly
# once in this parent process before any of those pipeline processes launch.
# Single-task Kraken/dhSegment runs keep the process-local prewarm path to avoid
# delaying unrelated multi-detector smoke work. Orli always uses the parent path
# because its deterministic evidence is persisted across builds.
shared_evidence_root="$OUTPUT_DIR/.learned-evidence"
mkdir -p "$shared_evidence_root"
declare -A learned_task_counts=()
for task_detector in "${task_detectors[@]}"; do
  case "$task_detector" in
    kraken_page_mask|orli_page_mask|dhsegment_page_mask|doc_ufcn_page_mask|mask_rcnn_page_mask|eynollah_page_mask|docextractor_page_mask|pagenet_page_mask)
      learned_task_counts["$task_detector"]=$(( ${learned_task_counts["$task_detector"]:-0} + 1 ))
      ;;
  esac
done

for learned_detector in kraken_page_mask orli_page_mask dhsegment_page_mask doc_ufcn_page_mask mask_rcnn_page_mask eynollah_page_mask docextractor_page_mask pagenet_page_mask; do
  learned_count="${learned_task_counts[$learned_detector]:-0}"
  prepare_shared_evidence=0
  if (( learned_count > 1 )); then
    prepare_shared_evidence=1
  elif [[ "$learned_detector" == "orli_page_mask" && "$learned_count" -gt 0 ]]; then
    prepare_shared_evidence=1
  fi
  if (( prepare_shared_evidence == 1 )); then
    learned_output="$shared_evidence_root/$learned_detector"
    rm -rf "$learned_output"
    echo
    echo "Shared Learned Golden Set Evidence — $learned_detector"
    echo "======================================================"
    echo "[learned-evidence][$learned_detector] shard tasks=$learned_count; preparing once before pipeline fan-out"
    evidence_started_epoch="$(date +%s.%N)"
    python -m hth.regression.learned_evidence prepare \
      --detector "$learned_detector" \
      --golden-set "hth-pipeline/$GOLDEN_SET" \
      --image-root "$regression_image_root" \
      --max-dimension "$MAX_DIMENSION" \
      --output "$learned_output" \
      --results-root results-repo
    evidence_finished_epoch="$(date +%s.%N)"
    printf 'shared_evidence\t%s\t%s\t%s\t%s\n'       "$learned_detector" "$learned_count" "$evidence_started_epoch" "$evidence_finished_epoch"       >> "$telemetry_root/learned-evidence.tsv"
  fi
done

worker_pids=()
declare -a static_pipeline_tasks=()
while IFS=$'\t' read -r pipeline_number task_csv estimated_seconds; do
  [[ -n "$pipeline_number" ]] || continue
  static_pipeline_tasks[$((pipeline_number-1))]="$task_csv"
  echo "Static schedule pipeline=$pipeline_number tasks=$task_csv estimate=${estimated_seconds}s"
done < <(python - "$effective_pipelines" "${DETECTOR_ALGORITHM,,}" "${#detector_configs[@]}" "${detector_estimates[@]}" <<'PYSTATICDISPATCH'
import sys
from hth.domain.execution_dispatch import plan_static_dispatch

pipelines = int(sys.argv[1])
multidetector = sys.argv[2] == "all"
task_count = int(sys.argv[3])
estimates = []
for raw in sys.argv[4:]:
    try:
        estimates.append(float(raw))
    except (TypeError, ValueError):
        estimates.append(None)
for row in plan_static_dispatch(
    task_count=task_count,
    pipeline_count=pipelines,
    multidetector=multidetector,
    estimates=estimates,
):
    print(f"{row['pipeline']}\t{','.join(str(i) for i in row['task_indexes'])}\t{row['estimated_seconds']:.1f}")
PYSTATICDISPATCH
)

startup_complete_epoch="$(date +%s.%N)"
startup_overhead_seconds="$(python - "$executor_started_epoch" "$startup_complete_epoch" <<'PYSTARTUP'
import sys
print(f"{max(0.0, float(sys.argv[2]) - float(sys.argv[1])):.6f}")
PYSTARTUP
)"
printf '{"executor_startup_overhead_seconds": %s, "definition": "run-detector-regressions start through pre-fan-out lifecycle, planning, and shared-evidence preparation"}\n' \
  "$startup_overhead_seconds" > "$OUTPUT_DIR/optimizer-overhead.json"
echo "Executor startup overhead: ${startup_overhead_seconds}s (included in optimizer shape wall time; shard timings start after fan-out)"

for ((pipeline_index=0; pipeline_index<effective_pipelines; pipeline_index++)); do
  detector_worker "$pipeline_index" "${static_pipeline_tasks[$pipeline_index]:-}" &
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
invalid_detectors=()
for detector_name in "${unique_detectors[@]}"; do
  mapfile -t detector_task_indexes < <(
    for ((task_index = 0; task_index < ${#task_detectors[@]}; task_index++)); do
      [[ "${task_detectors[$task_index]}" == "$detector_name" ]] && printf '%s\n' "$task_index"
    done
  )
  first_detector_task_index="${detector_task_indexes[0]}"
  expected_detector_shards="${task_shard_counts[$first_detector_task_index]}"
  detector_shard_dirs=()
  # Workers publish the exact canonical run directory for every completed task.
  # Consume that execution evidence here instead of rediscovering run-* trees;
  # filesystem ordering is not a completion contract and became especially
  # fragile once static LPT schedules allowed detectors to finish out of order.
  for task_index in "${detector_task_indexes[@]}"; do
    shard_index="${task_shard_indexes[$task_index]}"
    completed_run_file="$queue_dir/run-dirs/$(printf '%04d' "$task_index")"
    shard_run_dir=""
    if [[ -f "$completed_run_file" ]]; then
      shard_run_dir="$(cat "$completed_run_file")"
    fi
    if [[ -z "$shard_run_dir" || ! -d "$shard_run_dir" ]]; then
      echo "::error::Missing completed shard $((shard_index + 1))/$expected_detector_shards for $detector_name"
      exit 1
    fi
    detector_shard_dirs[$shard_index]="$shard_run_dir"
  done
  for ((shard_index = 0; shard_index < expected_detector_shards; shard_index++)); do
    if [[ -z "${detector_shard_dirs[$shard_index]:-}" ]]; then
      echo "::error::Missing completed shard $((shard_index + 1))/$expected_detector_shards for $detector_name"
      exit 1
    fi
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
      --image-root "$regression_image_root" \
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
  outcome_status="$(python - "$finalized_dir/reports/summary.json" <<'PYOUTCOME'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
state = summary.get("measurement_state") if isinstance(summary, dict) else None
print(state.get("status", "unknown") if isinstance(state, dict) else "unknown")
PYOUTCOME
  )"
  if [[ "$outcome_status" == "no_valid_measurements" ]]; then
    invalid_detectors+=("$detector_name")
    echo "::error::Detector regression produced no valid measurements: $detector_name"
  fi
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
if (( ${#invalid_detectors[@]} > 0 )); then
  echo "::error::Regression failed: no valid measurements for detector(s): ${invalid_detectors[*]}"
  exit 1
fi
