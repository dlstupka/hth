#!/usr/bin/env bash
# Shared collision-safe persistence transaction for HTH results-repository writers.
#
# The caller supplies an apply callback that reconstructs/merges and stages exactly
# the payload it owns. This helper supplies the race-safe persistence mechanics:
# refresh current remote state, reset to that state, invoke the callback, commit,
# push, and retry only confirmed concurrent-update collisions.
#
# Usage:
#   hth_hardened_persist <repo> <branch> <commit-message> <apply-callback> [label]
#
# The callback receives the 1-based attempt number. It runs after reset to the
# latest origin/<branch> on every attempt and must stage all intended changes.
#
# Outputs in the calling shell:
#   HTH_PERSIST_CHANGED   true|false
#   HTH_PERSIST_COMMIT    final local commit
#   HTH_PERSIST_ATTEMPTS  attempts consumed

hth_hardened_persist() {
  local repo="${1:?results repository path is required}"
  local branch="${2:?remote branch is required}"
  local commit_message="${3:?commit message is required}"
  local apply_callback="${4:?apply callback is required}"
  local label="${5:-Persistence}"
  local max_attempts="${HTH_PERSIST_MAX_ATTEMPTS:-5}"
  local backoff_seconds="${HTH_PERSIST_BACKOFF_SECONDS:-5}"

  if ! declare -F "$apply_callback" >/dev/null 2>&1; then
    echo "::error::${label}: persistence callback '$apply_callback' is not defined."
    return 2
  fi
  if ! [[ "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
    echo "::error::${label}: HTH_PERSIST_MAX_ATTEMPTS must be a positive integer."
    return 2
  fi
  if ! [[ "$backoff_seconds" =~ ^[0-9]+$ ]]; then
    echo "::error::${label}: HTH_PERSIST_BACKOFF_SECONDS must be a non-negative integer."
    return 2
  fi

  HTH_PERSIST_CHANGED=false
  HTH_PERSIST_COMMIT=""
  HTH_PERSIST_ATTEMPTS=0

  local attempt retry_delay push_log
  for attempt in $(seq 1 "$max_attempts"); do
    HTH_PERSIST_ATTEMPTS="$attempt"
    echo "${label}: persistence attempt ${attempt}/${max_attempts}"

    if (( attempt > 1 )); then
      retry_delay=$((backoff_seconds * (attempt - 1)))
      echo "${label}: concurrent update confirmed; waiting ${retry_delay}s before retrying against latest ${branch}."
      sleep "$retry_delay"
    fi

    # Every attempt begins from the newest durable state. The callback then
    # reapplies only its owned mutation so unrelated concurrent writes survive.
    git -C "$repo" fetch origin "$branch"
    git -C "$repo" reset --hard "origin/$branch"

    "$apply_callback" "$attempt"

    if git -C "$repo" diff --cached --quiet; then
      echo "${label}: durable state is already current."
      HTH_PERSIST_COMMIT="$(git -C "$repo" rev-parse HEAD)"
      return 0
    fi

    git -C "$repo" commit -m "$commit_message"
    HTH_PERSIST_CHANGED=true

    push_log="${RUNNER_TEMP:-/tmp}/hth-persist-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}-${attempt}.log"
    if git -C "$repo" push origin "HEAD:$branch" 2>&1 | tee "$push_log"; then
      HTH_PERSIST_COMMIT="$(git -C "$repo" rev-parse HEAD)"
      echo "${label}: persisted successfully at ${HTH_PERSIST_COMMIT}."
      return 0
    fi

    if grep -Eqi \
      'non-fast-forward|fetch first|failed to push some refs|updates were rejected because the remote contains work|updates were rejected because the tip of your current branch is behind' \
      "$push_log"; then
      if (( attempt == max_attempts )); then
        echo "::error::${label}: concurrent publication persisted through ${max_attempts} attempts."
        return 1
      fi
      continue
    fi

    echo "::error::${label}: push failed for a reason other than a concurrent update; refusing to misclassify and retry it."
    return 1
  done

  echo "::error::${label}: persistence attempts exhausted unexpectedly."
  return 1
}
