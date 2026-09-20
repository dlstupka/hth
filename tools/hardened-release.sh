#!/usr/bin/env bash
# Collision-safe immutable GitHub Release publication for deterministic assets.

hth_release_asset_digest() {
  local repository="${1:?repository is required}"
  local tag="${2:?release tag is required}"
  local asset_name="${3:?asset name is required}"
  local attempt digest
  for attempt in 1 2 3 4 5; do
    digest="$(
      gh api "repos/$repository/releases/tags/$tag" \
        --jq ".assets[] | select(.name == \"$asset_name\") | .digest" \
        2>/dev/null || true
    )"
    if [[ -n "$digest" ]]; then
      printf '%s' "$digest"
      return 0
    fi
    (( attempt == 5 )) || sleep "$attempt"
  done
  return 1
}

hth_publish_immutable_release() {
  local repository="${1:?repository is required}"
  local tag="${2:?release tag is required}"
  local asset="${3:?asset path is required}"
  local asset_name="${4:?asset name is required}"
  local asset_sha256="${5:?asset SHA-256 is required}"
  local title="${6:?release title is required}"
  local notes="${7:?release notes are required}"
  local expected="sha256:${asset_sha256,,}"
  local create_log="${RUNNER_TEMP:-/tmp}/hth-release-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}.log"
  local digest

  [[ -f "$asset" ]] || {
    echo "::error::Immutable release asset does not exist: $asset"
    return 2
  }
  [[ "$(basename "$asset")" == "$asset_name" ]] || {
    echo "::error::Immutable release asset name mismatch: expected $asset_name, found $(basename "$asset")"
    return 2
  }
  [[ "$asset_sha256" =~ ^[0-9a-fA-F]{64}$ ]] || {
    echo "::error::Immutable release asset SHA-256 is invalid"
    return 2
  }

  HTH_RELEASE_ACTIVITY=""
  if gh release view "$tag" --repo "$repository" >/dev/null 2>&1; then
    HTH_RELEASE_ACTIVITY=REUSED
  elif gh release create "$tag" "$asset" \
      --repo "$repository" --target main --title "$title" --notes "$notes" \
      >"$create_log" 2>&1; then
    HTH_RELEASE_ACTIVITY=CREATED
  elif gh release view "$tag" --repo "$repository" >/dev/null 2>&1; then
    # Another equivalent publisher won the check/create race. The digest check
    # below decides whether that concurrent release is safe to reuse.
    HTH_RELEASE_ACTIVITY=REUSED
  else
    cat "$create_log" >&2
    echo "::error::Immutable release creation failed and no concurrent release exists: $tag"
    return 1
  fi

  if ! digest="$(hth_release_asset_digest "$repository" "$tag" "$asset_name")"; then
    echo "::error::Immutable release asset digest did not become available: $tag/$asset_name"
    return 1
  fi
  if [[ "${digest,,}" != "$expected" ]]; then
    echo "::error::Immutable release collision: $tag/$asset_name expected $expected, found $digest"
    return 1
  fi
  echo "Immutable release ${HTH_RELEASE_ACTIVITY,,}: $repository@$tag ($digest)"
}
