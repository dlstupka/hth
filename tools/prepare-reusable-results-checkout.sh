#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: prepare-reusable-results-checkout.sh prepare|verify EXPECTED_REPOSITORY}"
expected_repository="${2:?expected results repository is required}"
: "${GITHUB_WORKSPACE:?GITHUB_WORKSPACE is required}"

workspace="$(cd "$GITHUB_WORKSPACE" && pwd -P)"
target="$workspace/results-repo"

if [[ -z "$workspace" || "$workspace" == "/" || "$target" != "$workspace/results-repo" ]]; then
  echo "Unsafe reusable checkout target: $target" >&2
  exit 2
fi

normalize_repository_slug() {
  local value="${1%.git}"
  value="${value#https://}"
  value="${value#http://}"
  value="${value#ssh://}"
  value="${value#git@github.com:}"
  value="${value#*@github.com/}"
  value="${value#github.com/}"
  printf '%s' "$value"
}

repository_slug="$(normalize_repository_slug "$expected_repository")"
canonical_origin="https://github.com/${repository_slug}.git"

origin_matches() {
  local origin normalized
  origin="$(git -C "$target" remote get-url origin 2>/dev/null)" || return 1
  normalized="$(normalize_repository_slug "$origin")"
  [[ "$normalized" == "$repository_slug" ]]
}

valid_checkout() {
  [[ -d "$target/.git" ]] || return 1
  git -C "$target" rev-parse --verify 'HEAD^{commit}' >/dev/null 2>&1 || return 1
  origin_matches
}

recreate_checkout() {
  echo "::warning::Reusable results checkout is invalid; recreating only $target"
  rm -rf -- "$target"
}

case "$mode" in
  prepare)
    [[ -e "$target" ]] || exit 0
    if ! valid_checkout; then
      recreate_checkout
      exit 0
    fi
    # Persistence temporarily authenticates origin by embedding x-access-token
    # credentials. Recognize that as the same repository, then remove the
    # credential-bearing URL before the checkout is reused.
    git -C "$target" remote set-url origin "$canonical_origin"
    if ! git -C "$target" reset --hard HEAD || ! git -C "$target" clean -ffd; then
      recreate_checkout
      exit 0
    fi
    echo "Reusable results checkout validated and cleaned: $target"
    ;;
  verify)
    if ! valid_checkout; then
      echo "Results checkout failed post-checkout validation: $target" >&2
      exit 1
    fi
    if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
      echo "Results checkout is not clean immediately after checkout: $target" >&2
      git -C "$target" status --short >&2
      exit 1
    fi
    echo "Results checkout verified: $(git -C "$target" rev-parse --short HEAD) ($repository_slug)"
    ;;
  *)
    echo "Unknown mode: $mode" >&2
    exit 2
    ;;
esac
