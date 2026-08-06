from __future__ import annotations

import json
from pathlib import Path

from hth.regression.sharding import (
    automatic_threads,
    budgeted_threads,
    best_smoke_observation,
    lease_expired,
    plan_execution,
    plan_shards,
    runner_max_threads,
    write_lease,
)
from hth.regression.runner import parse_args


def test_runner_profiles_and_auto_thread_thresholds() -> None:
    assert runner_max_threads("e7k") == 192
    assert runner_max_threads("e9k") == 64
    assert runner_max_threads("github-hosted") == 8
    assert runner_max_threads("github-hosted", available_cpus=4) == 8
    assert automatic_threads(60, 64) == 1
    assert automatic_threads(6 * 60, 64) == 4
    assert automatic_threads(20 * 60, 64) == 8
    assert automatic_threads(31 * 60, 64) == 64
    assert budgeted_threads(16, runner_label="e7k", active_pipelines=4) == 16
    assert budgeted_threads(4, runner_label="github-hosted", active_pipelines=4) == 2


def test_execution_plan_uses_the_aggregate_runner_budget() -> None:
    github = plan_execution("auto", runner_label="github-hosted", active_pipelines=4)
    assert github.runner_thread_budget == 8
    assert github.threads_per_pipeline == 2
    assert github.allocated_threads == 8
    assert github.unused_threads == 0

    e7k = plan_execution("auto", runner_label="e7k", active_pipelines=3)
    assert e7k.runner_thread_budget == 192
    assert e7k.threads_per_pipeline == 64
    assert e7k.allocated_threads == 192
    assert e7k.unused_threads == 0

    capped = plan_execution(16, runner_label="e7k", active_pipelines=3)
    assert capped.threads_per_pipeline == 16
    assert capped.allocated_threads == 48
    assert capped.unused_threads == 144


def test_long_plan_is_sharded_and_short_plan_is_not() -> None:
    assert plan_shards(60, runner_label="e9k").shard_count == 1
    plan = plan_shards(12 * 3600, runner_label="e9k")
    assert plan.threads == 64
    assert plan.shard_count > 1



def test_explicit_shards_override_wall_clock_and_cap_at_one_parameter_per_shard() -> None:
    explicit = plan_shards(12 * 3600, runner_label="e7k", requested_shards=6, possible_parameter_sets=6562)
    assert explicit.shard_count == 6
    assert explicit.estimate_source == "explicit-shard-count"

    capped = plan_shards(12 * 3600, runner_label="e7k", requested_shards=999, possible_parameter_sets=10)
    assert capped.shard_count == 10

def test_expiring_lease(tmp_path: Path) -> None:
    lease = tmp_path / "lease.json"
    write_lease(lease, owner="worker-1", ttl_seconds=60, shard="2/8")
    payload = json.loads(lease.read_text())
    assert not lease_expired(lease, now=payload["expires_epoch"] - 1)
    assert lease_expired(lease, now=payload["expires_epoch"] + 1)


def test_smoke_lookup_prefers_latest(tmp_path: Path) -> None:
    index = tmp_path / "runtime-index.json"
    index.write_text(json.dumps({"observations": [
        {"detector_id": "d", "mode": "smoke", "observed_at_utc": "1", "wall_clock_seconds": 2},
        {"detector_id": "d", "mode": "smoke", "observed_at_utc": "2", "wall_clock_seconds": 3},
    ]}))
    assert best_smoke_observation(index, "d")["wall_clock_seconds"] == 3


def test_runner_accepts_valid_shard_arguments() -> None:
    args = parse_args([
        "--detector-config", "d.json", "--golden-set", "g.json",
        "--image-root", "images", "--output", "out",
        "--shard-index", "2", "--shard-count", "4",
    ])
    assert (args.shard_index, args.shard_count) == (2, 4)


def test_named_optimization_runner_budgets_are_twice_vcpu_policy() -> None:
    from hth.regression.sharding import runner_max_threads

    assert runner_max_threads("e7k") == 192
    assert runner_max_threads("e9k") == 64
