from __future__ import annotations

import json
from pathlib import Path

from hth.regression.sharding import (
    automatic_threads,
    best_smoke_observation,
    lease_expired,
    plan_shards,
    runner_max_threads,
    write_lease,
)
from hth.regression.runner import parse_args


def test_runner_profiles_and_auto_thread_thresholds() -> None:
    assert runner_max_threads("e7k") == 48
    assert runner_max_threads("e9k") == 32
    assert automatic_threads(60, 48) == 1
    assert automatic_threads(6 * 60, 48) == 4
    assert automatic_threads(20 * 60, 48) == 8
    assert automatic_threads(31 * 60, 48) == 48


def test_long_plan_is_sharded_and_short_plan_is_not() -> None:
    assert plan_shards(60, runner_label="e9k").shard_count == 1
    plan = plan_shards(12 * 3600, runner_label="e9k")
    assert plan.threads == 32
    assert plan.shard_count > 1


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
