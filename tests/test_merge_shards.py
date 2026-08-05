from __future__ import annotations

from hth.regression.sharding import plan_shards


def test_shard_planner_caps_runner_threads() -> None:
    assert plan_shards(4 * 3600, runner_label="e7k", requested_threads="auto").threads == 48
    assert plan_shards(4 * 3600, runner_label="e9k", requested_threads="auto").threads == 32
