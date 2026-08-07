#!/usr/bin/env python3
"""Capture one serial execution-optimizer shape into parallelism-index.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hth.parallelism_store import observation_from_run, update_parallelism_index


def capture_observation(
    *,
    results_root: Path,
    run_dir: Path,
    wall_clock_seconds: float,
    runner_label: str,
    github_run_id: str,
    shape_sequence: int,
    observation_log: Path | None = None,
) -> dict[str, Any]:
    build = {
        "mode": "full",
        "runner_label": runner_label,
        "github_run_id": github_run_id,
        "optimizer_shape_sequence": shape_sequence,
        "source": "execution-optimizer",
    }
    observation = observation_from_run(
        run_dir,
        build=build,
        wall_clock_seconds=wall_clock_seconds,
    )
    # A single optimizer workflow intentionally repeats local run IDs across
    # execution shapes. Include the shape sequence so every experiment remains
    # independently addressable in the persistent raw observation store.
    observation["observation_id"] = (
        f"optimizer:{github_run_id}:{shape_sequence}:{observation['run_id']}"
    )
    observation["optimizer_shape_sequence"] = shape_sequence
    update_parallelism_index(results_root, [observation])
    if observation_log is not None:
        observation_log.parent.mkdir(parents=True, exist_ok=True)
        with observation_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(observation, sort_keys=True) + "\n")
    return observation


def replay_observations(*, results_root: Path, observation_log: Path) -> int:
    observations: list[dict[str, Any]] = []
    for line in observation_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("Optimizer observation log contains a non-object row")
        observations.append(payload)
    update_parallelism_index(results_root, observations)
    return len(observations)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--wall-clock-seconds", type=float)
    parser.add_argument("--runner-label")
    parser.add_argument("--github-run-id")
    parser.add_argument("--shape-sequence", type=int)
    parser.add_argument("--observation-log", type=Path)
    parser.add_argument("--replay-log", type=Path)
    args = parser.parse_args()
    if args.replay_log is not None:
        count = replay_observations(results_root=args.results_root, observation_log=args.replay_log)
        print(f"Replayed {count} optimizer observations")
        return 0
    required = {
        "run_dir": args.run_dir,
        "wall_clock_seconds": args.wall_clock_seconds,
        "runner_label": args.runner_label,
        "github_run_id": args.github_run_id,
        "shape_sequence": args.shape_sequence,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("capture mode requires: " + ", ".join(missing))
    observation = capture_observation(
        results_root=args.results_root,
        run_dir=args.run_dir,
        wall_clock_seconds=args.wall_clock_seconds,
        runner_label=args.runner_label,
        github_run_id=args.github_run_id,
        shape_sequence=args.shape_sequence,
        observation_log=args.observation_log,
    )
    print(
        "Captured optimizer shape "
        f"{observation.get('execution_shape')} "
        f"wall={observation.get('wall_clock_seconds')}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
