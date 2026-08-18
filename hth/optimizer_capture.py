#!/usr/bin/env python3
"""Capture execution-optimizer shape and shard observations."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hth.parallelism_store import observation_from_run, update_parallelism_index, update_parallelism_shards
from hth.runner_metrics import summarize_runner_metrics


def _append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def capture_observation(
    *,
    results_root: Path,
    run_dir: Path,
    wall_clock_seconds: float,
    runner_label: str,
    github_run_id: str,
    shape_sequence: int,
    startup_overhead_seconds: float | None = None,
    observation_log: Path | None = None,
    runner_metrics_log: Path | None = None,
) -> dict[str, Any]:
    build = {
        "mode": "full",
        "runner_label": runner_label,
        "github_run_id": github_run_id,
        "optimizer_run_id": github_run_id,
        "optimizer_shape_sequence": shape_sequence,
        "source": "execution-optimizer",
    }
    observation = observation_from_run(run_dir, build=build, wall_clock_seconds=wall_clock_seconds)
    observation["observation_id"] = f"optimizer:{github_run_id}:{shape_sequence}:{observation['run_id']}"
    observation["optimizer_run_id"] = str(github_run_id)
    observation["optimizer_shape_sequence"] = shape_sequence
    observation["source"] = "execution-optimizer"
    if startup_overhead_seconds is not None:
        observation["startup_overhead_seconds"] = max(0.0, float(startup_overhead_seconds))
        observation["startup_overhead_included_in_wall_clock"] = True
    if runner_metrics_log is not None:
        observation["runner_metrics"] = summarize_runner_metrics(
            runner_metrics_log,
            optimizer_run_id=str(github_run_id),
            shape_sequence=shape_sequence,
        )
    update_parallelism_index(results_root, [observation])
    _append_jsonl(observation_log, observation)
    return observation


def capture_shard_observation(
    *,
    results_root: Path,
    run_dir: Path,
    runner_label: str,
    github_run_id: str,
    shape_sequence: int,
    pipeline_number: int,
    shard_index: int,
    shard_count: int,
    threads: int,
    wall_clock_seconds: float,
    shard_log: Path | None = None,
    runner_metrics_log: Path | None = None,
) -> dict[str, Any]:
    info = _read_json(run_dir / "RUN-INFO.json")
    summary = _read_json(run_dir / "reports" / "summary.json")
    parameter_space = summary.get("parameter_space") if isinstance(summary.get("parameter_space"), dict) else {}
    actual_sets = info.get("actual_parameter_sets") or parameter_space.get("actual_parameter_sets")
    local_sets = info.get("locally_evaluated_parameter_sets") or parameter_space.get("locally_evaluated_parameter_sets")
    try:
        actual_sets = int(actual_sets)
    except (TypeError, ValueError):
        actual_sets = None
    try:
        local_sets = int(local_sets)
    except (TypeError, ValueError):
        local_sets = actual_sets
    wall = float(wall_clock_seconds)
    detector_id = str(info.get("detector") or summary.get("detector") or "unknown")
    runner = summary.get("runner") if isinstance(summary.get("runner"), dict) else {}
    record: dict[str, Any] = {
        "observation_id": f"optimizer-shard:{github_run_id}:{shape_sequence}:{shard_index}:{info.get('run_id', run_dir.name)}",
        "record_type": "optimizer-shard",
        "source": "execution-optimizer",
        "optimizer_run_id": str(github_run_id),
        "shape_sequence": shape_sequence,
        "observed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "detector_id": detector_id,
        "run_id": info.get("run_id") or run_dir.name,
        "pipeline_number": pipeline_number,
        "shard_index": shard_index,
        "shard_number": shard_index + 1,
        "shard_count": shard_count,
        "threads_per_pipeline": threads,
        "wall_clock_seconds": wall,
        "actual_parameter_sets": actual_sets,
        "locally_evaluated_parameter_sets": local_sets,
        "baseline_execution": info.get("baseline_execution") or parameter_space.get("baseline_execution"),
        "parameter_sets_per_second": (local_sets / wall) if local_sets is not None and wall > 0 else None,
        "runner": {
            "runner_label": runner_label,
            "runner_name": runner.get("runner_name") or info.get("runner_name"),
            "runner_labels": runner.get("github_runner_labels") or info.get("github_runner_labels"),
            "logical_cpu_count": runner.get("logical_cpu_count") or info.get("logical_cpu_count"),
        },
    }
    if runner_metrics_log is not None:
        metrics = summarize_runner_metrics(
            runner_metrics_log,
            optimizer_run_id=str(github_run_id),
            shape_sequence=shape_sequence,
        )
        record["runner_metrics_at_completion"] = metrics
    update_parallelism_shards(results_root, [record])
    _append_jsonl(shard_log, record)
    return record


def replay_observations(*, results_root: Path, observation_log: Path) -> int:
    observations: list[dict[str, Any]] = []
    if not observation_log.is_file():
        return 0
    for line in observation_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("Optimizer observation log contains a non-object row")
        observations.append(payload)
    update_parallelism_index(results_root, observations)
    return len(observations)


def replay_shard_observations(*, results_root: Path, shard_log: Path) -> int:
    rows: list[dict[str, Any]] = []
    if not shard_log.is_file():
        return 0
    for line in shard_log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    update_parallelism_shards(results_root, rows)
    return len(rows)


def assess_early_stop(observation_log: Path, *, threshold_pct: float = 1.0, consecutive: int = 3) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if observation_log.is_file():
        for line in observation_log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    rows.sort(key=lambda row: int(row.get("optimizer_shape_sequence") or 0))
    threshold = threshold_pct / 100.0
    best_rate: float | None = None
    best_shape: str | None = None
    streak = 0
    assessments: list[dict[str, Any]] = []
    for row in rows:
        rate_value = row.get("parameter_sets_per_second")
        if not isinstance(rate_value, (int, float)) or float(rate_value) <= 0:
            continue
        rate = float(rate_value)
        previous_best = best_rate
        meaningful = previous_best is None or rate > previous_best * (1.0 + threshold)
        improved = previous_best is None or rate > previous_best
        improvement_pct = None if previous_best is None else ((rate - previous_best) / previous_best) * 100.0
        if meaningful:
            streak = 0
        else:
            streak += 1
        if improved:
            best_rate = rate
            best_shape = str(row.get("execution_shape") or "unknown")
        assessments.append({
            "shape_sequence": row.get("optimizer_shape_sequence"),
            "execution_shape": row.get("execution_shape"),
            "parameter_sets_per_second": rate,
            "previous_best_parameter_sets_per_second": previous_best,
            "improvement_pct_vs_perceived_max": improvement_pct,
            "meaningful_improvement": meaningful,
            "non_improving_streak": streak,
        })
    should_stop = streak >= consecutive and len(assessments) >= consecutive + 1
    return {
        "should_stop": should_stop,
        "stop_reason": "throughput_plateau" if should_stop else None,
        "threshold_pct": threshold_pct,
        "required_consecutive_shapes": consecutive,
        "non_improving_streak": streak,
        "best_parameter_sets_per_second": best_rate,
        "best_execution_shape": best_shape,
        "completed_shapes": len(assessments),
        "assessments": assessments,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--shard-run-dir", type=Path)
    parser.add_argument("--wall-clock-seconds", type=float)
    parser.add_argument("--shard-wall-clock-seconds", type=float)
    parser.add_argument("--runner-label")
    parser.add_argument("--github-run-id")
    parser.add_argument("--shape-sequence", type=int)
    parser.add_argument("--startup-overhead-seconds", type=float)
    parser.add_argument("--pipeline-number", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--observation-log", type=Path)
    parser.add_argument("--shard-log", type=Path)
    parser.add_argument("--runner-metrics-log", type=Path)
    parser.add_argument("--observation-json", type=Path)
    parser.add_argument("--replay-log", type=Path)
    parser.add_argument("--replay-shard-log", type=Path)
    parser.add_argument("--assess-log", type=Path)
    parser.add_argument("--threshold-pct", type=float, default=1.0)
    parser.add_argument("--consecutive", type=int, default=3)
    args = parser.parse_args()

    if args.assess_log is not None:
        print(json.dumps(assess_early_stop(args.assess_log, threshold_pct=args.threshold_pct, consecutive=args.consecutive), sort_keys=True))
        return 0
    if args.replay_log is not None or args.replay_shard_log is not None:
        count = 0
        if args.replay_log is not None:
            count += replay_observations(results_root=args.results_root, observation_log=args.replay_log)
        if args.replay_shard_log is not None:
            count += replay_shard_observations(results_root=args.results_root, shard_log=args.replay_shard_log)
        print(f"Replayed {count} optimizer records")
        return 0
    if args.shard_run_dir is not None:
        required = (args.shard_wall_clock_seconds, args.runner_label, args.github_run_id, args.shape_sequence, args.pipeline_number, args.shard_index, args.shard_count, args.threads)
        if any(value is None for value in required):
            parser.error("shard capture requires shard timing, runner, run, shape, pipeline, shard, and thread fields")
        record = capture_shard_observation(
            results_root=args.results_root,
            run_dir=args.shard_run_dir,
            runner_label=str(args.runner_label),
            github_run_id=str(args.github_run_id),
            shape_sequence=int(args.shape_sequence),
            pipeline_number=int(args.pipeline_number),
            shard_index=int(args.shard_index),
            shard_count=int(args.shard_count),
            threads=int(args.threads),
            wall_clock_seconds=float(args.shard_wall_clock_seconds),
            shard_log=args.shard_log,
            runner_metrics_log=args.runner_metrics_log,
        )
        print(f"Persisted optimizer shard {record['shard_number']}/{record['shard_count']} wall={record['wall_clock_seconds']}s")
        return 0

    required = (args.run_dir, args.wall_clock_seconds, args.runner_label, args.github_run_id, args.shape_sequence)
    if any(value is None for value in required):
        parser.error("shape capture requires run-dir, wall-clock, runner, run id, and shape sequence")
    observation = capture_observation(
        results_root=args.results_root,
        run_dir=args.run_dir,
        wall_clock_seconds=float(args.wall_clock_seconds),
        runner_label=str(args.runner_label),
        github_run_id=str(args.github_run_id),
        shape_sequence=int(args.shape_sequence),
        startup_overhead_seconds=args.startup_overhead_seconds,
        observation_log=args.observation_log,
        runner_metrics_log=args.runner_metrics_log,
    )
    if args.observation_json is not None:
        args.observation_json.parent.mkdir(parents=True, exist_ok=True)
        args.observation_json.write_text(json.dumps(observation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Captured optimizer shape {observation.get('execution_shape')} wall={observation.get('wall_clock_seconds')}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
