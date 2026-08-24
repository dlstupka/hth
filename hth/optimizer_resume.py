#!/usr/bin/env python3
"""Prepare a completed-shape optimizer checkpoint for a new resume execution."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from hth.results_layout import canonical_index_path, readable_index_path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object row in {path}")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def _published_run_ids(results_root: Path) -> set[str]:
    index = readable_index_path(results_root, "optimizer-index.json")
    if not index.is_file():
        return set()
    payload = _read_json(index)
    runs = payload.get("runs") if isinstance(payload.get("runs"), dict) else {}
    return {str(run_id) for run_id in runs}


def _matches(metadata: dict[str, Any], *, detector: str, runner_label: str, runner_budget: int,
             thread_min: int, thread_max: int, enumeration: str, sharding: str,
             allow_thread_oversubscription: bool = False) -> bool:
    expected = {
        "detector_id": detector,
        "runner_label": runner_label,
        "runner_thread_budget": runner_budget,
        "thread_min": thread_min,
        "thread_max": thread_max,
        "pipeline_enumeration": enumeration,
        "sharding": sharding,
        "allow_thread_oversubscription": allow_thread_oversubscription,
    }
    for key, value in expected.items():
        actual = metadata.get(key, False) if key == "allow_thread_oversubscription" else metadata.get(key)
        if actual != value:
            return False
    return True


def _rewrite_run(rows: list[dict[str, Any]], *, old_run_id: str, new_run_id: str) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        if str(row.get("optimizer_run_id")) != old_run_id:
            continue
        row["optimizer_run_id"] = new_run_id
        row["resumed_from_optimizer_run_id"] = old_run_id
        if isinstance(row.get("build"), dict):
            row["build"] = dict(row["build"])
            row["build"]["optimizer_run_id"] = new_run_id
            row["build"]["github_run_id"] = new_run_id
            row["build"]["resumed_from_optimizer_run_id"] = old_run_id
        observation_id = row.get("observation_id")
        if isinstance(observation_id, str):
            row["observation_id"] = observation_id.replace(old_run_id, new_run_id, 1)
        rewritten.append(row)
    return rewritten


def prepare_resume(*, source_dir: Path, destination_dir: Path, results_root: Path, mode: str,
                   current_run_id: str, detector: str, runner_label: str, runner_budget: int,
                   thread_min: int, thread_max: int, enumeration: str, sharding: str,
                   pipeline_min: int, pipeline_max: int,
                   allow_thread_oversubscription: bool = False) -> dict[str, Any]:
    mode = mode.strip()
    if mode == "no":
        return {"resumed": False, "reason": "disabled", "completed_shapes": 0}

    metadata_path = source_dir / "run-metadata.json"
    observations_path = source_dir / "observations.jsonl"
    shards_path = source_dir / "shards.jsonl"
    if not metadata_path.is_file() or not observations_path.is_file():
        return {"resumed": False, "reason": "no-local-checkpoint", "completed_shapes": 0}

    metadata = _read_json(metadata_path)
    old_run_id = str(metadata.get("optimizer_run_id") or "")
    if not old_run_id:
        return {"resumed": False, "reason": "checkpoint-missing-run-id", "completed_shapes": 0}
    if mode not in {"auto", old_run_id}:
        return {"resumed": False, "reason": "requested-run-id-not-local", "completed_shapes": 0}
    if old_run_id in _published_run_ids(results_root):
        return {"resumed": False, "reason": "checkpoint-already-published", "completed_shapes": 0}
    if not _matches(metadata, detector=detector, runner_label=runner_label, runner_budget=runner_budget,
                    thread_min=thread_min, thread_max=thread_max, enumeration=enumeration,
                    sharding=sharding,
                    allow_thread_oversubscription=allow_thread_oversubscription):
        return {"resumed": False, "reason": "checkpoint-incompatible", "completed_shapes": 0}

    observations = _rewrite_run(_read_jsonl(observations_path), old_run_id=old_run_id, new_run_id=current_run_id)
    observations = [
        row for row in observations
        if pipeline_min <= int(row.get("active_pipelines") or 0) <= pipeline_max
    ]
    if not observations:
        return {"resumed": False, "reason": "no-compatible-completed-shapes", "completed_shapes": 0}

    completed_sequences = {int(row.get("optimizer_shape_sequence") or 0) for row in observations}
    shards = _rewrite_run(_read_jsonl(shards_path), old_run_id=old_run_id, new_run_id=current_run_id)
    shards = [row for row in shards if int(row.get("shape_sequence") or 0) in completed_sequences]

    destination_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination_dir / "observations.jsonl", observations)
    _write_jsonl(destination_dir / "shards.jsonl", shards)

    return {
        "resumed": True,
        "reason": "compatible-local-checkpoint",
        "resumed_from_optimizer_run_id": old_run_id,
        "optimizer_run_id": current_run_id,
        "completed_shapes": len(observations),
        "completed_shape_sequences": sorted(completed_sequences),
    }


def shape_completed(observation_log: Path, *, pipelines: int, threads: int) -> bool:
    for row in _read_jsonl(observation_log):
        if int(row.get("active_pipelines") or 0) == pipelines and int(row.get("threads_per_pipeline") or 0) == threads:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--source-dir", type=Path, required=True)
    prepare.add_argument("--destination-dir", type=Path, required=True)
    prepare.add_argument("--results-root", type=Path, required=True)
    prepare.add_argument("--mode", required=True)
    prepare.add_argument("--current-run-id", required=True)
    prepare.add_argument("--detector", required=True)
    prepare.add_argument("--runner-label", required=True)
    prepare.add_argument("--runner-budget", type=int, required=True)
    prepare.add_argument("--thread-min", type=int, required=True)
    prepare.add_argument("--thread-max", type=int, required=True)
    prepare.add_argument("--enumeration", required=True)
    prepare.add_argument("--sharding", required=True)
    prepare.add_argument("--pipeline-min", type=int, required=True)
    prepare.add_argument("--pipeline-max", type=int, required=True)
    prepare.add_argument("--allow-thread-oversubscription", choices=("true", "false"), default="false")

    completed = subparsers.add_parser("shape-completed")
    completed.add_argument("--observation-log", type=Path, required=True)
    completed.add_argument("--pipelines", type=int, required=True)
    completed.add_argument("--threads", type=int, required=True)

    args = parser.parse_args()
    if args.command == "shape-completed":
        return 0 if shape_completed(args.observation_log, pipelines=args.pipelines, threads=args.threads) else 1

    result = prepare_resume(
        source_dir=args.source_dir,
        destination_dir=args.destination_dir,
        results_root=args.results_root,
        mode=args.mode,
        current_run_id=args.current_run_id,
        detector=args.detector,
        runner_label=args.runner_label,
        runner_budget=args.runner_budget,
        thread_min=args.thread_min,
        thread_max=args.thread_max,
        enumeration=args.enumeration,
        sharding=args.sharding,
        pipeline_min=args.pipeline_min,
        pipeline_max=args.pipeline_max,
        allow_thread_oversubscription=args.allow_thread_oversubscription == "true",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
