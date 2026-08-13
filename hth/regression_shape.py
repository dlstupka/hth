#!/usr/bin/env python3
"""Resolve regression execution shapes from optimizer intelligence or manual input."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from hth.domain.execution_shape import (
    optimizer_row_matches_workload,
    runner_match_tier,
    select_preferred_shape,
)
from hth.shape_prediction import merge_prediction, predict_shape
from hth.regression.sharding import runner_max_threads


@dataclass(frozen=True)
class RunnerProfile:
    name: str
    label: str
    cpu_model: str
    physical_cores: int | None
    logical_cpus: int


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return "unknown"


def _physical_cores() -> int | None:
    try:
        completed = subprocess.run(
            ["lscpu", "-p=CORE,SOCKET"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    pairs = {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and not line.startswith("#")
    }
    return len(pairs) or None


def current_runner_profile(*, name: str | None = None, label: str | None = None) -> RunnerProfile:
    return RunnerProfile(
        name=(name or os.environ.get("HTH_RUNNER_NAME") or os.environ.get("RUNNER_NAME") or "unknown").strip(),
        label=(label or os.environ.get("HTH_RUNNER_LABEL") or "unknown").strip(),
        cpu_model=_cpu_model(),
        physical_cores=_physical_cores(),
        logical_cpus=max(1, os.cpu_count() or 1),
    )


def parse_manual_shape(value: str) -> tuple[int, int]:
    text = value.strip().lower().replace(" ", "")
    patterns = (
        r"^(\d+)p/(\d+)t$",
        r"^(\d+)/(\d+)$",
        r"^(\d+)p[,x:](\d+)t?$",
    )
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            pipelines, threads = (int(match.group(1)), int(match.group(2)))
            if pipelines < 1 or threads < 1:
                break
            return pipelines, threads
    raise ValueError(f"Manual execution shape must look like 8p/48t (got {value!r})")


def _runner_from_row(row: dict[str, Any]) -> dict[str, Any]:
    runner = row.get("runner")
    return runner if isinstance(runner, dict) else {}


def _normalize_model(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def _row_matches_workload(
    row: dict[str, Any], *, detector: str, detector_sha256: str,
    golden_sha256: str, max_dimension: int,
) -> bool:
    return optimizer_row_matches_workload(
        row, detector=detector, detector_sha256=detector_sha256,
        golden_sha256=golden_sha256, max_dimension=max_dimension,
    )


def _match_tier(row: dict[str, Any], profile: RunnerProfile) -> int | None:
    return runner_match_tier(
        row, name=profile.name, cpu_model=profile.cpu_model,
        physical_cores=profile.physical_cores, logical_cpus=profile.logical_cpus,
    )


def _shape_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "execution_shape": row.get("execution_shape"),
        "pipelines": _as_int(row.get("active_pipelines")),
        "shards": _as_int(row.get("shards")),
        "threads_per_pipeline": _as_int(row.get("threads_per_pipeline")),
        "allocated_threads": _as_int(row.get("allocated_threads")),
        "fastest_wall_clock_seconds": _as_float(row.get("wall_clock_seconds")),
        "parameter_sets_per_second": _as_float(row.get("parameter_sets_per_second")),
        "optimizer_shape_sequence": _as_int(row.get("optimizer_shape_sequence")),
        "optimizer_run_id": row.get("optimizer_run_id"),
    }



def compatible_optimizer_rows(
    *,
    parallelism_index: Path,
    detector_config: Path,
    golden_set: Path,
    max_dimension: int,
) -> tuple[str, list[dict[str, Any]]]:
    detector_config_payload = _read_json(detector_config)
    detector = str(detector_config_payload.get("detector") or detector_config.stem)
    detector_sha256 = _sha256(detector_config)
    golden_sha256 = _sha256(golden_set)
    if not parallelism_index.is_file():
        return detector, []
    index = _read_json(parallelism_index)
    observations = [row for row in index.get("observations", []) if isinstance(row, dict)]
    rows = [
        row for row in observations
        if _row_matches_workload(
            row,
            detector=detector,
            detector_sha256=detector_sha256,
            golden_sha256=golden_sha256,
            max_dimension=max_dimension,
        )
    ]
    if rows:
        return detector, rows

    # Some detector configs explicitly declare that calibration-grid edits do not
    # change their execution characteristics. This lets preferred execution reuse
    # optimizer evidence after a parameter-space expansion without weakening the
    # default config-SHA compatibility rule for other detectors.
    compatibility = str(detector_config_payload.get("optimizer_shape_compatibility") or "").strip().lower()
    if compatibility == "detector-implementation":
        rows = [
            row for row in observations
            if row.get("source") == "execution-optimizer"
            and str(row.get("detector_id") or "") == detector
            and str(row.get("mode") or "") == "full"
            and str(row.get("strategy") or "") == "exhaustive"
            and str(row.get("golden_set_sha256") or "") == golden_sha256
            and (_as_int(row.get("max_dimension")) in (None, max_dimension))
            and _as_int(row.get("possible_parameter_sets")) is not None
            and _as_int(row.get("actual_parameter_sets")) == _as_int(row.get("possible_parameter_sets"))
            and (_as_float(row.get("wall_clock_seconds")) or 0.0) > 0.0
        ]
    return detector, rows


def resolve_predicted_shape(
    *,
    parallelism_index: Path,
    predictions_index: Path | None,
    detector_config: Path,
    golden_set: Path,
    max_dimension: int,
    profile: RunnerProfile,
) -> dict[str, Any] | None:
    detector, rows = compatible_optimizer_rows(
        parallelism_index=parallelism_index,
        detector_config=detector_config,
        golden_set=golden_set,
        max_dimension=max_dimension,
    )
    if not rows:
        return None
    result = predict_shape(
        detector=detector,
        rows=rows,
        target_runner_name=profile.name,
        target_runner_label=profile.label,
        target_cpu_model=profile.cpu_model,
        target_physical_cores=profile.physical_cores,
        target_logical_cpus=profile.logical_cpus,
        predictions_index=predictions_index,
    )
    if result is None:
        return None
    predicted = result.get("predicted_shape") if isinstance(result.get("predicted_shape"), dict) else {}
    result["pipelines"] = int(predicted["pipelines"])
    result["threads_per_pipeline"] = int(predicted["threads_per_pipeline"])
    result["allocated_threads"] = int(predicted["allocated_threads"])
    result["workload"] = {
        "detector_config_sha256": _sha256(detector_config),
        "golden_set_sha256": _sha256(golden_set),
        "max_dimension": max_dimension,
        "mode": "full",
        "strategy": "exhaustive",
    }
    result["source"] = f"predicted-{result.get('confidence', 'unknown')}"
    return result


def resolve_preferred_shape(
    *,
    parallelism_index: Path,
    detector_config: Path,
    golden_set: Path,
    max_dimension: int,
    profile: RunnerProfile,
) -> dict[str, Any] | None:
    detector, compatible_rows = compatible_optimizer_rows(
        parallelism_index=parallelism_index,
        detector_config=detector_config,
        golden_set=golden_set,
        max_dimension=max_dimension,
    )
    if not compatible_rows:
        return None

    candidates_by_tier: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in compatible_rows:
        tier = _match_tier(row, profile)
        if tier is not None:
            candidates_by_tier[tier].append(row)

    selected_tier = next((tier for tier in (0, 1) if candidates_by_tier[tier]), None)
    if selected_tier is None:
        return None
    rows = candidates_by_tier[selected_tier]
    best = select_preferred_shape(_shape_from_row(row) for row in rows)
    if not best:
        return None

    source_names = {0: "exact-runner", 1: "hardware-profile"}
    legacy_workload = any(
        not str(row.get("detector_config_sha256") or "").strip()
        or _as_int(row.get("max_dimension")) is None
        for row in rows
    )
    source = source_names[selected_tier] + ("-legacy-workload" if legacy_workload else "")
    return {
        "detector": detector,
        "pipelines": int(best["pipelines"]),
        "threads_per_pipeline": int(best["threads_per_pipeline"]),
        "allocated_threads": int(best.get("allocated_threads") or int(best["pipelines"]) * int(best["threads_per_pipeline"])),
        "parameter_sets_per_second": best.get("parameter_sets_per_second"),
        "source": source,
        "matched_observations": len(rows),
        "runner_name": profile.name,
        "runner_label": profile.label,
        "logical_cpus": profile.logical_cpus,
        "cpu_model": profile.cpu_model,
    }


def _print_shell(result: dict[str, Any]) -> None:
    print(int(result["pipelines"]), int(result["threads_per_pipeline"]), str(result.get("source") or "unknown"))


def _append_github_env(path: Path | None, values: dict[str, Any]) -> None:
    if not path:
        return
    with path.open("a", encoding="utf-8") as stream:
        for key, value in values.items():
            stream.write(f"{key}={value}\n")


def resolve_workflow_shape(
    *, shape_mode: str, regression_mode: str, strategy: str, limit: str | None,
    detector: str, manual_shape: str | None, parallelism_index: Path,
    predictions_index: Path | None, detector_config_root: Path, golden_set: Path,
    max_dimension: int, profile: RunnerProfile, prediction_out: Path | None,
    runner_budget: int | None = None,
) -> dict[str, Any]:
    """Resolve all workflow shape policy in Python; YAML only supplies inputs."""
    budget = max(1, runner_budget or runner_max_threads(profile.label, profile.logical_cpus))

    def exact(pipelines: int, threads: int, source: str, prediction_file: Path | None = None) -> dict[str, Any]:
        allocated = pipelines * threads
        if allocated > budget:
            raise ValueError(
                f"Execution shape {pipelines}p/{threads}t allocates {allocated} "
                f"threads against detected runner budget {budget}"
            )
        result = {
            "exact": True, "pipelines": pipelines, "threads_per_pipeline": threads,
            "allocated_threads": allocated, "runner_budget": budget, "source": source,
        }
        if prediction_file:
            result["prediction_file"] = str(prediction_file)
        return result

    mode = (shape_mode or "auto").strip().lower()
    if mode == "auto":
        return {"exact": False, "source": "auto", "runner_budget": budget}

    if mode == "manual":
        if not manual_shape:
            raise ValueError("Manual execution shape is required (example: 8p/48t)")
        pipelines, threads = parse_manual_shape(manual_shape)
        return exact(pipelines, threads, "manual")

    if mode != "preferred":
        raise ValueError(f"Unknown execution shape mode: {shape_mode}")

    if regression_mode != "full" or strategy != "exhaustive" or str(limit or "").strip():
        return {"exact": False, "source": "auto-fallback-incompatible-workload", "runner_budget": budget}
    if detector == "all":
        return {"exact": False, "source": "auto-fallback-all", "runner_budget": budget}

    detector_config = detector_config_root / f"{detector}.json"
    preferred = resolve_preferred_shape(
        parallelism_index=parallelism_index, detector_config=detector_config,
        golden_set=golden_set, max_dimension=max_dimension, profile=profile,
    )
    if preferred:
        return exact(
            int(preferred["pipelines"]), int(preferred["threads_per_pipeline"]),
            f"preferred-{preferred['source']}",
        )

    predicted = resolve_predicted_shape(
        parallelism_index=parallelism_index, predictions_index=predictions_index,
        detector_config=detector_config, golden_set=golden_set,
        max_dimension=max_dimension, profile=profile,
    )
    if predicted:
        if prediction_out:
            prediction_out.parent.mkdir(parents=True, exist_ok=True)
            prediction_out.write_text(json.dumps(predicted, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return exact(
            int(predicted["pipelines"]), int(predicted["threads_per_pipeline"]),
            str(predicted.get("source") or "predicted"), prediction_out,
        )
    return {"exact": False, "source": "auto-fallback-no-shape-history", "runner_budget": budget}


def workflow_shape_env(result: dict[str, Any]) -> dict[str, Any]:
    env = {
        "HTH_EXACT_EXECUTION_SHAPE_SOURCE": result.get("source", "unknown"),
        "HTH_EXECUTION_THREAD_BUDGET": result.get("runner_budget", ""),
        "HTH_EXECUTION_SHAPE_RESOLVED": "exact" if result.get("exact") else "auto",
    }
    if result.get("exact"):
        env.update({
            "THREADS": int(result["threads_per_pipeline"]),
            "DETECTOR_PIPELINES": int(result["pipelines"]),
            "SHARDS": int(result["pipelines"]),
            "HTH_EXACT_EXECUTION_SHAPE": "1",
            "HTH_ALLOW_THREAD_OVERSUBSCRIPTION": "false",
        })
    if result.get("prediction_file"):
        env["HTH_SHAPE_PREDICTION_FILE"] = result["prediction_file"]
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    manual = sub.add_parser("manual", help="Parse a manual Np/Mt execution shape")
    manual.add_argument("shape")

    preferred = sub.add_parser("preferred", help="Resolve persisted optimizer preference for this runner profile")
    preferred.add_argument("--parallelism-index", type=Path, required=True)
    preferred.add_argument("--detector-config", type=Path, required=True)
    preferred.add_argument("--golden-set", type=Path, required=True)
    preferred.add_argument("--max-dimension", type=int, required=True)
    preferred.add_argument("--runner-name")
    preferred.add_argument("--runner-label")

    predicted = sub.add_parser("predicted", help="Predict a shape from same-detector optimizer history when no compatible preference exists")
    predicted.add_argument("--parallelism-index", type=Path, required=True)
    predicted.add_argument("--predictions-index", type=Path)
    predicted.add_argument("--prediction-out", type=Path)
    predicted.add_argument("--detector-config", type=Path, required=True)
    predicted.add_argument("--golden-set", type=Path, required=True)
    predicted.add_argument("--max-dimension", type=int, required=True)
    predicted.add_argument("--runner-name")
    predicted.add_argument("--runner-label")

    record = sub.add_parser("record-prediction", help="Merge a generated shape prediction into the persistent prediction history")
    record.add_argument("--prediction-file", type=Path, required=True)
    record.add_argument("--predictions-index", type=Path, required=True)

    workflow = sub.add_parser("workflow-resolve", help="Resolve workflow execution shape and write GitHub environment")
    workflow.add_argument("--shape-mode", required=True)
    workflow.add_argument("--regression-mode", required=True)
    workflow.add_argument("--strategy", required=True)
    workflow.add_argument("--limit", default="")
    workflow.add_argument("--detector", required=True)
    workflow.add_argument("--manual-shape", default="")
    workflow.add_argument("--parallelism-index", type=Path, required=True)
    workflow.add_argument("--predictions-index", type=Path)
    workflow.add_argument("--detector-config-root", type=Path, required=True)
    workflow.add_argument("--golden-set", type=Path, required=True)
    workflow.add_argument("--max-dimension", type=int, required=True)
    workflow.add_argument("--runner-name")
    workflow.add_argument("--runner-label")
    workflow.add_argument("--prediction-out", type=Path)
    workflow.add_argument("--github-env", type=Path)

    args = parser.parse_args()
    if args.command == "workflow-resolve":
        profile = current_runner_profile(name=args.runner_name, label=args.runner_label)
        result = resolve_workflow_shape(
            shape_mode=args.shape_mode, regression_mode=args.regression_mode,
            strategy=args.strategy, limit=args.limit, detector=args.detector,
            manual_shape=args.manual_shape, parallelism_index=args.parallelism_index,
            predictions_index=args.predictions_index,
            detector_config_root=args.detector_config_root, golden_set=args.golden_set,
            max_dimension=args.max_dimension, profile=profile,
            prediction_out=args.prediction_out,
        )
        _append_github_env(args.github_env, workflow_shape_env(result))
        if result.get("exact"):
            print(
                f"Resolved execution shape: {result['pipelines']}p/"
                f"{result['threads_per_pipeline']}t ({result['source']}; "
                f"{result['allocated_threads']}/{result['runner_budget']} threads)"
            )
        else:
            print(f"Execution shape: auto planner ({result['source']})")
        return 0

    if args.command == "manual":
        pipelines, threads = parse_manual_shape(args.shape)
        _print_shell({"pipelines": pipelines, "threads_per_pipeline": threads, "source": "manual"})
        return 0

    if args.command == "record-prediction":
        prediction = _read_json(args.prediction_file)
        merge_prediction(args.predictions_index, prediction)
        return 0

    profile = current_runner_profile(name=args.runner_name, label=args.runner_label)
    if args.command == "predicted":
        result = resolve_predicted_shape(
            parallelism_index=args.parallelism_index,
            predictions_index=args.predictions_index,
            detector_config=args.detector_config,
            golden_set=args.golden_set,
            max_dimension=args.max_dimension,
            profile=profile,
        )
        if result is None:
            return 2
        if args.prediction_out:
            args.prediction_out.parent.mkdir(parents=True, exist_ok=True)
            args.prediction_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _print_shell(result)
        return 0

    result = resolve_preferred_shape(
        parallelism_index=args.parallelism_index,
        detector_config=args.detector_config,
        golden_set=args.golden_set,
        max_dimension=args.max_dimension,
        profile=profile,
    )
    if result is None:
        return 2
    _print_shell(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
