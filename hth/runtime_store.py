#!/usr/bin/env python3
"""Persist HTH detector runtime observations and order multi-detector queues."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hth.contracts import (
    RUNTIME_INDEX_SCHEMA_VERSION,
    RUNTIME_OBSERVATION_SCHEMA_VERSION,
    adapt_runtime_index,
)
MAX_OBSERVATIONS_PER_DETECTOR = 200


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _detector_name(config: Path) -> str:
    return str(_read_json(config).get("detector") or config.stem)


def observation_from_run(run_dir: Path, *, build: dict[str, Any]) -> dict[str, Any]:
    info = _read_json(run_dir / "RUN-INFO.json")
    params = _read_json(run_dir / "parameters.json")
    summary = _read_json(run_dir / "reports" / "summary.json")

    detector = str(info.get("detector") or params.get("detector") or summary.get("detector") or "unknown")
    parameter_space = summary.get("parameter_space") if isinstance(summary.get("parameter_space"), dict) else {}
    progress = summary.get("progress") if isinstance(summary.get("progress"), dict) else {}
    runner = summary.get("runner") if isinstance(summary.get("runner"), dict) else {}
    pipeline = info.get("detector_pipeline") if isinstance(info.get("detector_pipeline"), dict) else {}
    detector_config = Path(str(info.get("detector_config") or params.get("detector_config") or ""))

    elapsed = _as_float(info.get("elapsed_seconds"))
    actual_sets = _as_int(parameter_space.get("actual_parameter_sets") or info.get("actual_parameter_sets"))
    page_evaluations = _as_int(parameter_space.get("actual_page_evaluations") or summary.get("page_evaluation_count"))
    pages = _as_int(parameter_space.get("golden_set_pages") or len(summary.get("page_ordinals", [])))
    eval_rate = _as_float(progress.get("average_eval_rate"))
    if eval_rate is None and elapsed and actual_sets:
        eval_rate = actual_sets / elapsed

    config_sha = _sha256(detector_config) if detector_config.is_file() else None
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    observation_run_id = info.get("run_id") or run_dir.name
    # Run IDs are second-resolution timestamps and concurrent detector pipelines can
    # legitimately create the same run ID.  Include detector identity so one
    # detector cannot overwrite another detector's runtime observation.
    observation_id = f"{build.get('github_run_id', 'local')}:{detector}:{observation_run_id}"
    return {
        "schema_version": RUNTIME_OBSERVATION_SCHEMA_VERSION,
        "observation_id": observation_id,
        "observed_at_utc": info.get("finished_at_utc") or now,
        "run_id": info.get("run_id") or run_dir.name,
        "detector_id": detector,
        "detector_config_sha256": config_sha,
        "golden_set_sha256": info.get("golden_set_sha256") or summary.get("golden_set_sha256"),
        "mode": build.get("mode"),
        "requested_strategy": info.get("requested_strategy") or params.get("requested_strategy"),
        "resolved_strategy": info.get("strategy") or params.get("strategy"),
        "strategy_fallback_reason": info.get("strategy_fallback_reason") or params.get("strategy_fallback_reason"),
        "parameter_set_limit": params.get("limit"),
        "possible_parameter_sets": _as_int(info.get("possible_parameter_sets") or parameter_space.get("possible_parameter_sets")),
        "planned_parameter_sets": _as_int(info.get("planned_parameter_sets") or parameter_space.get("planned_parameter_sets")),
        "actual_parameter_sets": actual_sets,
        "golden_set_pages": pages,
        "actual_page_evaluations": page_evaluations,
        "wall_clock_seconds": elapsed,
        "parameter_sets_per_second": eval_rate,
        "max_dimension": _as_int(params.get("max_dimension")),
        "configured_threads": _as_int(info.get("threads") or params.get("threads")),
        "detector_pipelines": _as_int(pipeline.get("pipeline_count") or pipeline.get("count") or pipeline.get("detector_pipelines")),
        "detector_pipeline_number": _as_int(pipeline.get("pipeline_number") or pipeline.get("number") or pipeline.get("detector_pipeline_number")),
        "pipeline_stagger_minutes": _as_int(pipeline.get("stagger_minutes") or pipeline.get("pipeline_stagger_minutes")),
        "detector_loading_strategy": pipeline.get("loading_strategy"),
        "scheduler_estimate_seconds": _as_float(pipeline.get("runtime_estimate_seconds")),
        "scheduler_estimate_source": pipeline.get("runtime_estimate_source"),
        "runner": {
            "execution_environment": runner.get("execution_environment") or info.get("execution_environment"),
            "runner_environment": runner.get("runner_environment") or info.get("runner_environment"),
            "runner_name": runner.get("runner_name") or info.get("runner_name"),
            "runner_labels": runner.get("github_runner_labels") or info.get("github_runner_labels"),
            "runner_os": runner.get("runner_os") or info.get("runner_os"),
            "runner_arch": runner.get("runner_arch") or info.get("runner_arch"),
            "cpu_model": runner.get("cpu_model") or info.get("cpu_model"),
            "physical_core_count": _as_int(runner.get("physical_core_count") or info.get("physical_core_count")),
            "logical_cpu_count": _as_int(runner.get("logical_cpu_count") or info.get("logical_cpu_count")),
            "available_cpu_count": _as_int(runner.get("available_cpu_count") or info.get("available_cpu_count")),
            "smt_enabled": runner.get("smt_enabled") if runner.get("smt_enabled") is not None else info.get("smt_enabled"),
            "memory_total_bytes": _as_int(runner.get("memory_bytes") or info.get("memory_bytes") or runner.get("memory_total_bytes") or info.get("memory_total_bytes")),
            "memory_gib": _as_float(runner.get("memory_gib") or info.get("memory_gib")),
            "python_version": runner.get("python_version") or info.get("python_version"),
            "opencv_version": runner.get("opencv_version") or info.get("opencv_version"),
            "numpy_version": runner.get("numpy_version") or info.get("numpy_version"),
            "opencv_benchmark_seconds": _as_float(runner.get("opencv_benchmark_seconds") or info.get("opencv_benchmark_seconds")),
        },
        "build": build,
    }


def update_runtime_index(results_root: Path, observations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path = results_root / "runtime-index.json"
    if path.is_file():
        index = adapt_runtime_index(_read_json(path))
    else:
        index = {"schema_version": RUNTIME_INDEX_SCHEMA_VERSION, "observations": [], "latest": {}}

    by_id = {
        str(item.get("observation_id")): item
        for item in index.get("observations", [])
        if isinstance(item, dict) and item.get("observation_id")
    }
    for observation in observations:
        by_id[str(observation["observation_id"])] = observation

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in by_id.values():
        grouped.setdefault(str(item.get("detector_id") or "unknown"), []).append(item)

    trimmed: list[dict[str, Any]] = []
    latest: dict[str, dict[str, Any]] = {}
    for detector, items in grouped.items():
        items.sort(key=lambda row: str(row.get("observed_at_utc") or ""), reverse=True)
        kept = items[:MAX_OBSERVATIONS_PER_DETECTOR]
        trimmed.extend(kept)
        if kept:
            latest[detector] = {
                "observation_id": kept[0].get("observation_id"),
                "wall_clock_seconds": kept[0].get("wall_clock_seconds"),
                "mode": kept[0].get("mode"),
                "resolved_strategy": kept[0].get("resolved_strategy"),
                "configured_threads": kept[0].get("configured_threads"),
                "observed_at_utc": kept[0].get("observed_at_utc"),
            }

    trimmed.sort(key=lambda row: (str(row.get("detector_id")), str(row.get("observed_at_utc") or "")))
    index.update({
        "schema_version": RUNTIME_INDEX_SCHEMA_VERSION,
        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "observations": trimmed,
        "latest": latest,
    })
    _write_json(path, index)
    return index


def _observation_score(
    item: dict[str, Any], *, mode: str, search_strategy: str, threads: int,
    max_dimension: int, golden_set_sha256: str, runner_label: str,
) -> tuple[int, str]:
    score = 0
    reasons: list[str] = []
    if item.get("mode") == mode:
        score += 64; reasons.append("mode")
    if item.get("resolved_strategy") == search_strategy or item.get("requested_strategy") == search_strategy:
        score += 32; reasons.append("strategy")
    if _as_int(item.get("configured_threads")) == threads:
        score += 16; reasons.append("threads")
    if _as_int(item.get("max_dimension")) == max_dimension:
        score += 8; reasons.append("dimension")
    if item.get("golden_set_sha256") == golden_set_sha256:
        score += 4; reasons.append("golden-set")
    runner = item.get("runner") if isinstance(item.get("runner"), dict) else {}
    labels = runner.get("runner_labels") if isinstance(runner.get("runner_labels"), list) else []
    if runner_label and runner_label in labels:
        score += 2; reasons.append("runner")
    return score, "+".join(reasons) or "detector-history"


def select_runtime_observation(
    index: dict[str, Any], detector: str, *, mode: str, search_strategy: str,
    threads: int, max_dimension: int, golden_set_sha256: str, runner_label: str,
) -> tuple[dict[str, Any] | None, str]:
    """Return the canonical runtime observation for one detector/context."""
    candidates = [
        item for item in index.get("observations", [])
        if isinstance(item, dict) and item.get("detector_id") == detector
        and _as_float(item.get("wall_clock_seconds")) is not None
    ]
    if not candidates:
        return None, "no-history"
    ranked = sorted(
        candidates,
        key=lambda item: (
            _observation_score(
                item, mode=mode, search_strategy=search_strategy, threads=threads,
                max_dimension=max_dimension, golden_set_sha256=golden_set_sha256,
                runner_label=runner_label,
            )[0],
            str(item.get("observed_at_utc") or ""),
        ),
        reverse=True,
    )
    best = ranked[0]
    score, reason = _observation_score(
        best, mode=mode, search_strategy=search_strategy, threads=threads,
        max_dimension=max_dimension, golden_set_sha256=golden_set_sha256,
        runner_label=runner_label,
    )
    return best, f"runtime-index:{reason}:score={score}"


def estimate_runtime(
    index: dict[str, Any], detector: str, *, mode: str, search_strategy: str,
    threads: int, max_dimension: int, golden_set_sha256: str, runner_label: str,
) -> tuple[float | None, str]:
    best, source = select_runtime_observation(
        index, detector, mode=mode, search_strategy=search_strategy, threads=threads,
        max_dimension=max_dimension, golden_set_sha256=golden_set_sha256,
        runner_label=runner_label,
    )
    return (_as_float(best.get("wall_clock_seconds")) if best else None), source


def coherent_execution_profile(
    index: dict[str, Any], detector_ids: list[str], *, golden_set_sha256: str = "",
) -> dict[str, Any] | None:
    """Recover a coherent multi-detector execution context from runtime history.

    A generated calibration report combines best-known detector records from
    different builds.  Scheduler recommendations must use one real execution
    build, not infer "mixed" from that heterogeneous collection.
    """
    wanted = set(detector_ids)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in index.get("observations", []):
        if not isinstance(row, dict) or str(row.get("detector_id") or "") not in wanted:
            continue
        if golden_set_sha256 and str(row.get("golden_set_sha256") or "") != golden_set_sha256:
            continue
        build = row.get("build") if isinstance(row.get("build"), dict) else {}
        build_id = str(build.get("github_run_id") or build.get("github_run_number") or "").strip()
        if build_id:
            groups.setdefault(build_id, []).append(row)

    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for build_id, rows in groups.items():
        unique = {str(row.get("detector_id")): row for row in rows}
        values = list(unique.values())
        threads = {_as_int(row.get("configured_threads")) for row in values}
        pipelines = {_as_int(row.get("detector_pipelines")) for row in values}
        loading = {str(row.get("detector_loading_strategy") or "").lower() for row in values}
        modes = {str(row.get("mode") or "") for row in values}
        strategies = {str(row.get("resolved_strategy") or row.get("requested_strategy") or "") for row in values}
        dimensions = {_as_int(row.get("max_dimension")) for row in values}
        if None in threads or len(threads) != 1 or None in pipelines or len(pipelines) != 1:
            continue
        if len(loading) != 1 or len(modes) != 1 or len(strategies) != 1 or len(dimensions) != 1:
            continue
        latest = max(str(row.get("observed_at_utc") or "") for row in values)
        sample = values[0]
        runner = sample.get("runner") if isinstance(sample.get("runner"), dict) else {}
        labels = runner.get("runner_labels") if isinstance(runner.get("runner_labels"), list) else []
        profile = {
            "build_id": build_id,
            "coverage": len(values),
            "threads": next(iter(threads)),
            "pipeline_count": next(iter(pipelines)),
            "loading_strategy": next(iter(loading)) or "lpt",
            "mode": next(iter(modes)),
            "strategy": next(iter(strategies)),
            "max_dimension": next(iter(dimensions)),
            "runner_label": str(labels[0]) if labels else "",
            "observed_at_utc": latest,
        }
        candidates.append((len(values), latest, profile))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _ranked_quality(calibration_index: dict[str, Any], detector: str, golden_sha: str) -> float | None:
    values = []
    for item in calibration_index.get("entries", []):
        if not isinstance(item, dict) or item.get("detector_id") != detector:
            continue
        if golden_sha and item.get("golden_set_sha256") != golden_sha:
            continue
        selection = item.get("selection") if isinstance(item.get("selection"), dict) else {}
        value = _as_float(selection.get("best_avg_iou"))
        if value is not None:
            values.append((str(item.get("created_at_utc") or ""), value))
    return max(values)[1] if values else None


def _runtime_observations_from_calibration_store(calibration_index_path: Path | None) -> list[dict[str, Any]]:
    if calibration_index_path is None or not calibration_index_path.is_file():
        return []
    calibration_index = _read_json(calibration_index_path)
    observations: list[dict[str, Any]] = []
    for entry in calibration_index.get("entries", []):
        if not isinstance(entry, dict):
            continue
        record_path = entry.get("record_path")
        if not record_path:
            continue
        info_path = calibration_index_path.parent / str(record_path) / "RUN-INFO.json"
        if not info_path.is_file():
            continue
        try:
            info = _read_json(info_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        build = entry.get("build") if isinstance(entry.get("build"), dict) else {}
        pipeline = info.get("detector_pipeline") if isinstance(info.get("detector_pipeline"), dict) else {}
        observations.append({
            "observation_id": f"calibration:{entry.get('calibration_id')}",
            "observed_at_utc": info.get("finished_at_utc") or entry.get("created_at_utc"),
            "detector_id": info.get("detector") or entry.get("detector_id"),
            "golden_set_sha256": info.get("golden_set_sha256") or entry.get("golden_set_sha256"),
            "mode": build.get("mode"),
            "requested_strategy": info.get("requested_strategy"),
            "resolved_strategy": info.get("strategy"),
            "wall_clock_seconds": _as_float(info.get("elapsed_seconds")),
            "configured_threads": _as_int(info.get("threads")),
            "max_dimension": None,
            "detector_pipelines": _as_int(pipeline.get("pipeline_count")),
            "runner": {
                "runner_labels": info.get("github_runner_labels") or [],
                "runner_name": info.get("runner_name"),
                "cpu_model": info.get("cpu_model"),
            },
        })
    return observations


def order_configs(
    configs: list[Path], *, loading_strategy: str, runtime_index_path: Path | None,
    calibration_index_path: Path | None, mode: str, search_strategy: str, threads: int,
    max_dimension: int, golden_set_sha256: str, runner_label: str,
) -> list[tuple[Path, float | None, str, float | None]]:
    runtime_index = _read_json(runtime_index_path) if runtime_index_path and runtime_index_path.is_file() else {"observations": []}
    calibration_index = _read_json(calibration_index_path) if calibration_index_path and calibration_index_path.is_file() else {"entries": []}

    # Calibration records contain durable RUN-INFO for every persisted detector.
    # Supplement only detectors missing from runtime-index so historical
    # observation-ID collisions cannot leave LPT reporting `no-history` even
    # though a compatible persisted run exists.  Native runtime observations
    # remain authoritative whenever they are present.
    runtime_rows = [row for row in runtime_index.get("observations", []) if isinstance(row, dict)]
    runtime_detectors = {str(row.get("detector_id")) for row in runtime_rows if row.get("detector_id")}
    calibration_rows = _runtime_observations_from_calibration_store(calibration_index_path)
    runtime_rows.extend(
        row for row in calibration_rows
        if str(row.get("detector_id") or "") not in runtime_detectors
    )
    runtime_index["observations"] = runtime_rows
    rows = []
    known_estimates = []
    for config in configs:
        detector = _detector_name(config)
        estimate, source = estimate_runtime(
            runtime_index, detector, mode=mode, search_strategy=search_strategy, threads=threads,
            max_dimension=max_dimension, golden_set_sha256=golden_set_sha256, runner_label=runner_label,
        )
        if estimate is not None:
            known_estimates.append(estimate)
        quality = _ranked_quality(calibration_index, detector, golden_set_sha256)
        rows.append([config, estimate, source, quality])

    conservative_unknown = max(known_estimates) if known_estimates else 0.0
    if loading_strategy == "lpt":
        rows.sort(key=lambda row: (row[1] if row[1] is not None else conservative_unknown, _detector_name(row[0])), reverse=True)
    elif loading_strategy == "ranked":
        rows.sort(key=lambda row: (row[3] if row[3] is not None else -1.0, _detector_name(row[0])), reverse=True)
    elif loading_strategy != "fifo":
        raise ValueError(f"Unknown detector loading strategy: {loading_strategy}")
    return [(row[0], row[1], row[2], row[3]) for row in rows]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    order = sub.add_parser("order")
    order.add_argument("--config", type=Path, action="append", required=True)
    order.add_argument("--loading-strategy", choices=("lpt", "fifo", "ranked"), required=True)
    order.add_argument("--runtime-index", type=Path)
    order.add_argument("--calibration-index", type=Path)
    order.add_argument("--mode", required=True)
    order.add_argument("--search-strategy", required=True)
    order.add_argument("--threads", type=int, required=True)
    order.add_argument("--max-dimension", type=int, required=True)
    order.add_argument("--golden-set-sha256", required=True)
    order.add_argument("--runner-label", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    rows = order_configs(
        args.config, loading_strategy=args.loading_strategy,
        runtime_index_path=args.runtime_index, calibration_index_path=args.calibration_index,
        mode=args.mode, search_strategy=args.search_strategy, threads=args.threads,
        max_dimension=args.max_dimension, golden_set_sha256=args.golden_set_sha256,
        runner_label=args.runner_label,
    )
    for config, estimate, source, quality in rows:
        estimate_text = "unknown" if estimate is None else f"{estimate:.3f}"
        quality_text = "unknown" if quality is None else f"{quality:.8f}"
        print(f"{config}\t{estimate_text}\t{source}\t{quality_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
