#!/usr/bin/env python3
"""Build execution-optimizer intelligence, run-local tables, and processing profiles."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from hth.persistence import canonical_index_path, readable_index_path, read_json as _read_json, atomic_write_json as _write_json, load_index, write_index
from hth.optimizer_history import completed_run_records, persist_completed_run
from typing import Any, Iterable

from hth.contracts import OPTIMIZER_INDEX_SCHEMA_VERSION, adapt_optimizer_index
from hth.domain.execution_shape import DETERMINISTIC_OPTIMIZER_STRATEGIES, select_preferred_shape




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


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _duration(seconds: Any) -> str:
    value = _as_float(seconds)
    if value is None:
        return "unknown"
    total = max(0, int(round(value)))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def _format_bytes(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return "unknown"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for candidate in units:
        unit = candidate
        if abs(number) < 1024 or candidate == units[-1]:
            break
        number /= 1024
    return f"{number:.1f} {unit}"


def _runner_key(row: dict[str, Any]) -> str:
    runner = row.get("runner") if isinstance(row.get("runner"), dict) else {}
    labels = runner.get("runner_labels")
    if isinstance(labels, list):
        label_text = ",".join(sorted(str(item) for item in labels))
    else:
        label_text = str(labels or "")
    identity = {
        "runner_label": runner.get("runner_label"),
        "runner_name": runner.get("runner_name"),
        "labels": label_text,
        "cpu_model": runner.get("cpu_model"),
        "logical_cpu_count": runner.get("logical_cpu_count"),
    }
    return _canonical_hash(identity)[:16]


def _runner_title(row: dict[str, Any]) -> str:
    runner = row.get("runner") if isinstance(row.get("runner"), dict) else {}
    label = str(runner.get("runner_label") or "unknown")
    name = str(runner.get("runner_name") or "").strip()
    logical = _as_int(runner.get("logical_cpu_count"))
    suffix = f" — {name}" if name and name.lower() != label.lower() else ""
    cpu_suffix = f" ({logical} vCPU)" if logical else ""
    return f"{label}{suffix}{cpu_suffix}"


def _runner_labels(row: dict[str, Any]) -> str:
    runner = row.get("runner") if isinstance(row.get("runner"), dict) else {}
    labels = runner.get("runner_labels")
    if isinstance(labels, list):
        return ", ".join(str(item) for item in labels)
    return str(labels or runner.get("runner_label") or "unknown")


def _comparable(rows: Iterable[dict[str, Any]], detector_id: str, optimizer_run_id: str | None = None, optimizer_run_ids: set[str] | None = None) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("detector_id")) != detector_id:
            continue
        if row.get("source") != "execution-optimizer":
            continue
        if optimizer_run_id is not None and str(row.get("optimizer_run_id")) != str(optimizer_run_id):
            continue
        if optimizer_run_ids is not None and str(row.get("optimizer_run_id") or "") not in optimizer_run_ids:
            continue
        if row.get("mode") != "full":
            continue
        strategy = str(row.get("strategy") or "")
        if strategy not in DETERMINISTIC_OPTIMIZER_STRATEGIES:
            continue
        actual_sets = _as_int(row.get("actual_parameter_sets"))
        possible_sets = _as_int(row.get("possible_parameter_sets"))
        if actual_sets is None or actual_sets <= 0:
            continue
        # Normal exhaustive observations must cover the complete declared grid.
        # Execution optimizer observations instead use a canonical bounded
        # benchmark workload; their explicit benchmark budget is part of the
        # workload/compatibility identity and must be fully consumed.
        benchmark = _as_int(row.get("optimizer_benchmark_parameter_sets"))
        if benchmark is not None and benchmark > 0 and row.get("source") == "execution-optimizer":
            if actual_sets != min(possible_sets or benchmark, benchmark):
                continue
        elif strategy in {"exhaustive", "exhaustive-with-zombies"} and actual_sets != possible_sets:
            continue
        if (_as_float(row.get("wall_clock_seconds")) or 0) <= 0:
            continue
        result.append(row)
    return result


def _shape_from_row(row: dict[str, Any], *, baseline_wall: float | None, observation_count: int = 1, median_wall: float | None = None) -> dict[str, Any]:
    wall = _as_float(row.get("wall_clock_seconds")) or 0.0
    metrics = row.get("runner_metrics") if isinstance(row.get("runner_metrics"), dict) else {}
    return {
        "execution_shape": row.get("execution_shape"),
        "pipelines": _as_int(row.get("active_pipelines")),
        "shards": _as_int(row.get("shards")),
        "threads_per_pipeline": _as_int(row.get("threads_per_pipeline")),
        "allocated_threads": _as_int(row.get("allocated_threads")),
        "observation_count": observation_count,
        "fastest_wall_clock_seconds": wall,
        "median_wall_clock_seconds": wall if median_wall is None else median_wall,
        "startup_overhead_seconds": _as_float(row.get("startup_overhead_seconds")),
        "startup_overhead_included_in_wall_clock": bool(row.get("startup_overhead_included_in_wall_clock")),
        "parameter_sets_per_second": _as_float(row.get("parameter_sets_per_second")),
        "page_evaluations_per_second": _as_float(row.get("page_evaluations_per_second")),
        "effective_acceleration": _as_float(row.get("effective_acceleration")),
        "parallel_efficiency": _as_float(row.get("parallel_efficiency")),
        "observed_speedup_vs_one_pipeline": (baseline_wall / wall) if baseline_wall and wall > 0 else None,
        "optimizer_shape_sequence": _as_int(row.get("optimizer_shape_sequence")),
        "optimizer_run_id": str(row.get("optimizer_run_id")) if row.get("optimizer_run_id") is not None else None,
        "runner_metrics": metrics,
    }



def build_optimizer_index(parallelism_index: dict[str, Any], detector_id: str, optimizer_run_id: str | None = None, optimizer_run_ids: set[str] | None = None) -> dict[str, Any]:
    rows = _comparable(
        (row for row in parallelism_index.get("observations", []) if isinstance(row, dict)),
        detector_id,
        optimizer_run_id,
        optimizer_run_ids,
    )
    runner_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        # Historical optimizer intelligence is compatibility scoped, not merely
        # runner-label scoped.  This prevents a changed Golden Set/grid from
        # contaminating a detector's execution preference while still allowing
        # compatible shapes from separate runs to coalesce.
        group_key = str(row.get("compatibility_key") or _runner_key(row))
        runner_groups.setdefault(group_key, []).append(row)

    runners: list[dict[str, Any]] = []
    for runner_key, runner_rows in runner_groups.items():
        one_pipeline = [row for row in runner_rows if _as_int(row.get("active_pipelines")) == 1]
        baseline_wall = min((_as_float(row.get("wall_clock_seconds")) for row in one_pipeline if _as_float(row.get("wall_clock_seconds")) is not None), default=None)

        shapes: list[dict[str, Any]] = []
        if optimizer_run_id is not None:
            # A run-local report must show exactly the shapes exercised in this execution.
            for row in sorted(runner_rows, key=lambda item: int(item.get("optimizer_shape_sequence") or 0)):
                shapes.append(_shape_from_row(row, baseline_wall=baseline_wall))
        else:
            shape_groups: dict[str, list[dict[str, Any]]] = {}
            for row in runner_rows:
                shape_groups.setdefault(str(row.get("execution_shape") or "unknown"), []).append(row)
            for shape_rows in shape_groups.values():
                walls = sorted(float(row["wall_clock_seconds"]) for row in shape_rows)
                fastest = min(shape_rows, key=lambda row: float(row["wall_clock_seconds"]))
                shapes.append(_shape_from_row(fastest, baseline_wall=baseline_wall, observation_count=len(shape_rows), median_wall=statistics.median(walls)))
            shapes.sort(key=lambda shape: (int(shape.get("pipelines") or 0), int(shape.get("threads_per_pipeline") or 0)))

        best_shape = select_preferred_shape(shapes)
        sample = runner_rows[0]
        sample_runner = sample.get("runner") if isinstance(sample.get("runner"), dict) else {}
        runners.append({
            "runner_key": _runner_key(sample),
            "compatibility_key": sample.get("compatibility_key"),
            "workload_key": sample.get("workload_key"),
            "runner_label": sample_runner.get("runner_label"),
            "runner_title": _runner_title(sample),
            "runner_labels": _runner_labels(sample),
            "runner_specs": {
                "cpu_model": sample_runner.get("cpu_model"),
                "physical_core_count": _as_int(sample_runner.get("physical_core_count")),
                "logical_cpu_count": _as_int(sample_runner.get("logical_cpu_count")),
                "memory_gib": _as_float(sample_runner.get("memory_gib")),
            },
            "best_shape": best_shape,
            "shapes": shapes,
        })

    all_shapes = [shape for runner in runners for shape in runner.get("shapes", [])]
    best_across_runners = select_preferred_shape(all_shapes)
    return {
        "schema_version": OPTIMIZER_INDEX_SCHEMA_VERSION,
        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "detector_id": detector_id,
        "optimizer_run_id": optimizer_run_id,
        "runner_count": len(runners),
        "observation_count": len(rows),
        "best_across_runners": best_across_runners,
        "runners": runners,
    }




def _filter_optimizer_index_to_compatibility(
    index: dict[str, Any],
    compatibility_keys: set[str],
) -> dict[str, Any]:
    """Keep only workload/runner groups compatible with the current run."""
    if not compatibility_keys:
        return index
    filtered = dict(index)
    runners = [
        runner for runner in index.get("runners", [])
        if isinstance(runner, dict)
        and str(runner.get("compatibility_key") or "") in compatibility_keys
    ]
    filtered["runners"] = runners
    filtered["runner_count"] = len(runners)
    filtered["observation_count"] = sum(
        int(shape.get("observation_count") or 1)
        for runner in runners
        for shape in runner.get("shapes", [])
        if isinstance(shape, dict)
    )
    all_shapes = [
        shape
        for runner in runners
        for shape in runner.get("shapes", [])
        if isinstance(shape, dict)
    ]
    filtered["best_across_runners"] = select_preferred_shape(all_shapes)
    return filtered


PREFERRED_SHAPE_RANGE_THRESHOLD_PCT = 2.0

def _preferred_shape_range(runner: dict[str, Any]) -> str:
    """Summarize measured shapes within 2% of the runner's best observed throughput."""
    shapes = [shape for shape in runner.get("shapes", []) if isinstance(shape, dict)]
    rates = [(shape, _as_float(shape.get("parameter_sets_per_second"))) for shape in shapes]
    rates = [(shape, rate) for shape, rate in rates if rate is not None]
    if not rates:
        return "—"
    best_rate = max(rate for _, rate in rates)
    floor = best_rate * (1.0 - PREFERRED_SHAPE_RANGE_THRESHOLD_PCT / 100.0)
    near_best = [shape for shape, rate in rates if rate >= floor]
    measured_shapes = sorted(
        {
            (_as_int(shape.get("pipelines")), _as_int(shape.get("threads_per_pipeline")))
            for shape in near_best
            if _as_int(shape.get("pipelines")) is not None
            and _as_int(shape.get("threads_per_pipeline")) is not None
        },
        key=lambda item: (item[0], item[1]),
    )
    if not measured_shapes:
        return "—"
    return ", ".join(f"{pipelines}p/{threads}t" for pipelines, threads in measured_shapes)

def _run_metadata_lookup(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = index.get("run_metadata_by_id")
    return value if isinstance(value, dict) else {}


def _display_search_method(value: Any) -> str:
    """Normalize optimizer search-method names for report presentation.

    Historical metadata may contain the legacy ``binary`` name either alone or
    inside a comma-separated/multi-value representation.  Normalize each token
    without rewriting persisted optimizer evidence.
    """
    if isinstance(value, (list, tuple, set)):
        methods = [str(item).strip() for item in value if str(item).strip()]
    else:
        methods = [item.strip() for item in str(value or "unknown").split(",") if item.strip()]
    normalized = ["powers-of-2" if method == "binary" else method for method in methods]
    return ", ".join(normalized) if normalized else "unknown"


def _shape_search_method(index: dict[str, Any], shape: dict[str, Any]) -> str:
    run_id = shape.get("optimizer_run_id")
    metadata = _run_metadata_lookup(index).get(str(run_id)) if run_id is not None else None
    if isinstance(metadata, dict):
        return _display_search_method(metadata.get("pipeline_enumeration"))
    return "legacy"


def _shape_optimization_time(index: dict[str, Any], shape: dict[str, Any]) -> str:
    run_id = shape.get("optimizer_run_id")
    metadata = _run_metadata_lookup(index).get(str(run_id)) if run_id is not None else None
    if isinstance(metadata, dict) and metadata.get("optimization_wall_seconds") is not None:
        return _duration(metadata.get("optimization_wall_seconds"))
    if run_id is not None:
        walls = [
            _as_float(candidate.get("fastest_wall_clock_seconds"))
            for runner in index.get("runners", [])
            for candidate in runner.get("shapes", [])
            if str(candidate.get("optimizer_run_id") or "") == str(run_id)
        ]
        walls = [wall for wall in walls if wall is not None]
        if walls:
            return _duration(sum(walls))
    return "—"


def _represented_search_methods(index: dict[str, Any]) -> str:
    methods: set[str] = set()
    metadata_by_id = _run_metadata_lookup(index)
    for runner in index.get("runners", []):
        for shape in runner.get("shapes", []):
            run_id = shape.get("optimizer_run_id")
            metadata = metadata_by_id.get(str(run_id)) if run_id is not None else None
            if isinstance(metadata, dict) and metadata.get("pipeline_enumeration"):
                methods.add(_display_search_method(metadata["pipeline_enumeration"]))
    return ", ".join(sorted(methods)) if methods else "legacy"



def _shape_prediction_coverage(index: dict[str, Any]) -> dict[str, Any]:
    anchors = sorted({
        _as_int((runner.get("runner_specs") or {}).get("logical_cpu_count"))
        for runner in index.get("runners", [])
        if isinstance(runner, dict)
        and isinstance(runner.get("best_shape"), dict)
        and _as_int((runner.get("runner_specs") or {}).get("logical_cpu_count")) is not None
    })
    anchors = [value for value in anchors if value is not None]
    count = len(anchors)
    if count == 0:
        readiness = "none"
        desired = "missing: at least one completed optimizer run"
    elif count == 1:
        readiness = "low"
        desired = "missing: a second vCPU size to establish shape scaling"
    elif count == 2:
        readiness = "moderate"
        desired = "desired: a third vCPU size to validate interpolation/extrapolation"
    else:
        readiness = "high"
        desired = "basic vCPU shape coverage is sufficient; additional runner sizes are optional validation"

    history = index.get("prediction_history")
    predictions = history if isinstance(history, list) else []
    verified = sum(1 for row in predictions if isinstance(row, dict) and row.get("status") == "verified")
    pending = sum(1 for row in predictions if isinstance(row, dict) and row.get("status") != "verified")
    return {
        "anchors": anchors,
        "readiness": readiness,
        "desired": desired,
        "verified_predictions": verified,
        "pending_predictions": pending,
    }


def _render_shape_prediction_coverage(index: dict[str, Any]) -> list[str]:
    coverage = _shape_prediction_coverage(index)
    anchors = ", ".join(str(value) for value in coverage["anchors"]) if coverage["anchors"] else "none"
    checks = f"{coverage['verified_predictions']} verified / {coverage['pending_predictions']} pending"
    return [
        f"**Shape-prediction coverage:** vCPU anchors `{anchors}`; readiness **{coverage['readiness']}**; prediction checks **{checks}**.",
        f"**Desired / missing optimization data:** {coverage['desired']}.",
        "",
    ]


def _render_preferred_configuration(index: dict[str, Any]) -> list[str]:
    lines = [
        "Compatible completed optimizer runs are coalesced by detector, workload, and concrete runner profile. Repeated shapes retain all observations; the preferred shape is selected canonically by throughput, then lower resource use for throughput-equivalent shapes.",
        "",
        "| Detector | Runner | CPU | Physical | Logical | RAM | Preferred pipelines | Threads / pipeline | Preferred shape range (≤2%) | Search method | Optimization time | Allocated | Sets/s | Shape time | Observations |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for runner in sorted(index.get("runners", []), key=lambda item: str(item.get("runner_title") or "")):
        best = runner.get("best_shape") if isinstance(runner.get("best_shape"), dict) else {}
        if not best:
            continue
        specs = runner.get("runner_specs") if isinstance(runner.get("runner_specs"), dict) else {}
        rate = _as_float(best.get("parameter_sets_per_second"))
        memory = _as_float(specs.get("memory_gib"))
        lines.append(
            "| {detector} | {runner} | {cpu} | {physical} | {logical} | {memory} | {pipelines} | {threads} | {shape_range} | {search_method} | {optimization_time} | {allocated} | {rate} | {wall} | {observations} |".format(
                detector=index.get("detector_id") or "unknown",
                runner=runner.get("runner_title") or "unknown",
                cpu=str(specs.get("cpu_model") or "—").replace("|", "/"),
                physical=specs.get("physical_core_count") or "—",
                logical=specs.get("logical_cpu_count") or "—",
                memory=f"{memory:.1f} GiB" if memory is not None else "—",
                pipelines=best.get("pipelines") or "?",
                threads=best.get("threads_per_pipeline") or "?",
                shape_range=_preferred_shape_range(runner),
                search_method=_shape_search_method(index, best),
                optimization_time=_shape_optimization_time(index, best),
                allocated=best.get("allocated_threads") or "?",
                rate=f"{rate:.2f}" if rate is not None else "unknown",
                wall=_duration(best.get("fastest_wall_clock_seconds")),
                observations=sum(
                    int(shape.get("observation_count") or 1)
                    for shape in runner.get("shapes", [])
                    if isinstance(shape, dict)
                ),
            )
        )
    lines.extend([
        "",
        "**Search method legend:** `adaptive` = sparse wide-range search with local refinement around the measured peak and ≤2% preferred-shape boundaries; `powers-of-2` = logarithmic power-of-two pipeline sweep; `exhaustive` = every legal pipeline count in the requested range.",
        "",
    ])
    lines.extend(_render_shape_prediction_coverage(index))
    return lines


def _nav_slug(text: str) -> str:
    slug = "-".join(part for part in re.sub(r"[^a-zA-Z0-9]+", " ", text).lower().split() if part)
    return slug or "section"


def _render_optimizer_navigation(*, detectors: list[str] | None = None) -> list[str]:
    lines = [
        '<a id="table-of-contents"></a>',
        "",
        "<details open>",
        "<summary><strong>Navigation</strong></summary>",
        "",
        "- [Preferred Detector Run Configuration](#preferred-detector-run-configuration)",
        "- [Detector Run Profile Plot](#detector-run-profile-plot)",
    ]
    if detectors:
        for detector in detectors:
            lines.append(f"  - [{detector}](#detector-run-profile-{_nav_slug(detector)})")
    lines.append("- [Detector Pipeline-Thread Shape Optimization Data](#detector-pipeline-thread-shape-optimization-data)")
    if detectors:
        for detector in detectors:
            lines.append(f"  - [{detector}](#detector-shape-data-{_nav_slug(detector)})")
    lines.extend(["", "</details>", ""])
    return lines


def _render_shape_table(index: dict[str, Any]) -> list[str]:
    lines = [
        "This table contains measurements from this optimizer execution only. Bold identifies this run’s measured throughput winner; the preferred configuration above is selected from all compatible coalesced optimizer evidence.",
        "",
        "| Runner | Pipelines | Shards | Threads / pipeline | Allocated | Wall | Startup overhead | Sets/s | Speedup | Δ from run best | Avg load | Peak load | Avg CPU | Peak RAM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows: list[tuple[str, dict[str, Any], bool]] = []
    for runner in index.get("runners", []):
        best_shape = (runner.get("best_shape") or {}).get("execution_shape")
        best_seq = (runner.get("best_shape") or {}).get("optimizer_shape_sequence")
        for shape in runner.get("shapes", []):
            best = shape.get("execution_shape") == best_shape and shape.get("optimizer_shape_sequence") == best_seq
            rows.append((str(runner.get("runner_title") or "unknown"), shape, best))
    rows.sort(key=lambda item: (int(item[1].get("pipelines") or 0), int(item[1].get("threads_per_pipeline") or 0), item[0], int(item[1].get("optimizer_shape_sequence") or 10**9)))
    best_rates: dict[str, float] = {}
    for runner_title, shape, _ in rows:
        rate = _as_float(shape.get("parameter_sets_per_second"))
        if rate is not None:
            best_rates[runner_title] = max(best_rates.get(runner_title, rate), rate)
    for runner_title, shape, best in rows:
        metrics = shape.get("runner_metrics") if isinstance(shape.get("runner_metrics"), dict) else {}
        speedup = _as_float(shape.get("observed_speedup_vs_one_pipeline"))
        rate = _as_float(shape.get("parameter_sets_per_second"))
        avg_load = _as_float(metrics.get("avg_load1"))
        peak_load = _as_float(metrics.get("peak_load1"))
        avg_cpu = _as_float(metrics.get("avg_cpu_pct"))
        peak_ram = metrics.get("peak_ram_used_bytes")
        best_rate = best_rates.get(runner_title)
        delta_from_best = ((rate / best_rate) - 1.0) * 100.0 if rate is not None and best_rate else None
        # The best shape is exactly zero; all other valid deltas are non-positive by definition.
        if delta_from_best is not None and abs(delta_from_best) < 0.005:
            delta_from_best = 0.0
        lines.append(
            "| {runner} | {pipelines} | {shards} | {threads} | {allocated} | {wall} | {startup} | {rate} | {speedup} | {delta_from_best} | {avg_load} | {peak_load} | {avg_cpu} | {peak_ram} |".format(
                runner=f"**{runner_title}**" if best else runner_title,
                pipelines=shape.get("pipelines") or "?",
                shards=shape.get("shards") or "?",
                threads=shape.get("threads_per_pipeline") or "?",
                allocated=shape.get("allocated_threads") or "?",
                wall=_duration(shape.get("fastest_wall_clock_seconds")),
                startup=_duration(shape.get("startup_overhead_seconds")) if shape.get("startup_overhead_seconds") is not None else "—",
                rate=f"{rate:.2f}" if rate is not None else "unknown",
                speedup=f"{speedup:.2f}×" if speedup is not None else "—",
                delta_from_best=f"{delta_from_best:.2f}%" if delta_from_best is not None else "—",
                avg_load=f"{avg_load:.1f}" if avg_load is not None else "—",
                peak_load=f"{peak_load:.1f}" if peak_load is not None else "—",
                avg_cpu=f"{avg_cpu:.1f}%" if avg_cpu is not None else "—",
                peak_ram=_format_bytes(peak_ram) if peak_ram is not None else "—",
            )
        )
    lines.extend([
        "",
        "**Startup-overhead note:** executor startup is measured from `run-detector-regressions` entry through detector lifecycle preparation, planning, shared learned-evidence resolution/preparation, and initial queue setup before pipeline fan-out. It remains included in **Wall** and therefore in shape-level **Sets/s** as a constant reminder of incurred end-to-end cost. Per-shard parameter-set throughput is timed after fan-out and does not include this pre-fan-out startup overhead.",
        "",
    ])
    return lines


def render_markdown(index: dict[str, Any], run_metadata: dict[str, Any] | None = None, preferred_index: dict[str, Any] | None = None) -> str:
    lines = [
        "### Execution optimizer summary",
        "",
        f"Detector: `{index.get('detector_id')}`  ",
    ]
    if index.get("optimizer_run_id") is not None:
        metadata = run_metadata or {}
        resumed_from = metadata.get("resumed_from_optimizer_run_id")
        if resumed_from:
            lines.append(
                f"Optimizer run: **{index.get('optimizer_run_id')}** — resumed from optimizer run **{resumed_from}**; "
                "execution data below contains shapes completed in this execution or reused from that compatible local checkpoint; "
                "the preferred configuration may use all compatible completed optimizer evidence."
            )
        else:
            lines.append(f"Optimizer run: **{index.get('optimizer_run_id')}** — execution data below contains only shapes completed in this execution; the preferred configuration may use all compatible completed optimizer evidence.")
    lines.extend(["", *_render_optimizer_navigation()])

    lines.extend([
        '<a id="preferred-detector-run-configuration"></a>',
        "<details open>",
        "<summary><strong>1. Preferred Detector Run Configuration</strong></summary>",
        "",
    ])
    lines.extend(_render_preferred_configuration(preferred_index or index))
    lines.extend(["</details>", "", "[↑ Back to Navigation](#table-of-contents)", ""])

    lines.extend([
        '<a id="detector-run-profile-plot"></a>',
        "<details open>",
        "<summary><strong>2. Detector Run Profile Plot</strong></summary>",
        "",
        "Compatible completed measurements are plotted as detector pipelines versus parameter sets/second; thread count is annotated at each measured shape.",
        f"**Search method:** `{_display_search_method((run_metadata or {}).get('pipeline_enumeration')) if (run_metadata or {}).get('pipeline_enumeration') else _represented_search_methods(index)}`",
        "",
        "![Detector Run Profile Plot](heatmap.svg)",
        "",
        "</details>",
        "",
        "[↑ Back to Navigation](#table-of-contents)",
        "",
        '<a id="detector-pipeline-thread-shape-optimization-data"></a>',
        "<details>",
        "<summary><strong>3. Detector Pipeline-Thread Shape Optimization Data</strong></summary>",
        "",
    ])
    if index.get("optimizer_run_id") is not None:
        if run_metadata and run_metadata.get("resumed_from_optimizer_run_id"):
            lines.extend(["Shapes completed in this execution or reused from its compatible checkpoint are shown below.", ""])
        else:
            lines.extend(["Shapes completed in this execution are shown below.", ""])
    else:
        lines.extend(["Coalesced compatible shape measurements are shown below.", ""])
    lines.extend(_render_shape_table(index))
    if run_metadata:
        early = run_metadata.get("early_stop") if isinstance(run_metadata.get("early_stop"), dict) else {}
        if early.get("stop_reason") in {"throughput_peak_bracketed", "throughput_plateau"}:
            lines.extend([
                "**Early stop:** perceived throughput peak/plateau bracketed by completed shapes "
                f"more than {early.get('threshold_pct', 2.0)}% below the peak on both available sides.",
                "",
            ])
        elif run_metadata.get("stop_reason"):
            lines.extend([f"**Stop reason:** `{run_metadata.get('stop_reason')}`", ""])
    lines.extend(["</details>", "", "[↑ Back to Navigation](#table-of-contents)", ""])
    return "\n".join(lines)


def render_all_markdown(indices: list[dict[str, Any]]) -> str:
    """Render the accumulated completed optimizer intelligence for all detectors."""
    indices = sorted(indices, key=lambda item: str(item.get("detector_id") or ""))
    detectors = [str(item.get("detector_id") or "unknown") for item in indices]
    lines = [
        "### Execution optimizer summary",
        "",
        "Detector: `all`  ",
        "This report coalesces compatible measurements from completed optimizer runs only.",
        "",
        *_render_optimizer_navigation(detectors=detectors),
        '<a id="preferred-detector-run-configuration"></a>',
        "<details open>",
        "<summary><strong>1. Preferred Detector Run Configuration</strong></summary>",
        "",
        "Compatible completed optimizer runs are coalesced by detector, workload, and concrete runner profile. Repeated shapes retain all observations; the preferred shape is selected canonically by throughput, then lower resource use for throughput-equivalent shapes.",
        "",
        "| Detector | Runner | CPU | Physical | Logical | RAM | Preferred pipelines | Threads / pipeline | Preferred shape range (≤2%) | Search method | Optimization time | Allocated | Sets/s | Shape time | Observations |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for index in indices:
        for row in _render_preferred_configuration(index)[4:]:
            if row.startswith("| "):
                lines.append(row)
    lines.extend([
        "",
        "**Search method legend:** `adaptive` = sparse wide-range search with local refinement around the measured peak and ≤2% preferred-shape boundaries; `powers-of-2` = logarithmic power-of-two pipeline sweep; `exhaustive` = every legal pipeline count in the requested range.",
        "",
        "**Shape-prediction / optimizer coverage**",
        "",
        "| Detector | Observed vCPU anchors | Prediction readiness | Prediction checks | Desired / missing optimization data |",
        "|---|---|---|---|---|",
    ])
    for coverage_index in indices:
        coverage = _shape_prediction_coverage(coverage_index)
        anchors = ", ".join(str(value) for value in coverage["anchors"]) if coverage["anchors"] else "none"
        checks = f"{coverage['verified_predictions']} verified / {coverage['pending_predictions']} pending"
        lines.append(
            f"| {coverage_index.get('detector_id') or 'unknown'} | {anchors} | {coverage['readiness']} | {checks} | {coverage['desired']} |"
        )
    lines.extend([
        "",
        "</details>",
        "",
        "[↑ Back to Navigation](#table-of-contents)",
        "",
    ])

    lines.extend([
        '<a id="detector-run-profile-plot"></a>',
        "<details open>",
        "<summary><strong>2. Detector Run Profile Plot</strong></summary>",
        "",
        "Compatible completed measurements are plotted by detector; thread count is annotated at each measured shape.",
        "",
    ])
    for index in indices:
        detector = str(index.get("detector_id") or "unknown")
        slug = _nav_slug(detector)
        lines.extend([
            f'<a id="detector-run-profile-{slug}"></a>',
            "<details>",
            f"<summary><strong>{detector}</strong></summary>",
            "",
            f"**Search method(s):** `{_represented_search_methods(index)}`",
            "",
            f"![{detector} Detector Run Profile Plot](profiles/{detector}.svg)",
            "",
            "</details>",
            "",
        ])
    lines.extend(["</details>", "", "[↑ Back to Navigation](#table-of-contents)", ""])

    lines.extend([
        '<a id="detector-pipeline-thread-shape-optimization-data"></a>',
        "<details>",
        "<summary><strong>3. Detector Pipeline-Thread Shape Optimization Data</strong></summary>",
        "",
        "Coalesced compatible shape measurements from completed optimizer runs are shown below.",
        "",
    ])
    for index in indices:
        detector = str(index.get("detector_id") or "unknown")
        slug = _nav_slug(detector)
        lines.extend([
            f'<a id="detector-shape-data-{slug}"></a>',
            "<details>",
            f"<summary><strong>{detector}</strong></summary>",
            "",
            f"**Search method(s):** `{_represented_search_methods(index)}`",
            "",
            *_render_shape_table(index),
            "</details>",
            "",
        ])
    lines.extend(["</details>", "", "[↑ Back to Navigation](#table-of-contents)", ""])
    return "\n".join(lines)



def render_heatmap_svg(index: dict[str, Any]) -> str:
    """Render the execution processing profile: pipelines on X, sets/s on Y."""
    runners = [runner for runner in index.get("runners", []) if runner.get("shapes")]
    width, height = 980, 560
    left, top, right, bottom = 92, 112, 40, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    points = [
        (runner, shape)
        for runner in runners
        for shape in runner.get("shapes", [])
        if (_as_int(shape.get("pipelines")) or 0) > 0 and (_as_float(shape.get("parameter_sets_per_second")) or 0) > 0
    ]
    max_pipeline = max((_as_int(shape.get("pipelines")) or 1 for _, shape in points), default=1)
    max_rate = max((_as_float(shape.get("parameter_sets_per_second")) or 0 for _, shape in points), default=1.0)
    min_log = 0.0
    max_log = math.log2(max_pipeline) if max_pipeline > 1 else 1.0

    def x_of(pipeline: int) -> float:
        value = math.log2(max(1, pipeline))
        return left + ((value - min_log) / max(1e-9, max_log - min_log)) * plot_w

    def y_of(rate: float) -> float:
        return top + plot_h - (rate / max(1e-9, max_rate * 1.15)) * plot_h

    palette = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#39c5cf"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;fill:#e6edf3}.muted{fill:#8b949e}.axis{stroke:#484f58;stroke-width:1}.grid{stroke:#21262d;stroke-width:1}.point{stroke:#e6edf3;stroke-width:1.2}</style>',
        f'<text x="24" y="34" font-size="22" font-weight="700">Execution processing profile — {html.escape(str(index.get("detector_id")))}</text>',
        '<text x="24" y="58" font-size="13" class="muted">Pipeline count vs parameter sets/second; thread count is annotated at each measured shape.</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
    ]
    for tick in range(0, 6):
        rate = max_rate * tick / 5
        y = y_of(rate)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="11">{rate:.1f}</text>')
    pipeline_ticks = sorted({1, max_pipeline} | {2**i for i in range(int(max_log) + 1) if 2**i <= max_pipeline})
    for pipeline in pipeline_ticks:
        x = x_of(pipeline)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-size="11">{pipeline}</text>')
    parts.append(f'<text x="{left + plot_w/2}" y="{height - 28}" text-anchor="middle" font-size="13" class="muted">detector pipelines (log₂ scale)</text>')
    parts.append(f'<text x="28" y="{top + plot_h/2}" text-anchor="middle" font-size="13" class="muted" transform="rotate(-90 28 {top + plot_h/2})">parameter sets / second</text>')

    for runner_index, runner in enumerate(runners):
        color = palette[runner_index % len(palette)]
        shapes = sorted(runner.get("shapes", []), key=lambda shape: int(shape.get("pipelines") or 0))
        coords: list[tuple[float, float, dict[str, Any]]] = []
        for shape in shapes:
            p = _as_int(shape.get("pipelines"))
            rate = _as_float(shape.get("parameter_sets_per_second"))
            if not p or not rate or rate <= 0:
                continue
            coords.append((x_of(p), y_of(rate), shape))
        if len(coords) > 1:
            path = " ".join(("M" if i == 0 else "L") + f" {x:.1f} {y:.1f}" for i, (x, y, _) in enumerate(coords))
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        best = runner.get("best_shape") or {}
        for point_index, (x, y, shape) in enumerate(coords):
            is_best = shape.get("execution_shape") == best.get("execution_shape") and shape.get("optimizer_shape_sequence") == best.get("optimizer_shape_sequence")
            radius = 7 if is_best else 5
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" class="point"/>')
            threads = shape.get("threads_per_pipeline") or "?"
            # Keep thread annotations deliberately simple. The earlier collision
            # staggering made dense adaptive neighborhoods harder to read than
            # the original consistent placement. Extra plot headroom remains,
            # so labels can sit just above/right of their measured point.
            label_x, label_y, anchor = x + 8, y - 8, "start"
            parts.append(f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" font-size="10">{threads}t</text>')
        legend_y = 78 + runner_index * 18
        parts.append(f'<rect x="{width - 320}" y="{legend_y - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{width - 300}" y="{legend_y}" font-size="11">{html.escape(str(runner.get("runner_title") or "unknown"))}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _read_jsonl(path: Path | None, optimizer_run_id: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path is None or not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if optimizer_run_id is not None and str(row.get("optimizer_run_id")) != str(optimizer_run_id):
            continue
        rows.append(row)
    return rows


def _preferred_executor_records(detectors: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for detector_id, detector_index in detectors.items():
        if not isinstance(detector_index, dict):
            continue
        for runner in detector_index.get("runners", []):
            if not isinstance(runner, dict):
                continue
            best = runner.get("best_shape") if isinstance(runner.get("best_shape"), dict) else {}
            if not best:
                continue
            records.append({
                "detector_id": detector_id,
                "runner_key": runner.get("runner_key"),
                "compatibility_key": runner.get("compatibility_key"),
                "workload_key": runner.get("workload_key"),
                "runner_label": runner.get("runner_label"),
                "runner_title": runner.get("runner_title"),
                "runner_labels": runner.get("runner_labels"),
                "runner_specs": runner.get("runner_specs") or {},
                "preferred_shape": best,
            })
    records.sort(key=lambda row: (str(row.get("detector_id")), str(row.get("runner_title"))))
    return records


def update_optimizer_artifacts(
    results_root: Path,
    detector_id: str,
    *,
    optimizer_run_id: str | None = None,
    run_metadata_path: Path | None = None,
    runner_metrics_log: Path | None = None,
    observation_log: Path | None = None,
    shard_log: Path | None = None,
) -> dict[str, Path]:
    parallelism_path = readable_index_path(results_root, "parallelism-index.json")
    if not parallelism_path.is_file():
        raise FileNotFoundError(f"Missing {parallelism_path}")
    parallelism = _read_json(parallelism_path)

    # Rehydrate aggregate planning state from durable completed runs before
    # computing preferences.  This makes optimizer-index.json rebuildable and
    # prevents retention/index migrations from collapsing cross-run history.
    durable_records = completed_run_records(results_root, detector_id)
    if durable_records:
        observations = list(parallelism.get("observations", [])) if isinstance(parallelism.get("observations"), list) else []
        by_id = {str(row.get("observation_id")): row for row in observations if isinstance(row, dict) and row.get("observation_id")}
        for record in durable_records:
            for row in record["observations"]:
                key = str(row.get("observation_id") or f"durable:{len(by_id)}")
                by_id[key] = row
        parallelism["observations"] = list(by_id.values())
    historical = build_optimizer_index(parallelism, detector_id)
    current = build_optimizer_index(parallelism, detector_id, optimizer_run_id) if optimizer_run_id is not None else historical

    # A run-local profile must compare like-for-like benchmark workloads.
    # Critical and exhaustive measurements are both valid execution evidence,
    # but their sets/s values are not directly comparable.
    compatible_historical = historical
    if optimizer_run_id is not None:
        current_compatibility_keys = {
            str(runner.get("compatibility_key"))
            for runner in current.get("runners", [])
            if isinstance(runner, dict) and runner.get("compatibility_key")
        }
        compatible_historical = _filter_optimizer_index_to_compatibility(
            historical,
            current_compatibility_keys,
        )

    run_metadata: dict[str, Any] = {}
    if run_metadata_path is not None and run_metadata_path.is_file():
        run_metadata = _read_json(run_metadata_path)
    runner_samples = _read_jsonl(runner_metrics_log, optimizer_run_id)

    index_path = canonical_index_path(results_root, "optimizer-index.json")
    read_index_path = readable_index_path(results_root, "optimizer-index.json")
    if read_index_path.is_file():
        existing = adapt_optimizer_index(_read_json(read_index_path))
    else:
        existing = {"schema_version": OPTIMIZER_INDEX_SCHEMA_VERSION, "detectors": {}, "runs": {}}
    detectors = existing.get("detectors") if isinstance(existing.get("detectors"), dict) else {}
    runs = existing.get("runs") if isinstance(existing.get("runs"), dict) else {}
    for record in durable_records:
        manifest = record["manifest"]
        durable_run_id = str(manifest.get("optimizer_run_id") or "").strip()
        if not durable_run_id:
            continue
        runs.setdefault(durable_run_id, {
            "optimizer_run_id": durable_run_id,
            "detector_id": detector_id,
            "run_metadata": manifest.get("run_metadata") if isinstance(manifest.get("run_metadata"), dict) else {},
        })
    if optimizer_run_id is not None:
        shard_rows = [
            row for row in parallelism.get("shard_observations", [])
            if isinstance(row, dict)
            and str(row.get("optimizer_run_id")) == str(optimizer_run_id)
            and str(row.get("detector_id")) == detector_id
        ]
        runs[str(optimizer_run_id)] = {
            "optimizer_run_id": str(optimizer_run_id),
            "detector_id": detector_id,
            "updated_at_utc": current["updated_at_utc"],
            "shape_observation_count": current.get("observation_count", 0),
            "shard_observation_count": len(shard_rows),
            "run_metadata": run_metadata,
            "runner_metrics_samples": runner_samples,
            "current_execution": current,
        }
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for run_id, run_record in runs.items():
        if not isinstance(run_record, dict):
            continue
        if str(run_record.get("detector_id") or "") != detector_id:
            continue
        metadata = run_record.get("run_metadata")
        if isinstance(metadata, dict):
            metadata_by_id[str(run_id)] = metadata
    historical["run_metadata_by_id"] = metadata_by_id
    current["run_metadata_by_id"] = metadata_by_id

    predictions_path = results_root / "optimizer-predictions.json"
    try:
        from hth.shape_prediction import verify_predictions
        compatible_rows = _comparable(
            (row for row in parallelism.get("observations", []) if isinstance(row, dict)),
            detector_id,
        )
        prediction_payload = verify_predictions(
            predictions_path,
            detector=detector_id,
            workload_rows=compatible_rows,
        )
    except Exception as exc:
        prediction_payload = None
        print(f"Warning: unable to verify saved shape predictions: {exc}")
    if isinstance(prediction_payload, dict):
        prediction_rows = [
            row for row in prediction_payload.get("predictions", [])
            if isinstance(row, dict) and str(row.get("detector_id") or "") == detector_id
        ]
        historical["prediction_history"] = prediction_rows
        current["prediction_history"] = prediction_rows

    detectors[detector_id] = historical

    existing.update({
        "schema_version": OPTIMIZER_INDEX_SCHEMA_VERSION,
        "updated_at_utc": current["updated_at_utc"],
        "detectors": detectors,
        "preferred_executor_configurations": _preferred_executor_records(detectors),
        "runs": runs,
    })
    write_index(results_root, "optimizer-index.json", existing)

    if optimizer_run_id is not None:
        persist_completed_run(
            results_root=results_root, detector=detector_id, run_id=str(optimizer_run_id),
            run_metadata=run_metadata, observation_log=observation_log, shard_log=shard_log,
            runner_metrics_log=runner_metrics_log,
        )

    output_dir = results_root / "execution-optimizer" / detector_id
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "summary.md"
    svg_path = output_dir / "heatmap.svg"
    markdown_path.write_text(render_markdown(current, run_metadata, preferred_index=compatible_historical), encoding="utf-8")
    # The published single-run optimizer summary is run-local below the preferred
    # configuration section, so its plot must visualize the same run-local
    # measurements as Section 3.  The preferred table may legitimately coalesce
    # compatible historical evidence, but plotting that aggregate here can make
    # the graph show an older/different run while the shape table shows the
    # current execution.
    svg_path.write_text(render_heatmap_svg(current), encoding="utf-8")
    return {"index": index_path, "markdown": markdown_path, "heatmap": svg_path}


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--optimizer-run-id")
    parser.add_argument("--run-metadata", type=Path)
    parser.add_argument("--runner-metrics-log", type=Path)
    parser.add_argument("--observation-log", type=Path)
    parser.add_argument("--shard-log", type=Path)
    args = parser.parse_args()
    paths = update_optimizer_artifacts(
        args.results_root,
        args.detector,
        optimizer_run_id=args.optimizer_run_id,
        run_metadata_path=args.run_metadata,
        runner_metrics_log=args.runner_metrics_log,
        observation_log=args.observation_log,
        shard_log=args.shard_log,
    )
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
