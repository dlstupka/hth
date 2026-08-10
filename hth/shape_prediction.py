#!/usr/bin/env python3
"""Predict detector execution shapes from persisted optimizer evidence and track prediction quality."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hth.optimizer_store import select_preferred_shape

PREDICTION_SCHEMA_VERSION = "1.0"
PREDICTION_METHOD = "vcpu-shape-interpolation-v1"


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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _shape_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipelines": _as_int(row.get("active_pipelines")),
        "threads_per_pipeline": _as_int(row.get("threads_per_pipeline")),
        "allocated_threads": _as_int(row.get("allocated_threads")),
        "parameter_sets_per_second": _as_float(row.get("parameter_sets_per_second")),
        "fastest_wall_clock_seconds": _as_float(row.get("wall_clock_seconds")),
        "optimizer_shape_sequence": _as_int(row.get("optimizer_shape_sequence")),
        "optimizer_run_id": row.get("optimizer_run_id"),
    }


def _runner_key(row: dict[str, Any]) -> tuple[str, str, int, int | None]:
    runner = row.get("runner") if isinstance(row.get("runner"), dict) else {}
    return (
        str(runner.get("runner_name") or "unknown"),
        " ".join(str(runner.get("cpu_model") or "").lower().split()),
        _as_int(runner.get("logical_cpu_count")) or 1,
        _as_int(runner.get("physical_core_count")),
    )


def preferred_evidence(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse compatible observations to one preferred shape per concrete runner."""
    groups: dict[tuple[str, str, int, int | None], list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        groups.setdefault(_runner_key(row), []).append(row)

    evidence: list[dict[str, Any]] = []
    for key, group in groups.items():
        best = select_preferred_shape(_shape_from_row(row) for row in group)
        if not best:
            continue
        name, model, logical, physical = key
        pipelines = _as_int(best.get("pipelines"))
        threads = _as_int(best.get("threads_per_pipeline"))
        allocated = _as_int(best.get("allocated_threads"))
        if not pipelines or not threads:
            continue
        budget = max(1, logical * 2)
        evidence.append({
            "runner_name": name,
            "cpu_model": model,
            "logical_cpus": logical,
            "physical_cores": physical,
            "pipelines": pipelines,
            "threads_per_pipeline": threads,
            "allocated_threads": allocated or pipelines * threads,
            "allocation_fraction": min(1.0, (allocated or pipelines * threads) / budget),
        })
    return sorted(evidence, key=lambda row: (int(row["logical_cpus"]), str(row["runner_name"])))


def _aggregate_by_vcpu(evidence: list[dict[str, Any]]) -> list[dict[str, float]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in evidence:
        grouped.setdefault(int(item["logical_cpus"]), []).append(item)
    anchors: list[dict[str, float]] = []
    for logical, items in sorted(grouped.items()):
        anchors.append({
            "logical_cpus": float(logical),
            "pipelines": float(statistics.median(float(item["pipelines"]) for item in items)),
            "allocation_fraction": float(statistics.median(float(item["allocation_fraction"]) for item in items)),
        })
    return anchors


def _power_interpolate(x: float, a: dict[str, float], b: dict[str, float], key: str) -> float:
    """Interpolate/extrapolate in log space; clamp scaling exponent to [0, 1]."""
    x1, x2 = a["logical_cpus"], b["logical_cpus"]
    y1, y2 = max(1e-9, a[key]), max(1e-9, b[key])
    if x1 == x2:
        return y1
    exponent = math.log(y2 / y1) / math.log(x2 / x1)
    exponent = min(1.0, max(0.0, exponent))
    if x <= x1:
        return y1 * ((x / x1) ** exponent)
    if x >= x2:
        return y2 * ((x / x2) ** exponent)
    fraction = math.log(x / x1) / math.log(x2 / x1)
    return math.exp(math.log(y1) + fraction * (math.log(y2) - math.log(y1)))


def _linear_interpolate(x: float, a: dict[str, float], b: dict[str, float], key: str) -> float:
    x1, x2 = a["logical_cpus"], b["logical_cpus"]
    if x1 == x2:
        return a[key]
    fraction = (x - x1) / (x2 - x1)
    return a[key] + fraction * (b[key] - a[key])


def _verified_correction(predictions_index: Path | None, detector: str) -> float:
    if predictions_index is None or not predictions_index.is_file():
        return 1.0
    payload = _read_json(predictions_index)
    ratios: list[float] = []
    for row in payload.get("predictions", []):
        if not isinstance(row, dict) or str(row.get("detector_id") or "") != detector:
            continue
        verification = row.get("verification")
        predicted = row.get("predicted_shape")
        if not isinstance(verification, dict) or not isinstance(predicted, dict):
            continue
        actual = verification.get("actual_shape")
        if not isinstance(actual, dict):
            continue
        pp = _as_int(predicted.get("pipelines"))
        ap = _as_int(actual.get("pipelines"))
        if pp and ap:
            ratios.append(ap / pp)
    if not ratios:
        return 1.0
    return min(1.25, max(0.75, float(statistics.median(ratios))))


def predict_shape(
    *,
    detector: str,
    rows: Iterable[dict[str, Any]],
    target_runner_name: str,
    target_runner_label: str,
    target_cpu_model: str,
    target_physical_cores: int | None,
    target_logical_cpus: int,
    predictions_index: Path | None = None,
) -> dict[str, Any] | None:
    evidence = preferred_evidence(rows)
    anchors = _aggregate_by_vcpu(evidence)
    if not anchors:
        return None

    target = float(max(1, target_logical_cpus))
    if len(anchors) == 1:
        anchor = anchors[0]
        pipeline_estimate = anchor["pipelines"] * target / anchor["logical_cpus"]
        allocation_fraction = anchor["allocation_fraction"]
        relation = "single-anchor-linear-scale"
    else:
        lower = max((a for a in anchors if a["logical_cpus"] <= target), key=lambda a: a["logical_cpus"], default=None)
        upper = min((a for a in anchors if a["logical_cpus"] >= target), key=lambda a: a["logical_cpus"], default=None)
        if lower is not None and upper is not None and lower is not upper:
            pipeline_estimate = _power_interpolate(target, lower, upper, "pipelines")
            allocation_fraction = _linear_interpolate(target, lower, upper, "allocation_fraction")
            relation = "interpolated"
        elif lower is None:
            pipeline_estimate = _power_interpolate(target, anchors[0], anchors[1], "pipelines")
            allocation_fraction = anchors[0]["allocation_fraction"]
            relation = "extrapolated-below"
        elif upper is None:
            pipeline_estimate = _power_interpolate(target, anchors[-2], anchors[-1], "pipelines")
            allocation_fraction = anchors[-1]["allocation_fraction"]
            relation = "extrapolated-above"
        else:
            pipeline_estimate = lower["pipelines"]
            allocation_fraction = lower["allocation_fraction"]
            relation = "same-vcpu-anchor"

    correction = _verified_correction(predictions_index, detector)
    pipeline_estimate *= correction
    pipelines = max(1, int(round(pipeline_estimate)))
    runner_budget = max(1, target_logical_cpus * 2)
    target_allocated = max(1, min(runner_budget, int(round(runner_budget * min(1.0, max(0.05, allocation_fraction))))))
    threads = max(1, int(round(target_allocated / pipelines)))
    if pipelines * threads > runner_budget:
        threads = max(1, runner_budget // pipelines)
    pipelines = min(pipelines, runner_budget)
    allocated = pipelines * threads

    unique_vcpus = len(anchors)
    inside_span = anchors[0]["logical_cpus"] <= target <= anchors[-1]["logical_cpus"]
    if unique_vcpus >= 3 and inside_span:
        confidence = "high"
    elif unique_vcpus >= 2:
        confidence = "moderate"
    else:
        confidence = "low"

    created = _now()
    identity = f"{detector}|{target_runner_name}|{target_logical_cpus}|{pipelines}|{threads}|{created}"
    prediction_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return {
        "prediction_id": prediction_id,
        "created_at_utc": created,
        "detector_id": detector,
        "method": PREDICTION_METHOD,
        "relation": relation,
        "confidence": confidence,
        "verified_pipeline_correction": round(correction, 6),
        "target_runner": {
            "runner_name": target_runner_name,
            "runner_label": target_runner_label,
            "cpu_model": target_cpu_model,
            "physical_core_count": target_physical_cores,
            "logical_cpu_count": target_logical_cpus,
            "thread_budget": runner_budget,
        },
        "predicted_shape": {
            "pipelines": pipelines,
            "threads_per_pipeline": threads,
            "allocated_threads": allocated,
        },
        "evidence_vcpu_anchors": [int(anchor["logical_cpus"]) for anchor in anchors],
        "evidence": evidence,
        "status": "pending",
    }


def merge_prediction(predictions_index: Path, prediction: dict[str, Any]) -> dict[str, Any]:
    payload = _read_json(predictions_index)
    rows = [row for row in payload.get("predictions", []) if isinstance(row, dict)]
    prediction_id = str(prediction.get("prediction_id") or "")
    if prediction_id and not any(str(row.get("prediction_id") or "") == prediction_id for row in rows):
        rows.append(prediction)
    payload.update({
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "updated_at_utc": _now(),
        "predictions": rows[-2000:],
    })
    _write_json(predictions_index, payload)
    return payload


def verify_predictions(
    predictions_index: Path,
    *,
    detector: str,
    workload_rows: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    if not predictions_index.is_file():
        return None
    payload = _read_json(predictions_index)
    rows = [row for row in payload.get("predictions", []) if isinstance(row, dict)]
    candidates = [row for row in workload_rows if isinstance(row, dict)]
    changed = False

    for prediction in rows:
        if str(prediction.get("detector_id") or "") != detector or str(prediction.get("status") or "pending") == "verified":
            continue
        target = prediction.get("target_runner") if isinstance(prediction.get("target_runner"), dict) else {}
        workload = prediction.get("workload") if isinstance(prediction.get("workload"), dict) else {}
        target_name = str(target.get("runner_name") or "")
        target_logical = _as_int(target.get("logical_cpu_count"))
        compatible = [
            row for row in candidates
            if (not workload.get("detector_config_sha256") or str(row.get("detector_config_sha256") or "") == str(workload.get("detector_config_sha256")))
            and (not workload.get("golden_set_sha256") or str(row.get("golden_set_sha256") or "") == str(workload.get("golden_set_sha256")))
            and (not workload.get("max_dimension") or _as_int(row.get("max_dimension")) == _as_int(workload.get("max_dimension")))
        ]
        exact = [
            row for row in compatible
            if str((row.get("runner") or {}).get("runner_name") or "") == target_name
        ]
        if not exact and target_logical:
            exact = [
                row for row in compatible
                if _as_int((row.get("runner") or {}).get("logical_cpu_count")) == target_logical
            ]
        if not exact:
            continue
        best = select_preferred_shape(_shape_from_row(row) for row in exact)
        if not best:
            continue
        actual_p = _as_int(best.get("pipelines"))
        actual_t = _as_int(best.get("threads_per_pipeline"))
        predicted = prediction.get("predicted_shape") if isinstance(prediction.get("predicted_shape"), dict) else {}
        predicted_p = _as_int(predicted.get("pipelines"))
        predicted_t = _as_int(predicted.get("threads_per_pipeline"))
        if not actual_p or not actual_t or not predicted_p or not predicted_t:
            continue
        prediction["status"] = "verified"
        prediction["verification"] = {
            "verified_at_utc": _now(),
            "actual_shape": {
                "pipelines": actual_p,
                "threads_per_pipeline": actual_t,
                "allocated_threads": _as_int(best.get("allocated_threads")) or actual_p * actual_t,
            },
            "pipeline_error_pct": round(((predicted_p / actual_p) - 1.0) * 100.0, 3),
            "threads_error_pct": round(((predicted_t / actual_t) - 1.0) * 100.0, 3),
            "exact_shape_match": predicted_p == actual_p and predicted_t == actual_t,
        }
        changed = True

    if changed:
        payload["updated_at_utc"] = _now()
        payload["predictions"] = rows
        _write_json(predictions_index, payload)
    return payload
