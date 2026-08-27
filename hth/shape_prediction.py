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

from hth.domain.execution_shape import select_preferred_shape
from hth.persistence import load_index_path, write_index, index_results_root

PREDICTION_SCHEMA_VERSION = "1.0"
PREDICTION_METHOD = "vcpu-linear-shape-scale-v2"


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




def _legacy_prediction_path(path: Path) -> Path | None:
    path = Path(path)
    if path.parent.name != "indexes":
        return None
    legacy = path.parent.parent / path.name
    return legacy if legacy.is_file() else None


def _read_prediction_payload(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.name == "optimizer-predictions.json":
        return load_index_path(path, "optimizer-predictions.json")
    return _read_json(path)


def _write_prediction_payload(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.name == "optimizer-predictions.json":
        write_index(index_results_root(path), "optimizer-predictions.json", payload)
        return
    _write_json(path, payload)


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


def _runner_profile_complete(row: dict[str, Any]) -> bool:
    runner = row.get("runner") if isinstance(row.get("runner"), dict) else {}
    model = " ".join(str(runner.get("cpu_model") or "").lower().split())
    return (
        model not in {"", "unknown", "--"}
        and (_as_int(runner.get("logical_cpu_count")) or 0) > 0
        and (_as_int(runner.get("physical_core_count")) or 0) > 0
    )


def preferred_evidence(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse compatible observations to one preferred shape per concrete runner.

    Host-incomplete legacy rows remain visible as historical evidence, but they
    are explicitly marked so they cannot masquerade as hardware-equivalent
    measurements when richer provenance exists.
    """
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
            "runner_profile_complete": all(_runner_profile_complete(row) for row in group),
        })
    return sorted(evidence, key=lambda row: (int(row["logical_cpus"]), str(row["runner_name"])))


def _aggregate_by_vcpu(evidence: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Collapse runner-local winners into simple vCPU anchors.

    Shape prediction deliberately has one scaling rule: pipeline count scales
    linearly with logical CPU count. Multiple runners at the same vCPU size are
    reduced to the median pipeline fraction and allocation fraction before the
    same scaler is applied.
    """
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in evidence:
        grouped.setdefault(int(item["logical_cpus"]), []).append(item)
    anchors: list[dict[str, float]] = []
    for logical, items in sorted(grouped.items()):
        anchors.append({
            "logical_cpus": float(logical),
            "pipeline_fraction": float(statistics.median(float(item["pipelines"]) / logical for item in items)),
            "allocation_fraction": float(statistics.median(float(item["allocation_fraction"]) for item in items)),
            "runner_count": float(len(items)),
        })
    return anchors


def scale_shape_from_anchor(
    *,
    source_logical_cpus: int,
    source_pipelines: float,
    source_allocation_fraction: float,
    target_logical_cpus: int,
) -> dict[str, int]:
    """Canonical HTH cross-runner shape scaler.

    Pipeline count occupies the same fraction of the target machine as it did on
    the measured machine. For example 32 pipelines on 192 logical CPUs predicts
    round(32 * 32 / 192) == 5 pipelines on 32 logical CPUs. Boundary shapes stay
    boundaries: 1 pipeline remains at least 1 and a max-width shape scales to the
    target max. Once the target pipeline count is selected, threads/pipeline is
    recomputed from the target runner budget (2x target vCPU), matching normal
    execution-shape planning rather than carrying source-runner allocation forward.
    """
    source_logical = max(1, int(source_logical_cpus))
    target_logical = max(1, int(target_logical_cpus))
    target_budget = target_logical * 2
    pipelines = max(1, int(round(float(source_pipelines) * target_logical / source_logical)))
    pipelines = min(pipelines, target_logical)

    # Cross-vCPU intelligence predicts concurrency, not a source-runner thread
    # allocation.  After selecting the target pipeline count, derive the local
    # threads/pipeline exactly as the execution planner does for that runner.
    # ``source_allocation_fraction`` remains in the signature for persisted
    # prediction-schema compatibility but is intentionally not used here.
    _ = source_allocation_fraction
    threads = max(1, target_budget // pipelines)
    allocated = pipelines * threads
    return {
        "pipelines": pipelines,
        "threads_per_pipeline": threads,
        "allocated_threads": allocated,
    }


def _nearest_anchor(anchors: list[dict[str, float]], target_logical_cpus: int) -> dict[str, float]:
    target = float(max(1, target_logical_cpus))
    return min(
        anchors,
        key=lambda anchor: (
            abs(math.log(max(1.0, float(anchor["logical_cpus"])) / target)),
            abs(float(anchor["logical_cpus"]) - target),
            float(anchor["logical_cpus"]),
        ),
    )


def resolve_shape(
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
    """Resolve one detector shape through the single canonical shape path.

    Exact-runner evidence wins. Hardware-equivalent evidence at the same vCPU
    size comes next. If neither exists, any collected detector optimizer evidence
    becomes a simple linearly scaled vCPU anchor. No interpolation curve,
    verified-correction multiplier, or second scaling policy is applied.
    """
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        return None

    evidence = preferred_evidence(rows)
    if not evidence:
        return None

    normalized_target_model = " ".join(str(target_cpu_model or "").lower().split())
    exact = [item for item in evidence if str(item.get("runner_name") or "") == str(target_runner_name or "")]
    if exact:
        chosen = exact[0]
        shape = {
            "pipelines": int(chosen["pipelines"]),
            "threads_per_pipeline": int(chosen["threads_per_pipeline"]),
            "allocated_threads": int(chosen["allocated_threads"]),
        }
        relation = "exact-runner"
        confidence = "high"
        scaled = False
        anchor = {
            "logical_cpus": float(chosen["logical_cpus"]),
            "pipeline_fraction": float(chosen["pipelines"]) / max(1, int(chosen["logical_cpus"])),
            "allocation_fraction": float(chosen["allocation_fraction"]),
        }
    else:
        hardware = [
            item for item in evidence
            if bool(item.get("runner_profile_complete"))
            and normalized_target_model
            and str(item.get("cpu_model") or "") == normalized_target_model
            and int(item.get("logical_cpus") or 0) == int(target_logical_cpus)
            and (
                target_physical_cores is None
                or int(item.get("physical_cores") or 0) == int(target_physical_cores)
            )
        ]
        characterized = [item for item in evidence if bool(item.get("runner_profile_complete"))]
        pool = hardware if hardware else (characterized if characterized else evidence)
        anchors = _aggregate_by_vcpu(pool)
        if not anchors:
            return None
        anchor = _nearest_anchor(anchors, target_logical_cpus)
        source_logical = int(anchor["logical_cpus"])
        source_pipelines = float(anchor["pipeline_fraction"]) * source_logical
        shape = scale_shape_from_anchor(
            source_logical_cpus=source_logical,
            source_pipelines=source_pipelines,
            source_allocation_fraction=float(anchor["allocation_fraction"]),
            target_logical_cpus=target_logical_cpus,
        )
        relation = "hardware-profile" if hardware else "scaled-vcpu"
        confidence = "moderate" if hardware or (characterized and len(anchors) >= 2) else "low"
        scaled = not bool(hardware)

    created = _now()
    identity = (
        f"{detector}|{target_runner_name}|{target_logical_cpus}|"
        f"{shape['pipelines']}|{shape['threads_per_pipeline']}|{created}"
    )
    result = {
        "prediction_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
        "created_at_utc": created,
        "detector_id": detector,
        "method": PREDICTION_METHOD,
        "relation": relation,
        "confidence": confidence,
        "verified_pipeline_correction": 1.0,
        "target_runner": {
            "runner_name": target_runner_name,
            "runner_label": target_runner_label,
            "cpu_model": target_cpu_model,
            "physical_core_count": target_physical_cores,
            "logical_cpu_count": target_logical_cpus,
            "thread_budget": max(1, target_logical_cpus * 2),
        },
        "predicted_shape": shape,
        "evidence_vcpu_anchors": sorted({int(item["logical_cpus"]) for item in evidence}),
        "evidence": evidence,
        "anchor_logical_cpus": int(anchor["logical_cpus"]),
        "status": "pending" if scaled else "measured-equivalent",
    }
    return result


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
    """Backward-compatible prediction API; all math delegates to resolve_shape."""
    result = resolve_shape(
        detector=detector,
        rows=rows,
        target_runner_name=target_runner_name,
        target_runner_label=target_runner_label,
        target_cpu_model=target_cpu_model,
        target_physical_cores=target_physical_cores,
        target_logical_cpus=target_logical_cpus,
        predictions_index=predictions_index,
    )
    if result is None or result.get("relation") in {"exact-runner", "hardware-profile"}:
        return None
    return result


def merge_prediction(predictions_index: Path, prediction: dict[str, Any]) -> dict[str, Any]:
    payload = _read_prediction_payload(predictions_index)
    rows = [row for row in payload.get("predictions", []) if isinstance(row, dict)]
    prediction_id = str(prediction.get("prediction_id") or "")
    if prediction_id and not any(str(row.get("prediction_id") or "") == prediction_id for row in rows):
        rows.append(prediction)
    payload.update({
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "updated_at_utc": _now(),
        "predictions": rows[-2000:],
    })
    _write_prediction_payload(predictions_index, payload)
    return payload


def verify_predictions(
    predictions_index: Path,
    *,
    detector: str,
    workload_rows: Iterable[dict[str, Any]],
) -> dict[str, Any] | None:
    if not predictions_index.is_file() and _legacy_prediction_path(predictions_index) is None:
        return None
    payload = _read_prediction_payload(predictions_index)
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
        _write_prediction_payload(predictions_index, payload)
    return payload
