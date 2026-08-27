"""Shared interpretation of persisted execution-optimizer intelligence.

This module is deliberately library-like: persistence, plotting, and optimizer
execution remain elsewhere.  Consumers ask this layer for compatible evidence
and one normalized measured-or-predicted execution shape so dispatch,
regression execution, and future reporting code do not reimplement optimizer
compatibility or cross-vCPU scaling policy.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from hth.domain.execution_shape import (
    DETERMINISTIC_OPTIMIZER_STRATEGIES,
    optimizer_row_matches_workload,
    select_preferred_shape,
)
from hth.shape_prediction import resolve_shape


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runner_from_row(row: dict[str, Any]) -> dict[str, Any]:
    runner = row.get("runner")
    return runner if isinstance(runner, dict) else {}


def runner_labels_from_row(row: dict[str, Any]) -> list[str]:
    """Return the concrete persisted runner label set when available."""
    runner = runner_from_row(row)
    labels = runner.get("runner_labels")
    if isinstance(labels, list):
        clean = [str(value).strip() for value in labels if str(value).strip()]
        if clean:
            return clean
    label = str(runner.get("runner_label") or "").strip()
    if label and label not in {"unknown", "github-hosted"}:
        return ["self-hosted", label]
    return ["ubuntu-latest"]


def row_matches_required_labels(row: dict[str, Any], required_labels: Iterable[str]) -> bool:
    """GitHub self-hosted selectors are subsets of a runner's full labels."""
    observed = set(runner_labels_from_row(row))
    required = {str(value).strip() for value in required_labels if str(value).strip()}
    return bool(required) and required.issubset(observed)


def logical_cpus_from_capacity_label(label: str | None) -> int | None:
    """Interpret HTH capacity labels such as ``192t`` as 192 logical CPUs."""
    match = re.fullmatch(r"(\d+)t", str(label or "").strip().lower())
    if not match:
        return None
    return max(1, int(match.group(1)))


def shape_from_row(row: dict[str, Any]) -> dict[str, Any]:
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



def legacy_published_optimizer_index(path: Path, detector: str) -> dict[str, Any] | None:
    """Recover completed optimizer evidence from a pre-index published summary."""
    if not path.is_file():
        return None

    def key(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def seconds(text: str) -> float:
        total = 0.0
        for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", text):
            total += float(value) * {"h": 3600.0, "m": 60.0, "s": 1.0}[unit]
        return total

    header: dict[str, int] | None = None
    groups: dict[str, list[dict[str, Any]]] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.startswith("|") or raw.startswith("|---"):
            continue
        cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
        normalized = [key(cell.replace("**", "")) for cell in cells]
        if "runner" in normalized and "pipelines" in normalized:
            header = {name: idx for idx, name in enumerate(normalized)}
            continue
        if header is None:
            continue

        def field(*names: str) -> str:
            for name in names:
                idx = header.get(key(name))
                if idx is not None and idx < len(cells):
                    return cells[idx].replace("**", "").strip()
            return ""

        runner = field("runner")
        if not runner or runner.lower().startswith("unknown"):
            continue
        try:
            pipelines = int(field("pipelines"))
            shards = int(field("shards") or pipelines)
            threads = int(field("threads / pipeline", "threads per pipeline"))
            allocated = int(field("allocated threads", "allocated"))
            rate = float(field("sets/s", "parameter sets / second"))
        except (TypeError, ValueError):
            continue
        wall = seconds(field("fastest wall", "wall", "shape time"))
        if wall <= 0.0 or rate <= 0.0:
            continue
        speedup = None
        try:
            speedup = float(field("speedup vs 1 pipeline", "speedup").rstrip("×x"))
        except ValueError:
            pass
        groups.setdefault(runner, []).append({
            "pipelines": pipelines, "shards": shards,
            "threads_per_pipeline": threads, "allocated_threads": allocated,
            "fastest_wall_clock_seconds": wall, "parameter_sets_per_second": rate,
            "observed_speedup_vs_one_pipeline": speedup,
            "execution_shape": f"{pipelines}p/{shards}s/{threads}t",
            "optimizer_shape_sequence": pipelines,
        })
    if not groups:
        return None
    runner_title, shapes = max(groups.items(), key=lambda item: (len(item[1]), max((x["pipelines"] for x in item[1]), default=0)))
    shapes.sort(key=lambda shape: shape["pipelines"])
    best = select_preferred_shape(shapes)
    return {
        "schema_version": 1, "detector_id": detector, "optimizer_run_id": "legacy-published",
        "runner_count": 1, "observation_count": len(shapes), "best_across_runners": best,
        "runners": [{"runner_title": runner_title, "shapes": shapes, "best_shape": best}],
    }


def _legacy_summary_optimizer_rows(
    *, parallelism_index: Path, detector: str, golden_sha256: str, max_dimension: int,
) -> list[dict[str, Any]]:
    """Adapt the shared legacy-summary recovery result into intelligence rows."""
    summary = parallelism_index.parent.parent / "execution-optimizer" / detector / "summary.md"
    recovered = legacy_published_optimizer_index(summary, detector)
    if not recovered:
        return []
    rows: list[dict[str, Any]] = []
    for runner_group in recovered.get("runners", []):
        runner_title = str(runner_group.get("runner_title") or "")
        capacity = re.match(r"\s*(\d+)t\b", runner_title.lower())
        logical = int(capacity.group(1)) if capacity else None
        name_match = re.search(r"—\s*([^()]+?)(?:\s*\(|$)", runner_title)
        runner_name = name_match.group(1).strip() if name_match else "legacy-published"
        runner_label = f"{logical}t" if logical else "unknown"
        for shape in runner_group.get("shapes", []):
            pipelines = _as_int(shape.get("pipelines")); threads = _as_int(shape.get("threads_per_pipeline"))
            if not pipelines or not threads:
                continue
            rows.append({
                "source": "execution-optimizer", "detector_id": detector, "mode": "full",
                "strategy": "exhaustive", "golden_set_sha256": golden_sha256,
                "possible_parameter_sets": 1, "actual_parameter_sets": 1, "max_dimension": max_dimension,
                "wall_clock_seconds": shape.get("fastest_wall_clock_seconds"),
                "parameter_sets_per_second": shape.get("parameter_sets_per_second"),
                "active_pipelines": pipelines, "shards": _as_int(shape.get("shards")) or pipelines,
                "threads_per_pipeline": threads,
                "allocated_threads": _as_int(shape.get("allocated_threads")) or pipelines * threads,
                "execution_shape": shape.get("execution_shape"), "optimizer_run_id": "legacy-published",
                "runner": {"runner_label": runner_label,
                           "runner_labels": (["self-hosted", runner_label] if logical else ["self-hosted"]),
                           "runner_name": runner_name, "logical_cpu_count": logical},
                "optimizer_intelligence_recovery": "published-summary",
            })
    return rows


def compatible_optimizer_rows(
    *,
    parallelism_index: Path,
    detector_config: Path,
    golden_set: Path,
    max_dimension: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Load optimizer rows compatible with this detector workload.

    Detector configs may explicitly declare implementation-level shape
    compatibility so calibration-grid edits do not invalidate execution-shape
    evidence.  This is the single workload-compatibility path for consumers.
    """
    detector_config_payload = _read_json(detector_config)
    detector = str(detector_config_payload.get("detector") or detector_config.stem)
    detector_sha256 = _sha256(detector_config)
    golden_sha256 = _sha256(golden_set)
    index = _read_json(parallelism_index)
    observations = [row for row in index.get("observations", []) if isinstance(row, dict)]

    def matches(row: dict[str, Any], detector_sha: str) -> bool:
        return optimizer_row_matches_workload(
            row,
            detector=detector,
            detector_sha256=detector_sha,
            golden_sha256=golden_sha256,
            max_dimension=max_dimension,
        )

    rows = [row for row in observations if matches(row, detector_sha256)]
    if rows:
        return detector, rows

    compatibility = str(detector_config_payload.get("optimizer_shape_compatibility") or "").strip().lower()
    if compatibility == "detector-implementation":
        rows = [
            row for row in observations
            if row.get("source") == "execution-optimizer"
            and str(row.get("detector_id") or "") == detector
            and str(row.get("mode") or "") == "full"
            and str(row.get("strategy") or "") in DETERMINISTIC_OPTIMIZER_STRATEGIES
            and str(row.get("golden_set_sha256") or "") == golden_sha256
            and (_as_int(row.get("max_dimension")) in (None, max_dimension))
            and matches(row, str(row.get("detector_config_sha256") or ""))
        ]
    if not rows:
        rows = _legacy_summary_optimizer_rows(
            parallelism_index=parallelism_index, detector=detector,
            golden_sha256=golden_sha256, max_dimension=max_dimension,
        )
    return detector, rows


def resolve_optimizer_intelligence(
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
    """Resolve measured evidence first, then simple linear vCPU prediction.

    Cross-vCPU scaling intentionally uses the current HTH rule:
    ``new_pipelines = round(old_pipelines / old_vcpu * new_vcpu)`` with
    ``max pipelines = new_vcpu`` and ``max thread budget = new_vcpu * 2``.
    After pipeline scaling, threads/pipeline is recomputed from the target runner
    budget, so the destination machine owns the local thread allocation.
    """
    compatible = [row for row in rows if isinstance(row, dict)]
    if not compatible:
        return None
    resolved = resolve_shape(
        detector=detector,
        rows=compatible,
        target_runner_name=target_runner_name,
        target_runner_label=target_runner_label,
        target_cpu_model=target_cpu_model,
        target_physical_cores=target_physical_cores,
        target_logical_cpus=max(1, int(target_logical_cpus)),
        predictions_index=predictions_index,
    )
    if not resolved:
        return None
    relation = str(resolved.get("relation") or "unknown")
    resolved["provenance"] = "predicted" if relation == "scaled-vcpu" else "measured"
    resolved["matched_observations"] = len(compatible)
    return resolved


def resolve_selector_intelligence(
    *,
    detector: str,
    rows: Iterable[dict[str, Any]],
    required_labels: Iterable[str],
    target_runner_label: str,
    target_logical_cpus: int | None,
) -> dict[str, Any] | None:
    """Resolve intelligence before a concrete GitHub runner has been assigned.

    Measured evidence from the requested label selector wins.  If that does not
    exist and the target's vCPU capacity is known, fall back to the same linear
    cross-vCPU predictor used after job start.
    """
    compatible = [row for row in rows if isinstance(row, dict)]
    measured_rows = [row for row in compatible if row_matches_required_labels(row, required_labels)]
    if measured_rows:
        best = select_preferred_shape(shape_from_row(row) for row in measured_rows)
        if not best:
            return None
        pipelines = int(best["pipelines"])
        threads = int(best["threads_per_pipeline"])
        rate = _as_float(best.get("parameter_sets_per_second"))
        matching = [
            row for row in measured_rows
            if _as_int(row.get("active_pipelines")) == pipelines
            and _as_int(row.get("threads_per_pipeline")) == threads
            and (rate is None or _as_float(row.get("parameter_sets_per_second")) == rate)
        ]
        row = matching[0] if matching else measured_rows[0]
        return {
            "detector_id": detector,
            "relation": "requested-runner",
            "confidence": "high",
            "provenance": "measured",
            "matched_observations": len(measured_rows),
            "predicted_shape": {
                "pipelines": pipelines,
                "threads_per_pipeline": threads,
                "allocated_threads": int(best.get("allocated_threads") or pipelines * threads),
            },
            "evidence_row": row,
        }

    if not target_logical_cpus:
        return None
    return resolve_optimizer_intelligence(
        detector=detector,
        rows=compatible,
        target_runner_name="requested",
        target_runner_label=target_runner_label,
        target_cpu_model="",
        target_physical_cores=None,
        target_logical_cpus=target_logical_cpus,
    )


def resolve_optimizer_start_hint(
    *,
    parallelism_index: Path,
    detector_config: Path,
    golden_set: Path,
    max_dimension: int,
    target_runner_name: str,
    target_runner_label: str,
    target_cpu_model: str,
    target_physical_cores: int | None,
    target_logical_cpus: int,
) -> dict[str, Any] | None:
    """Return the best shared-intelligence seed for an adaptive optimizer run.

    The hint is deliberately only a starting point.  Adaptive search remains
    responsible for exploring the full legal pipeline range and proving the
    peak/plateau with measured evidence.
    """
    detector, rows = compatible_optimizer_rows(
        parallelism_index=parallelism_index,
        detector_config=detector_config,
        golden_set=golden_set,
        max_dimension=max_dimension,
    )
    if not rows:
        return None
    result = resolve_optimizer_intelligence(
        detector=detector,
        rows=rows,
        target_runner_name=target_runner_name,
        target_runner_label=target_runner_label,
        target_cpu_model=target_cpu_model,
        target_physical_cores=target_physical_cores,
        target_logical_cpus=target_logical_cpus,
    )
    if not result:
        return None
    shape = result.get("predicted_shape") if isinstance(result.get("predicted_shape"), dict) else {}
    pipelines = _as_int(shape.get("pipelines"))
    threads = _as_int(shape.get("threads_per_pipeline"))
    if not pipelines or not threads:
        return None
    return {
        "detector_id": detector,
        "pipelines": pipelines,
        "threads_per_pipeline": threads,
        "relation": str(result.get("relation") or "unknown"),
        "provenance": str(result.get("provenance") or "unknown"),
        "confidence": str(result.get("confidence") or "unknown"),
    }
