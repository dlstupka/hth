from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable


def _timestamp(record: dict[str, Any]) -> tuple[float, int]:
    raw = record.get("created_at_utc") or record.get("date") or ""
    epoch = 0.0
    if raw:
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            epoch = dt.timestamp()
        except (TypeError, ValueError):
            pass
    build_value = record.get("build_number")
    if build_value is None and isinstance(record.get("build"), dict):
        build_value = record["build"].get("github_run_number")
    try:
        build = int(build_value or 0)
    except (TypeError, ValueError):
        build = 0
    return epoch, build


def calibration_status(record: dict[str, Any]) -> str:
    return str(record.get("status") or record.get("calibration_status") or "").lower()


def calibration_search_type(record: dict[str, Any]) -> str:
    direct = record.get("search_type")
    if direct:
        return str(direct).lower()
    search = record.get("search")
    search = search if isinstance(search, dict) else {}
    if search.get("exhaustive_complete"):
        return "exhaustive"
    return str(search.get("strategy") or "").lower()


def _float_value(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quality(record: dict[str, Any]) -> tuple[float, float, float, float, tuple[float, int]]:
    """Rank compatible authoritative calibrations by detector quality.

    Avg IoU is the primary detector-ranking metric.  Minimum IoU, failure count,
    and StdDev are deterministic tie-breakers; recency only breaks otherwise
    equivalent calibration evidence.  Missing metrics sort below known metrics.
    """
    selection = record.get("selection") if isinstance(record.get("selection"), dict) else {}
    mean_iou = _float_value(record.get("mean_iou"))
    if mean_iou is None:
        mean_iou = _float_value(selection.get("best_avg_iou"))
    minimum_iou = _float_value(record.get("minimum_iou"))
    if minimum_iou is None:
        minimum_iou = _float_value(selection.get("minimum_iou"))
    failures = _float_value(record.get("failures"))
    if failures is None:
        failures = _float_value(selection.get("failure_count"))
    stddev = _float_value(record.get("stddev_iou"))
    if stddev is None:
        stddev = _float_value(selection.get("stddev_iou"))
    return (
        mean_iou if mean_iou is not None else float("-inf"),
        minimum_iou if minimum_iou is not None else float("-inf"),
        -(failures if failures is not None else float("inf")),
        -(stddev if stddev is not None else float("inf")),
        _timestamp(record),
    )


def authoritative_record(records: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the best authoritative calibration without letting smoke usurp it.

    Provenance remains the first gate: a complete authoritative exhaustive/full
    calibration always outranks partial or smoke evidence.  Within that
    authoritative population, however, "best known" means best measured detector
    quality rather than newest build.  This prevents a later exhaustive rerun that
    merely beats its factory baseline from replacing a stronger compatible incumbent.

    When no authoritative full calibration exists, preserve the historical fallback
    behavior and use the newest available evidence.
    """
    candidates = [row for row in records if isinstance(row, dict)]
    if not candidates:
        return None
    authoritative = [
        row for row in candidates
        if calibration_status(row) == "authoritative"
        and calibration_search_type(row) in {"exhaustive", "cartesian"}
    ]
    if authoritative:
        return max(authoritative, key=_quality)
    return max(candidates, key=_timestamp)
