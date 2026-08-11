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


def authoritative_record(records: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Select detector evidence provenance-first and never quality-first."""
    candidates = [row for row in records if isinstance(row, dict)]
    if not candidates:
        return None
    authoritative = [
        row for row in candidates
        if calibration_status(row) == "authoritative"
        and calibration_search_type(row) in {"exhaustive", "cartesian"}
    ]
    return max(authoritative or candidates, key=_timestamp)
