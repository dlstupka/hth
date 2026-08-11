from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable


def _timestamp(record: dict[str, Any]) -> tuple[float, int]:
    """Stable newest-first timestamp/build ordering without using quality metrics."""
    raw = record.get("created_at_utc") or record.get("date") or ""
    epoch = 0.0
    if raw:
        try:
            text = str(raw).replace("Z", "+00:00")
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            epoch = dt.timestamp()
        except (TypeError, ValueError):
            epoch = 0.0
    try:
        build = int(record.get("build_number") or 0)
    except (TypeError, ValueError):
        build = 0
    return epoch, build


def authoritative_record(records: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the record that authoritatively represents one detector.

    Selection is provenance-first, never quality-first:
      1. newest compatible full/exhaustive authoritative calibration;
      2. otherwise newest compatible available observation (smoke/partial).

    Avg IoU, failures, or any other quality metric never participate in
    selecting which historical record represents the detector.
    """
    candidates = [record for record in records if isinstance(record, dict)]
    if not candidates:
        return None

    authoritative = [
        record for record in candidates
        if str(record.get("status") or "").lower() == "authoritative"
        and str(record.get("search_type") or "").lower() == "exhaustive"
    ]
    pool = authoritative if authoritative else candidates
    return max(pool, key=_timestamp)
