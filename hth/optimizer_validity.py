"""Canonical validity policy for persisted execution-optimizer evidence."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

INVALID_REASON_PIPELINE_FANOUT = "single-detector pipeline fan-out bug"

# Static scheduling first introduced the single-detector fan-out regression in
# 6243a667 and 3db939df is the first commit containing the repair.
_BAD_PIPELINE_COMMITS = frozenset({
    "6243a667893df37a4d7ed5721e621b7f43119a77",
    "b74122bb5e0204b3458d218c37a4bce70278bf03",
    "aa5c3237b15134d2ee3744de4f6c452ee3c0e668",
    "c16b1186e333801783784546a04bff209ed2a0e4",
    "a99abf12e835d6cdd5b78d0055c5cea5a45d0128",
    "5e225e6ab09e903f6d5561cb7e5aeb213a9e06c4",
    "dfe6d488b92444818e9d0d32e5b200cbb2c319d2",
})
_BAD_STARTED_UTC = datetime(2026, 8, 27, 19, 56, 49, tzinfo=timezone.utc)
_FIXED_UTC = datetime(2026, 8, 28, 20, 19, 30, tzinfo=timezone.utc)


def optimizer_evidence_is_valid(record: dict[str, Any]) -> bool:
    """Missing validity is backward-compatible; explicit false is a hard ban."""
    return record.get("valid") is not False


def _parse_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _commit(record: dict[str, Any]) -> str:
    candidates = [record.get("pipeline_commit"), record.get("github_sha")]
    for key in ("build", "run_metadata"):
        nested = record.get(key)
        if isinstance(nested, dict):
            candidates.extend((nested.get("pipeline_commit"), nested.get("github_sha")))
    for value in candidates:
        text = str(value or "").strip().lower()
        if text:
            return text
    return ""


def _timestamp(record: dict[str, Any]) -> datetime | None:
    for key in ("observed_at_utc", "captured_at_utc", "created_at_utc", "updated_at_utc"):
        parsed = _parse_utc(record.get(key))
        if parsed is not None:
            return parsed
    metadata = record.get("run_metadata")
    if isinstance(metadata, dict):
        epoch = metadata.get("optimization_started_epoch")
        try:
            return datetime.fromtimestamp(int(epoch), tz=timezone.utc)
        except (TypeError, ValueError, OSError):
            pass
    return None


def affected_by_pipeline_fanout_bug(record: dict[str, Any]) -> bool:
    commit = _commit(record)
    if commit:
        if commit in _BAD_PIPELINE_COMMITS:
            return True
        # A full persisted SHA outside the known bad set is stronger provenance
        # than timestamp fallback. Short/unknown identifiers still fall through.
        if len(commit) >= 40:
            return False
    stamp = _timestamp(record)
    return stamp is not None and _BAD_STARTED_UTC <= stamp < _FIXED_UTC


def migrate_optimizer_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """Return one optimizer record with explicit validity metadata."""
    migrated = dict(record)
    if migrated.get("valid") is False:
        migrated.setdefault("invalid_reason", INVALID_REASON_PIPELINE_FANOUT)
        return migrated
    if affected_by_pipeline_fanout_bug(migrated):
        migrated["valid"] = False
        migrated["invalid_reason"] = INVALID_REASON_PIPELINE_FANOUT
    else:
        migrated.setdefault("valid", True)
        if migrated.get("valid") is True:
            migrated.pop("invalid_reason", None)
    return migrated


def migrate_optimizer_run(
    manifest: dict[str, Any],
    observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply whole-run invalidation when any source evidence is affected."""
    migrated = migrate_optimizer_evidence(manifest)
    rows = observations or []
    if migrated.get("valid") is not False and any(
        migrate_optimizer_evidence(row).get("valid") is False for row in rows if isinstance(row, dict)
    ):
        migrated["valid"] = False
        migrated["invalid_reason"] = INVALID_REASON_PIPELINE_FANOUT
    return migrated
