"""Explicit regression run-mode and evidence-tier semantics."""
from __future__ import annotations

from typing import Any, Mapping


RUN_MODES = frozenset({"smoke", "full"})
EVIDENCE_TIERS = frozenset({"provisional", "partial", "authoritative"})


def evidence_tier_for(run_mode: str, *, exhaustive_complete: bool) -> str:
    """Return the canonical tier once, at execution materialization time."""
    mode = str(run_mode).strip().lower()
    if mode not in RUN_MODES:
        raise ValueError(f"Unsupported regression run mode: {run_mode!r}")
    if mode == "smoke":
        return "provisional"
    return "authoritative" if exhaustive_complete else "partial"


def validate_run_semantics(run_mode: Any, evidence_tier: Any) -> tuple[str, str]:
    mode = str(run_mode or "").strip().lower()
    tier = str(evidence_tier or "").strip().lower()
    if mode not in RUN_MODES:
        raise ValueError(f"Unsupported regression run mode: {run_mode!r}")
    if tier not in EVIDENCE_TIERS:
        raise ValueError(f"Unsupported regression evidence tier: {evidence_tier!r}")
    if mode == "smoke" and tier != "provisional":
        raise ValueError(f"Smoke runs must be provisional, not {tier!r}")
    if mode == "full" and tier == "provisional":
        raise ValueError("Full runs cannot carry provisional evidence")
    return mode, tier


def legacy_run_semantics(
    *payloads: Mapping[str, Any],
    fallback_mode: str | None = None,
) -> tuple[str, str]:
    """Adapt pre-contract artifacts at the compatibility boundary only."""
    explicit_mode = next(
        (payload.get("run_mode") for payload in payloads if payload.get("run_mode")),
        None,
    )
    explicit_tier = next(
        (payload.get("evidence_tier") for payload in payloads if payload.get("evidence_tier")),
        None,
    )
    if explicit_mode and explicit_tier:
        return validate_run_semantics(explicit_mode, explicit_tier)
    tier = str(explicit_tier or "") or next(
        (
            str(payload.get("calibration_status") or "").strip().lower()
            for payload in payloads
            if payload.get("calibration_status")
        ),
        "",
    )
    mode = str(explicit_mode or fallback_mode or "").strip().lower()
    if not mode:
        mode = "smoke" if tier == "provisional" else "full"
    if not tier:
        exhaustive_complete = any(
            bool((payload.get("search") or {}).get("exhaustive_complete"))
            for payload in payloads
            if isinstance(payload.get("search"), dict)
        )
        tier = evidence_tier_for(mode, exhaustive_complete=exhaustive_complete)
    return validate_run_semantics(mode, tier)
