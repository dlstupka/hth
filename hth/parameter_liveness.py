"""Configuration-level parameter-liveness audit utilities.

This audit is deliberately conservative. It validates declared zombie metadata and
reports already-pinned dimensions, but it never infers that a parameter is dead
merely because it is fixed or absent from a current refinement grid.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def audit_detector_config(config: dict[str, Any]) -> dict[str, Any]:
    detector = str(config.get("detector") or "unknown")
    parameters = config.get("parameters", {}) if isinstance(config.get("parameters"), dict) else {}
    profiles = config.get("profiles", {}) if isinstance(config.get("profiles"), dict) else {}
    baseline = profiles.get("baseline", {}) if isinstance(profiles.get("baseline"), dict) else {}
    zombies = config.get("zombie_parameters", {}) if isinstance(config.get("zombie_parameters"), dict) else {}
    errors: list[str] = []
    for name, raw in zombies.items():
        if name in parameters:
            errors.append(f"{name}: zombie parameter must not also be in parameters")
            continue
        if not isinstance(raw, dict):
            errors.append(f"{name}: zombie specification must be an object")
            continue
        values = raw.get("values")
        if not isinstance(values, list) or not values:
            errors.append(f"{name}: zombie parameter must retain a non-empty values domain")
            continue
        if "pinned_value" not in raw:
            errors.append(f"{name}: zombie parameter must declare pinned_value")
        elif raw.get("pinned_value") not in values:
            errors.append(f"{name}: pinned_value must be one of the retained values")
        if not str(raw.get("reason") or "").strip():
            errors.append(f"{name}: zombie parameter must record audit reason")
        if not str(raw.get("audit_scope") or "").strip():
            errors.append(f"{name}: zombie parameter must record audit_scope")
        if name not in baseline:
            errors.append(f"{name}: baseline must retain the parameter for reproducibility")

    singleton = sorted(
        str(name) for name, raw in parameters.items()
        if isinstance(raw, dict) and isinstance(raw.get("values"), list) and len(raw["values"]) == 1
    )
    baseline_only = sorted(str(name) for name in baseline if name not in parameters and name not in zombies)
    return {
        "detector": detector,
        "live_parameter_count": len(parameters),
        "zombie_parameters": sorted(str(name) for name in zombies),
        "singleton_parameters": singleton,
        "baseline_only_parameters": baseline_only,
        "errors": errors,
    }


def audit_detector_directory(detector_dir: Path) -> dict[str, Any]:
    records = []
    for path in sorted(detector_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.append(audit_detector_config(payload))
    return {
        "schema_version": "0.1",
        "detector_count": len(records),
        "zombie_detector_count": sum(bool(item["zombie_parameters"]) for item in records),
        "error_count": sum(len(item["errors"]) for item in records),
        "detectors": records,
    }
