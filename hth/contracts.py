"""Versioned persistence contracts and compatibility adapters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from hth.domain.result_metrics import normalize_summary_metrics

CALIBRATION_INDEX_SCHEMA_VERSION = "1.0"
RUNTIME_INDEX_SCHEMA_VERSION = "1.0"
PARALLELISM_INDEX_SCHEMA_VERSION = "2.3"
OPTIMIZER_INDEX_SCHEMA_VERSION = "2.1"
REGRESSION_SUMMARY_SCHEMA_VERSION = "0.8"
OPTIMIZER_OBSERVATION_SCHEMA_VERSION = "1.0"
RUNTIME_OBSERVATION_SCHEMA_VERSION = "1.0"


def _object(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def read_json_contract(path: Path, adapter: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return adapter(_object(payload, str(path)))


def adapt_calibration_index(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(_object(payload, "calibration-index"))
    data.setdefault("schema_version", "legacy")
    data["entries"] = [row for row in data.get("entries", []) if isinstance(row, dict)]
    data["preferred"] = data.get("preferred") if isinstance(data.get("preferred"), dict) else {}
    return data


def adapt_runtime_index(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(_object(payload, "runtime-index"))
    data.setdefault("schema_version", "legacy")
    data["observations"] = [row for row in data.get("observations", []) if isinstance(row, dict)]
    data["latest"] = data.get("latest") if isinstance(data.get("latest"), dict) else {}
    return data


def adapt_parallelism_index(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(_object(payload, "parallelism-index"))
    data.setdefault("schema_version", "legacy")
    data["observations"] = [row for row in data.get("observations", []) if isinstance(row, dict)]
    data["shard_observations"] = [row for row in data.get("shard_observations", []) if isinstance(row, dict)]
    return data


def adapt_optimizer_index(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(_object(payload, "optimizer-index"))
    data.setdefault("schema_version", "legacy")
    data["detectors"] = data.get("detectors") if isinstance(data.get("detectors"), dict) else {}
    data["runs"] = data.get("runs") if isinstance(data.get("runs"), dict) else {}
    return data


def adapt_regression_summary(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(_object(payload, "regression-summary"))
    data.setdefault("schema_version", "legacy")
    return normalize_summary_metrics(data)
