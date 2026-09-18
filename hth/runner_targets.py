"""Canonical runner-target catalog shared by dispatch and workflow tooling."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "runner-targets.json"
SCHEMA_VERSION = "1.0"


def load_runner_targets(path: Path = CATALOG_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Runner-target catalog has an unsupported schema")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Runner-target catalog has no targets")
    seen: set[str] = set()
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError("Runner-target catalog contains an invalid target")
        target_id = str(target.get("id") or "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", target_id) or target_id in seen:
            raise ValueError(f"Runner-target catalog contains an invalid or duplicate ID: {target_id!r}")
        runs_on = target.get("runs_on")
        if not isinstance(runs_on, list) or not runs_on or not all(
            isinstance(label, str) and label.strip() for label in runs_on
        ):
            raise ValueError(f"Runner target {target_id!r} has no valid runs_on labels")
        if not str(target.get("setup_label") or "").strip():
            raise ValueError(f"Runner target {target_id!r} has no setup_label")
        seen.add(target_id)
    default = str(payload.get("default_target") or "")
    if default not in seen:
        raise ValueError("Runner-target catalog default_target is not defined")
    return payload


def runner_target_ids(path: Path = CATALOG_PATH) -> list[str]:
    return [str(target["id"]) for target in load_runner_targets(path)["targets"]]


def resolve_runner_target(target_id: str, path: Path = CATALOG_PATH) -> dict[str, Any]:
    requested = str(target_id or "").strip()
    for target in load_runner_targets(path)["targets"]:
        if target["id"] == requested:
            return {
                "id": requested,
                "description": str(target.get("description") or ""),
                "runs_on": list(target["runs_on"]),
                "setup_label": str(target["setup_label"]),
            }
    raise ValueError(f"Unknown runner target: {requested!r}")
