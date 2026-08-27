"""Canonical persistence primitives for HTH results repositories.

Durable per-run evidence is authoritative. Aggregate JSON indexes are derived,
rebuildable views. All results-repository readers and writers should cross this
boundary rather than open/write index files ad hoc.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hth.contracts import (
    CALIBRATION_INDEX_SCHEMA_VERSION,
    OPTIMIZER_INDEX_SCHEMA_VERSION,
    PARALLELISM_INDEX_SCHEMA_VERSION,
    RUNTIME_INDEX_SCHEMA_VERSION,
    adapt_calibration_index,
    adapt_optimizer_index,
    adapt_parallelism_index,
    adapt_runtime_index,
)

INDEX_DIRECTORY = "indexes"


@dataclass(frozen=True)
class IndexContract:
    filename: str
    schema_version: str
    adapter: Callable[[dict[str, Any]], dict[str, Any]]
    empty: Callable[[], dict[str, Any]]


def _empty_calibration() -> dict[str, Any]:
    return {"schema_version": CALIBRATION_INDEX_SCHEMA_VERSION, "entries": [], "preferred": {}}


def _empty_runtime() -> dict[str, Any]:
    return {"schema_version": RUNTIME_INDEX_SCHEMA_VERSION, "observations": [], "latest": {}}


def _empty_parallelism() -> dict[str, Any]:
    return {"schema_version": PARALLELISM_INDEX_SCHEMA_VERSION, "observations": [], "shard_observations": []}


def _empty_optimizer() -> dict[str, Any]:
    return {"schema_version": OPTIMIZER_INDEX_SCHEMA_VERSION, "detectors": {}, "runs": {}}


def _identity(payload: dict[str, Any]) -> dict[str, Any]:
    return dict(payload)


def _empty_observations() -> dict[str, Any]:
    return {"schema_version": 1, "observations": []}


def _empty_predictions() -> dict[str, Any]:
    return {"schema_version": "1.0", "predictions": []}


INDEX_CONTRACTS: dict[str, IndexContract] = {
    "calibration-index.json": IndexContract("calibration-index.json", CALIBRATION_INDEX_SCHEMA_VERSION, adapt_calibration_index, _empty_calibration),
    "runtime-index.json": IndexContract("runtime-index.json", RUNTIME_INDEX_SCHEMA_VERSION, adapt_runtime_index, _empty_runtime),
    "parallelism-index.json": IndexContract("parallelism-index.json", PARALLELISM_INDEX_SCHEMA_VERSION, adapt_parallelism_index, _empty_parallelism),
    "optimizer-index.json": IndexContract("optimizer-index.json", OPTIMIZER_INDEX_SCHEMA_VERSION, adapt_optimizer_index, _empty_optimizer),
    "optimizer-predictions.json": IndexContract("optimizer-predictions.json", "1.0", _identity, _empty_predictions),
    "multidetector-index.json": IndexContract("multidetector-index.json", "1", _identity, _empty_observations),
    "parameter-provenance-index.json": IndexContract("parameter-provenance-index.json", "1", _identity, _empty_observations),
    "orli-evidence-index.json": IndexContract("orli-evidence-index.json", "1", _identity, _empty_observations),
}
INDEX_FILENAMES = frozenset(INDEX_CONTRACTS)


def contract_for(filename: str) -> IndexContract:
    try:
        return INDEX_CONTRACTS[filename]
    except KeyError as exc:
        raise ValueError(f"Unknown HTH results index: {filename}") from exc


def canonical_index_path(results_root: Path, filename: str) -> Path:
    contract_for(filename)
    return Path(results_root) / INDEX_DIRECTORY / filename


def legacy_index_path(results_root: Path, filename: str) -> Path:
    contract_for(filename)
    return Path(results_root) / filename


def readable_index_path(results_root: Path, filename: str) -> Path:
    canonical = canonical_index_path(results_root, filename)
    if canonical.is_file():
        return canonical
    legacy = legacy_index_path(results_root, filename)
    return legacy if legacy.is_file() else canonical


def index_results_root(index_path: Path) -> Path:
    path = Path(index_path)
    return path.parent.parent if path.parent.name == INDEX_DIRECTORY else path.parent


def resolve_index_relative_path(index_path: Path, relative_path: str | Path) -> Path:
    return index_results_root(index_path) / Path(relative_path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace one JSON object without exposing a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def load_index(results_root: Path, filename: str) -> dict[str, Any]:
    contract = contract_for(filename)
    path = readable_index_path(results_root, filename)
    if not path.is_file():
        return contract.empty()
    return contract.adapter(read_json(path))


def load_index_path(index_path: Path, filename: str | None = None) -> dict[str, Any]:
    path = Path(index_path)
    name = filename or path.name
    return load_index(index_results_root(path), name)


def write_index(results_root: Path, filename: str, payload: dict[str, Any]) -> Path:
    contract = contract_for(filename)
    data = contract.adapter(dict(payload))
    if data.get("schema_version") in (None, "legacy"):
        data["schema_version"] = contract.schema_version
    path = canonical_index_path(results_root, filename)
    atomic_write_json(path, data)
    return path


def legacy_indexes(results_root: Path) -> list[Path]:
    root = Path(results_root)
    return [root / filename for filename in sorted(INDEX_FILENAMES) if (root / filename).exists()]


def remove_legacy_indexes(results_root: Path) -> list[Path]:
    removed: list[Path] = []
    for path in legacy_indexes(results_root):
        path.unlink()
        removed.append(path)
    return removed
