"""Absolute, storage-efficient parameter-set provenance.

HTH keeps the historical 12-character parameter_set_id as a human-facing legacy
alias, but never relies on it as the authoritative identity.  New runs also
record a detector-namespaced full SHA-256 over the exact canonical parameters.

Cartesian spaces are stored once as an ordered grid definition.  A grid ordinal
is therefore enough to reconstruct any grid member exactly; only evaluated
parameter sets that cannot be reconstructed from that grid are stored explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

from .parameter_space import (
    canonical_parameters, parameter_set_id, _parameter_specs,
    parameter_set_equivalence_family_id, parameter_set_equivalence_family_sha256,
    parameter_set_equivalence_family_size, equivalence_parameter_names,
)

PROVENANCE_SCHEMA_VERSION = "1.0"
IDENTITY_SCHEMA_VERSION = "1"
DEFAULT_PARAMETER_SCHEMA_VERSION = "1"


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def parameter_schema_version(config: dict[str, Any]) -> str:
    value = config.get("parameter_schema_version", DEFAULT_PARAMETER_SCHEMA_VERSION)
    return str(value)


def identity_payload(detector: str, parameters: dict[str, Any], *, schema_version: str) -> dict[str, Any]:
    return {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "detector": str(detector),
        "parameter_schema_version": str(schema_version),
        "parameters": parameters,
    }


def parameter_identity_sha256(detector: str, parameters: dict[str, Any], *, schema_version: str) -> str:
    return hashlib.sha256(_canonical_bytes(identity_payload(detector, parameters, schema_version=schema_version))).hexdigest()


def grid_definition(config: dict[str, Any], *, include_zombies: bool = False) -> dict[str, Any]:
    # Use the same resolved parameter specification as execution.  This prevents
    # baseline-pinned equivalence/zombie dimensions from becoming false non-grid members.
    parameters = _parameter_specs(config, include_zombies=include_zombies)
    names = list(parameters)
    values = [list(parameters[name].get("values", [])) for name in names]
    definition = {
        "order": names,
        "values": {name: values[index] for index, name in enumerate(names)},
    }
    grid_sha = hashlib.sha256(_canonical_bytes(definition)).hexdigest()
    count = 1
    for domain in values:
        count *= len(domain)
    return {
        "sha256": grid_sha,
        "parameter_order": names,
        "values": definition["values"],
        "cartesian_count": count,
        "ordinal_base": 0,
        "ordering": "itertools.product-rightmost-fastest",
    }


def _canonical_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def grid_ordinal(config: dict[str, Any], parameters: dict[str, Any], *, include_zombies: bool = False) -> int | None:
    grid = grid_definition(config, include_zombies=include_zombies)
    names = grid["parameter_order"]
    if set(parameters) != set(names):
        return None
    ordinal = 0
    multiplier = 1
    for name in reversed(names):
        domain = list(grid["values"][name])
        target = _canonical_value(parameters[name])
        matches = [index for index, value in enumerate(domain) if _canonical_value(value) == target]
        if not matches:
            return None
        ordinal += matches[0] * multiplier
        multiplier *= len(domain)
    return ordinal


def parameters_from_ordinal(provenance: dict[str, Any], ordinal: int) -> dict[str, Any]:
    grid = provenance.get("grid", {})
    names = list(grid.get("parameter_order", []))
    values = grid.get("values", {})
    count = int(grid.get("cartesian_count", 0) or 0)
    ordinal = int(ordinal)
    if ordinal < 0 or ordinal >= count:
        raise ValueError(f"Grid ordinal outside [0, {count}): {ordinal}")
    indexes: dict[str, int] = {}
    remainder = ordinal
    for name in reversed(names):
        domain = list(values[name])
        indexes[name] = remainder % len(domain)
        remainder //= len(domain)
    return {name: list(values[name])[indexes[name]] for name in names}


def attach_identity(result: dict[str, Any], detector: str, config: dict[str, Any], *, strategy: str = "exhaustive") -> dict[str, Any]:
    parameters = dict(result.get("parameters") or {})
    schema = parameter_schema_version(config)
    result["legacy_parameter_set_id"] = str(result.get("parameter_set_id") or parameter_set_id(parameters))
    result["parameter_set_id"] = result["legacy_parameter_set_id"]
    result["parameter_identity_sha256"] = parameter_identity_sha256(detector, parameters, schema_version=schema)
    result["parameter_set_equivalence_family_id"] = parameter_set_equivalence_family_id(parameters, config)
    result["parameter_set_equivalence_family_sha256"] = parameter_set_equivalence_family_sha256(parameters, config)
    result["parameter_set_equivalence_family_size"] = parameter_set_equivalence_family_size(config)
    result["parameter_identity_schema_version"] = IDENTITY_SCHEMA_VERSION
    result["parameter_schema_version"] = schema
    include_zombies = strategy == "exhaustive-with-zombies"
    ordinal = grid_ordinal(config, parameters, include_zombies=include_zombies)
    result["parameter_grid_ordinal"] = ordinal
    result["parameter_grid_sha256"] = grid_definition(config, include_zombies=include_zombies)["sha256"] if ordinal is not None else None
    return result


def build_provenance(
    detector: str,
    config: dict[str, Any],
    results: Iterable[dict[str, Any]],
    *,
    strategy: str,
    complete_cartesian: bool,
) -> dict[str, Any]:
    schema = parameter_schema_version(config)
    grid = grid_definition(config, include_zombies=(strategy == "exhaustive-with-zombies"))
    profiles = {
        str(name): dict(parameters)
        for name, parameters in (config.get("profiles", {}) or {}).items()
        if isinstance(parameters, dict)
    }
    explicit: dict[str, dict[str, Any]] = {}
    evaluated = 0
    grid_evaluated = 0
    for result in results:
        attach_identity(result, detector, config, strategy=strategy)
        evaluated += 1
        if result.get("parameter_grid_ordinal") is not None:
            grid_evaluated += 1
            continue
        full = str(result["parameter_identity_sha256"])
        explicit[full] = {
            "sha256": full,
            "legacy_parameter_set_id": result["legacy_parameter_set_id"],
            "parameters": dict(result["parameters"]),
        }

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "identity": {
            "identity_schema_version": IDENTITY_SCHEMA_VERSION,
            "detector": detector,
            "parameter_schema_version": schema,
            "authoritative_key": "parameter_identity_sha256",
            "legacy_alias": "parameter_set_id",
            "legacy_alias_algorithm": "sha256(canonical-parameters)[:12]",
            "equivalence_family_algorithm": "sha256(canonical-parameters-with-enrolled-dimensions-normalized)[:12]",
            "equivalence_family_sentinel": "__HTH_EQUIVALENCE_FAMILY_ID__",
            "equivalence_parameters": equivalence_parameter_names(config),
            "equivalence_family_size": parameter_set_equivalence_family_size(config),
        },
        "grid": grid,
        "coverage": {
            "strategy": strategy,
            "complete_cartesian": bool(complete_cartesian),
            "evaluated_parameter_sets": evaluated,
            "evaluated_grid_parameter_sets": grid_evaluated,
            "explicit_non_grid_parameter_sets": len(explicit),
        },
        "profiles": profiles,
        "explicit_parameter_sets": explicit,
        "resolution": {
            "grid_members": "reconstruct from grid ordinal or scan deterministic grid by legacy/full id",
            "non_grid_members": "lookup explicit_parameter_sets",
        },
    }


def _iter_grid(provenance: dict[str, Any]):
    grid = provenance.get("grid", {})
    names = list(grid.get("parameter_order", []))
    domains = [list(grid.get("values", {}).get(name, [])) for name in names]
    for ordinal, combo in enumerate(itertools.product(*domains)):
        yield ordinal, dict(zip(names, combo, strict=True))


def resolve_parameter_set(provenance: dict[str, Any], identifier: str) -> dict[str, Any] | None:
    identifier = str(identifier).strip().lower()
    identity = provenance.get("identity", {})
    detector = str(identity.get("detector") or "")
    schema = str(identity.get("parameter_schema_version") or DEFAULT_PARAMETER_SCHEMA_VERSION)

    explicit = provenance.get("explicit_parameter_sets", {})
    if identifier in explicit:
        row = dict(explicit[identifier])
        row["source"] = "explicit"
        return row
    for full, row in explicit.items():
        if str(row.get("legacy_parameter_set_id") or "").lower() == identifier or full.startswith(identifier):
            resolved = dict(row)
            resolved["source"] = "explicit"
            return resolved

    for name, parameters in (provenance.get("profiles", {}) or {}).items():
        legacy = parameter_set_id(parameters)
        full = parameter_identity_sha256(detector, parameters, schema_version=schema)
        if identifier == legacy or full == identifier or full.startswith(identifier):
            return {
                "sha256": full,
                "legacy_parameter_set_id": legacy,
                "profile": name,
                "parameters": parameters,
                "source": "profile",
            }

    for ordinal, parameters in _iter_grid(provenance):
        legacy = parameter_set_id(parameters)
        full = parameter_identity_sha256(detector, parameters, schema_version=schema)
        if identifier == legacy or full == identifier or full.startswith(identifier):
            return {
                "sha256": full,
                "legacy_parameter_set_id": legacy,
                "grid_ordinal": ordinal,
                "parameters": parameters,
                "source": "grid",
            }
    return None


def provenance_from_legacy_parameters(parameters_path: Path) -> dict[str, Any]:
    payload = json.loads(parameters_path.read_text(encoding="utf-8"))
    config = payload.get("configuration")
    if not isinstance(config, dict):
        raise ValueError(f"No detector configuration embedded in {parameters_path}")
    detector = str(payload.get("detector") or config.get("detector") or "unknown")
    return build_provenance(
        detector,
        config,
        [],
        strategy=str(payload.get("strategy") or "legacy"),
        complete_cartesian=False,
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    resolve = sub.add_parser("resolve")
    source = resolve.add_mutually_exclusive_group(required=True)
    source.add_argument("--provenance", type=Path)
    source.add_argument("--legacy-parameters", type=Path)
    resolve.add_argument("--id", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.provenance:
        provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    else:
        provenance = provenance_from_legacy_parameters(args.legacy_parameters)
    resolved = resolve_parameter_set(provenance, args.id)
    if resolved is None:
        print(json.dumps({"found": False, "id": args.id}, indent=2))
        return 1
    print(json.dumps({"found": True, **resolved}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
