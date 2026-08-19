"""Black-box parameter-space utilities."""
from __future__ import annotations
import hashlib, itertools, json
from typing import Any


def canonical_parameters(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def parameter_set_sha256(parameters: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_parameters(parameters).encode("utf-8")).hexdigest()


def parameter_set_id(parameters: dict[str, Any]) -> str:
    return parameter_set_sha256(parameters)[:12]


EQUIVALENCE_FAMILY_SENTINEL = "__HTH_EQUIVALENCE_FAMILY_ID__"


def equivalence_parameter_names(config: dict[str, Any]) -> list[str]:
    """Return the durable parameter dimensions enrolled in equivalence families.

    ``equivalence_parameters`` is deliberately independent of current liveness
    classification: once a dimension is enrolled it remains normalized for this
    detector parameter schema even if later evidence reclassifies it.  Legacy
    configurations fall back to configured zombies so existing detectors require
    no migration merely to obtain a family identity.
    """
    explicit = config.get("equivalence_parameters")
    if isinstance(explicit, list):
        return sorted({str(name) for name in explicit})
    zombies = config.get("zombie_parameters", {})
    return sorted(str(name) for name in zombies) if isinstance(zombies, dict) else []


def parameter_set_equivalence_family_payload(parameters: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    payload = dict(parameters)
    for name in equivalence_parameter_names(config):
        if name in payload:
            payload[name] = EQUIVALENCE_FAMILY_SENTINEL
    return payload


def parameter_set_equivalence_family_sha256(parameters: dict[str, Any], config: dict[str, Any]) -> str:
    payload = parameter_set_equivalence_family_payload(parameters, config)
    return hashlib.sha256(canonical_parameters(payload).encode("utf-8")).hexdigest()


def parameter_set_equivalence_family_id(parameters: dict[str, Any], config: dict[str, Any]) -> str:
    return parameter_set_equivalence_family_sha256(parameters, config)[:12]


def parameter_set_equivalence_family_size(config: dict[str, Any]) -> int:
    """Cartesian multiplicity represented by one family for enrolled dimensions."""
    specs = {}
    for name in equivalence_parameter_names(config):
        if isinstance(config.get("zombie_parameters"), dict) and name in config["zombie_parameters"]:
            specs[name] = config["zombie_parameters"][name]
        elif name in config.get("parameters", {}):
            specs[name] = config["parameters"][name]
    size = 1
    for spec in specs.values():
        values = list(spec.get("values", [])) if isinstance(spec, dict) else []
        size *= max(1, len(values))
    return size


def _parameter_specs(config: dict[str, Any], *, include_zombies: bool = False) -> dict[str, dict[str, Any]]:
    specs = {str(name): dict(spec) for name, spec in config.get("parameters", {}).items()}
    zombies = config.get("zombie_parameters", {}) if isinstance(config.get("zombie_parameters"), dict) else {}
    for name, raw in zombies.items():
        spec = dict(raw)
        values = list(spec.get("values", []))
        if not values:
            raise ValueError(f"Zombie parameter {name!r} must retain its audited values")
        if include_zombies:
            specs[str(name)] = {**spec, "values": values}
        else:
            baseline = config.get("profiles", {}).get("baseline")
            if not isinstance(baseline, dict) or str(name) not in baseline:
                raise ValueError(f"Zombie parameter {name!r} must exist in profiles.baseline")
            baseline_value = baseline[str(name)]
            if "pinned_value" in spec and spec["pinned_value"] != baseline_value:
                raise ValueError(
                    f"Zombie parameter {name!r} pinned_value must match profiles.baseline "
                    f"({spec['pinned_value']!r} != {baseline_value!r})"
                )
            specs[str(name)] = {**spec, "values": [baseline_value]}
    return specs


def exhaustive_parameter_sets(config: dict[str, Any], *, include_zombies: bool = False) -> list[dict[str, Any]]:
    """Return the declared Cartesian space.

    Normal exhaustive search keeps audited zombie parameters pinned.  The explicit
    ``exhaustive-with-zombies`` strategy passes ``include_zombies=True`` to restore
    their retained value domains for deliberate regression/revalidation.
    """
    specs = _parameter_specs(config, include_zombies=include_zombies)
    names = list(specs)
    values = [specs[name]["values"] for name in names]
    return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*values)]


def value_index(values: list[Any], value: Any) -> int:
    try:
        return values.index(value)
    except ValueError:
        return min(range(len(values)), key=lambda i: abs(float(values[i])-float(value)))


def canonical_search_space(config: dict[str, Any], strategy: str) -> dict[str, Any]:
    """Return the canonical parameter-space contract used by execution and reporting.

    Counts are derived only from declared detector metadata.  Mandatory reference
    evaluations (baseline/historic-best) never alter these universe sizes.
    """
    live_count = len(exhaustive_parameter_sets(config, include_zombies=False))
    zombie_count = len(exhaustive_parameter_sets(config, include_zombies=True))
    configured_zombies = sorted(
        str(name) for name in (config.get("zombie_parameters", {}) if isinstance(config.get("zombie_parameters"), dict) else {})
    )
    effective_count = (
        zombie_count if strategy == "exhaustive-with-zombies"
        else live_count if strategy in {"exhaustive", "cartesian"}
        else None
    )
    return {
        "schema_version": "1",
        "strategy": strategy,
        "live_exhaustive_parameter_sets": live_count,
        "exhaustive_with_zombies_parameter_sets": zombie_count,
        "effective_parameter_sets": effective_count,
        "configured_zombie_parameters": configured_zombies,
        "denominator": "exhaustive-with-zombies",
    }
