"""Black-box parameter-space utilities."""
from __future__ import annotations
import hashlib, itertools, json
from typing import Any


def canonical_parameters(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def parameter_set_id(parameters: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_parameters(parameters).encode("utf-8")).hexdigest()[:12]


def exhaustive_parameter_sets(config: dict[str, Any]) -> list[dict[str, Any]]:
    names = list(config["parameters"])
    values = [config["parameters"][name]["values"] for name in names]
    return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*values)]


def value_index(values: list[Any], value: Any) -> int:
    try:
        return values.index(value)
    except ValueError:
        return min(range(len(values)), key=lambda i: abs(float(values[i])-float(value)))
