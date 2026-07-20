"""Authoritative exhaustive Cartesian strategy."""
from __future__ import annotations
from typing import Any
from ..parameter_space import canonical_parameters, exhaustive_parameter_sets


def generate(config: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    """Return the normal Cartesian space with configured named profiles first.

    Named profiles are ordinary parameter sets, not a separate execution path. Putting
    them first guarantees that a limited CI run still evaluates its configured baseline
    while a full run continues to evaluate the same unique Cartesian parameter space.
    """
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    for parameters in config.get("profiles", {}).values():
        canonical = canonical_parameters(parameters)
        if canonical not in seen:
            ordered.append(dict(parameters))
            seen.add(canonical)

    for parameters in exhaustive_parameter_sets(config):
        canonical = canonical_parameters(parameters)
        if canonical not in seen:
            ordered.append(parameters)
            seen.add(canonical)

    return ordered if limit is None else ordered[:limit]
