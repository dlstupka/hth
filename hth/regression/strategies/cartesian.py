"""Authoritative exhaustive Cartesian strategy."""
from __future__ import annotations
from typing import Any
from ..parameter_space import exhaustive_parameter_sets

def generate(config: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    sets = exhaustive_parameter_sets(config)
    return sets if limit is None else sets[:limit]
