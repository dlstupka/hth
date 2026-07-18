from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Candidate:
    method: str
    bbox: list[int] | None
    corners: list[list[float]] | None
    confidence: float
    score: float
    diagnostics: dict[str, Any]
