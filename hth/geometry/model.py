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
    # "ok" means a usable candidate was returned, "no_candidate" means the
    # detector ran normally but found no plausible geometry, and "error"
    # means the detector raised or violated the detector contract.
    status: str = "ok"
