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
    # Human-readable/provenance metadata is supplied by the registry so a
    # detector implementation only needs to return geometry and diagnostics.
    detector_name: str = ""
    origin: str = ""
    foundation: list[str] | None = None
    authors: list[str] | None = None
    version: str = ""
    repository: str = ""
