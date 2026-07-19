from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
import time
import traceback

import cv2
import numpy as np

from . import (
    detector_components,
    detector_contour,
    detector_grabcut,
    detector_hough,
    detector_lsd,
    detector_ransac,
)
from .model import Candidate
try:
    # Package mode: imported as hth.geometry.registry (tests, installed use).
    from hth.version import HTH_REPOSITORY, HTH_VERSION
except ModuleNotFoundError as exc:
    # Script mode: workflows execute hth/detect_geometry_candidates.py directly,
    # which places the hth/ directory itself on sys.path.
    if exc.name != "hth":
        raise
    from version import HTH_REPOSITORY, HTH_VERSION

Detector = Callable[..., Candidate]


@dataclass(frozen=True)
class DetectorSpec:
    """Small plugin contract; framework services stay outside detectors."""

    method: str
    name: str
    origin: str
    entrypoint: Detector
    foundation: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    version: str = ""
    repository: str = ""

    @property
    def display_name(self) -> str:
        return f"{self.name} ({self.origin})" if self.origin else self.name


# Order is intentional and preserves the pre-registry JSON candidate order.
# Method IDs remain stable for downstream compatibility; names are presentation.
_REGISTRY: tuple[DetectorSpec, ...] = (
    DetectorSpec(
        method=detector_contour.METHOD,
        name="Contour",
        origin="HTH",
        entrypoint=detector_contour.detect,
        foundation=("OpenCV",),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_components.METHOD,
        name="Connected Components",
        origin="OpenCV",
        entrypoint=detector_components.detect,
        foundation=("OpenCV",),
        authors=("OpenCV contributors",),
        version=cv2.__version__,
        repository="https://github.com/opencv/opencv",
    ),
    DetectorSpec(
        method=detector_ransac.METHOD,
        name="RANSAC",
        origin="HTH",
        entrypoint=detector_ransac.detect,
        foundation=("RANSAC", "OpenCV"),
        authors=("OpenAI ChatGPT",),
        version=HTH_VERSION,
        repository=HTH_REPOSITORY,
    ),
    DetectorSpec(
        method=detector_hough.METHOD,
        name="Hough Lines",
        origin="OpenCV",
        entrypoint=detector_hough.detect,
        foundation=("Hough transform", "OpenCV"),
        authors=("OpenCV contributors",),
        version=cv2.__version__,
        repository="https://github.com/opencv/opencv",
    ),
    DetectorSpec(
        method=detector_lsd.METHOD,
        name="Line Segment Detector",
        origin="OpenCV",
        entrypoint=detector_lsd.detect,
        foundation=("LSD", "OpenCV"),
        authors=("OpenCV contributors",),
        version=cv2.__version__,
        repository="https://github.com/opencv/opencv",
    ),
    DetectorSpec(
        method=detector_grabcut.METHOD,
        name="GrabCut",
        origin="OpenCV",
        entrypoint=detector_grabcut.detect,
        foundation=("GrabCut", "OpenCV"),
        authors=("OpenCV contributors",),
        version=cv2.__version__,
        repository="https://github.com/opencv/opencv",
    ),
)


def detector_specs() -> list[DetectorSpec]:
    return list(_REGISTRY)


def detector_names() -> list[str]:
    return [spec.method for spec in _REGISTRY]


def detector_catalog() -> list[dict[str, Any]]:
    return [
        {
            "method": spec.method,
            "name": spec.name,
            "display_name": spec.display_name,
            "origin": spec.origin,
            "foundation": list(spec.foundation),
            "authors": list(spec.authors),
            "version": spec.version,
            "repository": spec.repository,
        }
        for spec in _REGISTRY
    ]


def _apply_spec(candidate: Candidate, spec: DetectorSpec) -> Candidate:
    candidate.detector_name = spec.name
    candidate.origin = spec.origin
    candidate.foundation = list(spec.foundation)
    candidate.authors = list(spec.authors)
    candidate.version = spec.version
    candidate.repository = spec.repository
    return candidate


def _failed_candidate(
    spec: DetectorSpec,
    exc: BaseException,
    *,
    elapsed_ms: float,
) -> Candidate:
    """Represent a detector exception as data instead of aborting the page."""
    return _apply_spec(
        Candidate(
            method=spec.method,
            bbox=None,
            corners=None,
            confidence=0.0,
            score=0.0,
            diagnostics={
                "reason": "detector_exception",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(limit=8),
                "elapsed_ms": round(elapsed_ms, 3),
            },
            status="error",
        ),
        spec,
    )


def _normalize_candidate(
    spec: DetectorSpec,
    candidate: Candidate,
    *,
    elapsed_ms: float,
) -> Candidate:
    if not isinstance(candidate, Candidate):
        raise TypeError(
            f"Detector {spec.method!r} returned {type(candidate).__name__}, "
            "expected Candidate"
        )
    if candidate.method != spec.method:
        raise ValueError(
            f"Detector registry mismatch: registered as {spec.method!r}, "
            f"returned {candidate.method!r}"
        )

    candidate.diagnostics = dict(candidate.diagnostics or {})
    candidate.diagnostics.setdefault("elapsed_ms", round(elapsed_ms, 3))

    if candidate.status not in {"ok", "no_candidate", "error"}:
        raise ValueError(
            f"Detector {spec.method!r} returned invalid status {candidate.status!r}"
        )

    if candidate.status == "ok" and candidate.bbox is None:
        candidate.status = "no_candidate"

    if candidate.status != "ok":
        candidate.confidence = 0.0
        candidate.score = 0.0

    return _apply_spec(candidate, spec)


def run_registered_detectors(*, image_bgr: np.ndarray, mask: np.ndarray) -> list[Candidate]:
    """Run every registered detector independently with timing and isolation."""
    candidates: list[Candidate] = []
    for spec in _REGISTRY:
        started = time.perf_counter()
        try:
            candidate = spec.entrypoint(image_bgr=image_bgr, mask=mask)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            candidate = _normalize_candidate(spec, candidate, elapsed_ms=elapsed_ms)
        except Exception as exc:  # Detector plugins are an isolation boundary.
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            candidate = _failed_candidate(spec, exc, elapsed_ms=elapsed_ms)
        candidates.append(candidate)
    return candidates


def summarize_candidates(candidates: list[Candidate]) -> dict[str, object]:
    counts = Counter(candidate.status for candidate in candidates)
    return {
        "status_counts": {
            "ok": counts.get("ok", 0),
            "no_candidate": counts.get("no_candidate", 0),
            "error": counts.get("error", 0),
        },
        "failed_methods": [
            candidate.method for candidate in candidates if candidate.status == "error"
        ],
        "no_candidate_methods": [
            candidate.method
            for candidate in candidates
            if candidate.status == "no_candidate"
        ],
    }
