from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import time
import traceback

import numpy as np

from . import detector_contour, detector_hough, detector_ransac
from .model import Candidate

Detector = Callable[..., Candidate]

# Order is intentional and preserves the pre-registry JSON candidate order.
_REGISTRY: tuple[tuple[str, Detector], ...] = (
    (detector_contour.METHOD, detector_contour.detect),
    (detector_ransac.METHOD, detector_ransac.detect),
    (detector_hough.METHOD, detector_hough.detect),
)


def detector_names() -> list[str]:
    return [name for name, _ in _REGISTRY]


def _failed_candidate(
    method: str,
    exc: BaseException,
    *,
    elapsed_ms: float,
) -> Candidate:
    """Represent a detector exception as data instead of aborting the page."""
    return Candidate(
        method=method,
        bbox=None,
        corners=None,
        confidence=0.0,
        score=0.0,
        diagnostics={
            "reason": "detector_exception",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            # A short traceback is invaluable in Actions artifacts while still
            # keeping page-analysis.json reasonably compact.
            "traceback": traceback.format_exc(limit=8),
            "elapsed_ms": round(elapsed_ms, 3),
        },
        status="error",
    )


def _normalize_candidate(
    registered_name: str,
    candidate: Candidate,
    *,
    elapsed_ms: float,
) -> Candidate:
    if not isinstance(candidate, Candidate):
        raise TypeError(
            f"Detector {registered_name!r} returned {type(candidate).__name__}, "
            "expected Candidate"
        )
    if candidate.method != registered_name:
        raise ValueError(
            f"Detector registry mismatch: registered as {registered_name!r}, "
            f"returned {candidate.method!r}"
        )

    candidate.diagnostics = dict(candidate.diagnostics or {})
    candidate.diagnostics.setdefault("elapsed_ms", round(elapsed_ms, 3))

    if candidate.status not in {"ok", "no_candidate", "error"}:
        raise ValueError(
            f"Detector {registered_name!r} returned invalid status {candidate.status!r}"
        )

    # Existing detectors signal a normal miss with bbox=None. Preserve their
    # implementations while making the result explicit to downstream tools.
    if candidate.status == "ok" and candidate.bbox is None:
        candidate.status = "no_candidate"

    if candidate.status != "ok":
        candidate.confidence = 0.0
        candidate.score = 0.0

    return candidate


def run_registered_detectors(*, image_bgr: np.ndarray, mask: np.ndarray) -> list[Candidate]:
    """Run every registered detector independently.

    One detector may fail without preventing the remaining detectors from
    running or preventing the page-analysis output from being written.
    """
    candidates: list[Candidate] = []
    for name, detector in _REGISTRY:
        started = time.perf_counter()
        try:
            candidate = detector(image_bgr=image_bgr, mask=mask)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            candidate = _normalize_candidate(name, candidate, elapsed_ms=elapsed_ms)
        except Exception as exc:  # Detector plugins are an isolation boundary.
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            candidate = _failed_candidate(name, exc, elapsed_ms=elapsed_ms)
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
