from __future__ import annotations

from collections.abc import Callable

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


def run_registered_detectors(*, image_bgr: np.ndarray, mask: np.ndarray) -> list[Candidate]:
    candidates: list[Candidate] = []
    for name, detector in _REGISTRY:
        candidate = detector(image_bgr=image_bgr, mask=mask)
        if candidate.method != name:
            raise ValueError(
                f"Detector registry mismatch: registered as {name!r}, returned {candidate.method!r}"
            )
        candidates.append(candidate)
    return candidates
