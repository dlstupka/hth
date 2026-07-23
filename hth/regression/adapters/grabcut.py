"""GrabCut adapter using the authoritative detector registry contract."""
from __future__ import annotations

from typing import Any

import numpy as np

from hth.geometry.model import Candidate
from hth.geometry.registry import run_registered_detector


def detect(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> Candidate:
    return run_registered_detector(
        "grabcut",
        image_bgr=image_bgr,
        mask=mask,
        parameters=parameters,
    )


__all__ = ["detect"]
