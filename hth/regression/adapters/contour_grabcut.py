from __future__ import annotations

from typing import Any

import numpy as np

from hth.geometry.registry import run_registered_detector


def detect(*, image_bgr: np.ndarray, mask: np.ndarray, parameters: dict[str, Any] | None = None):
    return run_registered_detector("contour_grabcut", image_bgr=image_bgr, mask=mask, parameters=parameters)
