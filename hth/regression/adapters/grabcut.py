"""GrabCut adapter. The framework only sees a callable accepting image, mask and parameters."""
from __future__ import annotations
from hth.geometry.detector_grabcut import detect
__all__ = ["detect"]
