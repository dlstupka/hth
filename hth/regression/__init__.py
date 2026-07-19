"""Detector-agnostic regression framework."""

from .metrics import bbox_iou, edge_errors
from .parameter_space import canonical_parameters, exhaustive_parameter_sets, parameter_set_id

__all__ = [
    "bbox_iou",
    "edge_errors",
    "canonical_parameters",
    "exhaustive_parameter_sets",
    "parameter_set_id",
]
