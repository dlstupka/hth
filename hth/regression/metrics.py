"""Geometry regression metrics."""
from __future__ import annotations


def bbox_iou(a: list[int], b: list[int]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right-left) * max(0, bottom-top)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def edge_errors(predicted: list[int], approved: list[int]) -> dict[str, int | float]:
    values = [abs(predicted[i]-approved[i]) for i in range(4)]
    return {
        "left": values[0], "top": values[1], "right": values[2], "bottom": values[3],
        "mean": sum(values)/4.0, "maximum": max(values),
    }
