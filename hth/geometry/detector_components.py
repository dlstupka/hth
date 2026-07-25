from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .common import candidate_score, valid_bbox
from .model import Candidate

METHOD = "components"

BASELINE_PARAMETERS: dict[str, int | float] = {
    "minimum_component_area_fraction": 0.0015,
    "minimum_component_area_px": 25,
    "merge_area_ratio": 0.02,
    "merge_gap_fraction": 0.035,
    "minimum_bbox_area_fraction": 0.12,
    "minimum_selected_area_fraction": 0.04,
    "bbox_padding_fraction": 0.0,
    "morphology_close_fraction": 0.008,
    "morphology_dilate_fraction": 0.015,
}


def _parameters(overrides: dict[str, Any] | None) -> dict[str, int | float]:
    values = dict(BASELINE_PARAMETERS)
    if overrides:
        unknown = sorted(set(overrides) - set(values))
        if unknown:
            raise ValueError(
                f"Unknown Connected Components parameters: {', '.join(unknown)}"
            )
        values.update(overrides)

    integer_names = {"minimum_component_area_px"}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Connected Components parameter {name!r} must be numeric")
        values[name] = int(value) if name in integer_names else float(value)

    if not 0.0 <= float(values["minimum_component_area_fraction"]) <= 1.0:
        raise ValueError("minimum_component_area_fraction must be between 0 and 1")
    if int(values["minimum_component_area_px"]) < 1:
        raise ValueError("minimum_component_area_px must be at least 1")
    if not 0.0 <= float(values["merge_area_ratio"]) <= 1.0:
        raise ValueError("merge_area_ratio must be between 0 and 1")
    if not 0.0 <= float(values["merge_gap_fraction"]) <= 0.5:
        raise ValueError("merge_gap_fraction must be between 0 and 0.5")
    for name in (
        "minimum_bbox_area_fraction",
        "minimum_selected_area_fraction",
        "morphology_close_fraction",
        "morphology_dilate_fraction",
    ):
        if not 0.0 <= float(values[name]) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not 0.0 <= float(values["bbox_padding_fraction"]) <= 0.25:
        raise ValueError("bbox_padding_fraction must be between 0 and 0.25")
    for name in ("morphology_close_fraction", "morphology_dilate_fraction"):
        if not 0.0 <= float(values[name]) <= 0.10:
            raise ValueError(f"{name} must be between 0 and 0.10")
    return values


def _odd_kernel_size(fraction: float, width: int, height: int) -> int:
    """Return a scale-relative odd morphology kernel size; zero disables it."""
    if fraction <= 0.0:
        return 0
    size = max(1, int(round(min(width, height) * fraction)))
    return size if size % 2 == 1 else size + 1


def _morphology(mask: np.ndarray, values: dict[str, int | float]) -> tuple[np.ndarray, int, int]:
    height, width = mask.shape[:2]
    binary = (mask > 0).astype(np.uint8) * 255
    close_size = _odd_kernel_size(
        float(values["morphology_close_fraction"]), width, height
    )
    dilate_size = _odd_kernel_size(
        float(values["morphology_dilate_fraction"]), width, height
    )
    processed = binary
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    if dilate_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
        processed = cv2.dilate(processed, kernel, iterations=1)
    return processed, close_size, dilate_size


def _component_label_image(labels: np.ndarray) -> np.ndarray:
    """Render deterministic colors for connected-component labels."""
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in range(1, int(labels.max()) + 1):
        # Stable high-contrast pseudo-color without external plotting dependencies.
        color = (
            (37 * label + 53) % 256,
            (97 * label + 101) % 256,
            (193 * label + 29) % 256,
        )
        output[labels == label] = color
    return output


def debug_images(
    *, mask: np.ndarray, parameters: dict[str, Any] | None = None
) -> dict[str, np.ndarray]:
    """Return Connected Components intermediate images for regression debugging."""
    values = _parameters(parameters)
    after_morphology, _, _ = _morphology(mask, values)
    _, labels, _, _ = cv2.connectedComponentsWithStats(
        (after_morphology > 0).astype(np.uint8), connectivity=8
    )
    return {
        "after-morphology.png": after_morphology,
        "component-labels.png": _component_label_image(labels),
    }


def _boxes_are_near(a: list[int], b: list[int], gap: int) -> bool:
    """Return True when two axis-aligned boxes overlap after a small expansion."""
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _union_box(a: list[int], b: list[int]) -> list[int]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _padded_bbox(
    bbox: list[int],
    padding_fraction: float,
    width: int,
    height: int,
) -> list[int]:
    padding = int(round(min(width, height) * padding_fraction))
    return [
        max(0, bbox[0] - padding),
        max(0, bbox[1] - padding),
        min(width, bbox[2] + padding),
        min(height, bbox[3] + padding),
    ]


def detect(
    *,
    image_bgr: np.ndarray,
    mask: np.ndarray,
    parameters: dict[str, Any] | None = None,
) -> Candidate:
    """Estimate a document envelope from connected foreground regions.

    The detector uses the shared document mask, seeds the envelope with the
    largest meaningful component, and merges nearby components. All geometric
    thresholds are exposed through the black-box regression parameter mapping;
    omitting ``parameters`` preserves the original detector behavior.
    """
    del image_bgr
    values = _parameters(parameters)

    if mask.ndim != 2:
        raise ValueError(
            f"Connected Components detector expects a 2-D mask, got shape {mask.shape}"
        )

    height, width = mask.shape[:2]
    image_area = max(1, width * height)
    after_morphology, close_kernel, dilate_kernel = _morphology(mask, values)
    binary = (after_morphology > 0).astype(np.uint8)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    del labels, centroids

    minimum_area = max(
        int(values["minimum_component_area_px"]),
        round(image_area * float(values["minimum_component_area_fraction"])),
    )
    components: list[dict[str, object]] = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        components.append(
            {
                "label": label,
                "area": area,
                "box": [x, y, x + component_width, y + component_height],
            }
        )

    common_diagnostics = {
        "parameters": values,
        "component_count": max(0, count - 1),
        "minimum_component_area": minimum_area,
        "morphology_close_kernel_px": close_kernel,
        "morphology_dilate_kernel_px": dilate_kernel,
    }
    if not components:
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {**common_diagnostics, "reason": "no_significant_components"},
            status="no_candidate",
        )

    components.sort(key=lambda item: int(item["area"]), reverse=True)
    largest_area = int(components[0]["area"])
    largest_component_fraction = largest_area / image_area
    merge_area_floor = max(
        minimum_area,
        round(largest_area * float(values["merge_area_ratio"])),
    )
    merge_gap = max(
        0,
        round(min(width, height) * float(values["merge_gap_fraction"])),
    )

    selected = [components[0]]
    envelope = list(components[0]["box"])
    remaining = [
        component
        for component in components[1:]
        if int(component["area"]) >= merge_area_floor
    ]

    changed = True
    while changed:
        changed = False
        next_remaining: list[dict[str, object]] = []
        for component in remaining:
            box = list(component["box"])
            if _boxes_are_near(envelope, box, merge_gap):
                selected.append(component)
                envelope = _union_box(envelope, box)
                changed = True
            else:
                next_remaining.append(component)
        remaining = next_remaining

    envelope = _padded_bbox(
        envelope,
        float(values["bbox_padding_fraction"]),
        width,
        height,
    )
    if not valid_bbox(envelope):
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {**common_diagnostics, "reason": "invalid_component_envelope"},
            status="no_candidate",
        )

    left, top, right, bottom = envelope
    bbox_area = (right - left) * (bottom - top)
    selected_area = sum(int(component["area"]) for component in selected)
    bbox_area_fraction = bbox_area / image_area
    component_area_fraction = selected_area / image_area
    fill_ratio = selected_area / max(1, bbox_area)
    original_foreground = int(np.count_nonzero(mask[top:bottom, left:right]))
    text_density = original_foreground / max(1, bbox_area)

    diagnostics = {
        **common_diagnostics,
        "significant_components": len(components),
        "merged_components": len(selected),
        "merge_area_floor": merge_area_floor,
        "merge_gap_px": merge_gap,
        "selected_component_labels": [int(item["label"]) for item in selected],
        "bbox_area_fraction": round(bbox_area_fraction, 6),
        "component_area_fraction": round(component_area_fraction, 6),
        "fill_ratio": round(fill_ratio, 6),
        "largest_component_fraction": round(largest_component_fraction, 6),
        "largest_merged_fraction": round(component_area_fraction, 6),
        "envelope_fraction": round(bbox_area_fraction, 6),
        "text_density": round(text_density, 6),
    }

    if (
        bbox_area_fraction < float(values["minimum_bbox_area_fraction"])
        or component_area_fraction
        < float(values["minimum_selected_area_fraction"])
    ):
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {**diagnostics, "reason": "component_envelope_too_small"},
            status="no_candidate",
        )

    mask_score = candidate_score(mask, envelope)
    fill_score = min(1.0, fill_ratio / 0.55)
    area_score = min(1.0, bbox_area_fraction / 0.60)
    combined = 0.65 * mask_score + 0.20 * fill_score + 0.15 * area_score

    corners = [
        [float(left), float(top)],
        [float(right), float(top)],
        [float(right), float(bottom)],
        [float(left), float(bottom)],
    ]

    diagnostics.update(
        {
            "mask_score": round(mask_score, 6),
            "fill_score": round(fill_score, 6),
            "area_score": round(area_score, 6),
        }
    )
    return Candidate(
        METHOD,
        envelope,
        corners,
        round(combined, 6),
        round(combined, 6),
        diagnostics,
    )
