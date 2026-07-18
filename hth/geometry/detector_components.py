from __future__ import annotations

import cv2
import numpy as np

from .common import candidate_score, valid_bbox
from .model import Candidate

METHOD = "components"


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


def detect(*, image_bgr: np.ndarray, mask: np.ndarray) -> Candidate:
    """Estimate a document envelope from connected foreground regions.

    The detector intentionally uses the shared document mask rather than image
    color directly. It starts with the largest meaningful connected component
    and merges nearby components, allowing a fragmented page mask to produce a
    single conservative page envelope.
    """
    del image_bgr

    height, width = mask.shape[:2]
    image_area = max(1, width * height)
    binary = (mask > 0).astype(np.uint8)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    del centroids

    # Label zero is the background. Ignore tiny specks but retain fragments
    # large enough to contribute to a page envelope.
    minimum_area = max(25, round(image_area * 0.0015))
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

    if not components:
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {
                "reason": "no_significant_components",
                "component_count": max(0, count - 1),
                "minimum_component_area": minimum_area,
            },
        )

    components.sort(key=lambda item: int(item["area"]), reverse=True)
    largest_area = int(components[0]["area"])

    # Components much smaller than the seed are usually text/noise detached
    # from the page body. Keep a modest floor so genuinely split page regions
    # can still be merged.
    merge_area_floor = max(minimum_area, round(largest_area * 0.02))
    merge_gap = max(3, round(min(width, height) * 0.035))

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

    if not valid_bbox(envelope):
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {"reason": "invalid_component_envelope"},
        )

    left, top, right, bottom = envelope
    bbox_area = (right - left) * (bottom - top)
    selected_area = sum(int(component["area"]) for component in selected)
    bbox_area_fraction = bbox_area / image_area
    component_area_fraction = selected_area / image_area
    fill_ratio = selected_area / max(1, bbox_area)

    # Reject envelopes that are too small to plausibly describe a photographed
    # page. This is a normal miss, not a detector error.
    if bbox_area_fraction < 0.12 or component_area_fraction < 0.04:
        return Candidate(
            METHOD,
            None,
            None,
            0.0,
            0.0,
            {
                "reason": "component_envelope_too_small",
                "component_count": max(0, count - 1),
                "significant_components": len(components),
                "merged_components": len(selected),
                "bbox_area_fraction": round(bbox_area_fraction, 6),
                "component_area_fraction": round(component_area_fraction, 6),
            },
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

    return Candidate(
        METHOD,
        envelope,
        corners,
        round(combined, 6),
        round(combined, 6),
        {
            "component_count": max(0, count - 1),
            "significant_components": len(components),
            "merged_components": len(selected),
            "minimum_component_area": minimum_area,
            "merge_area_floor": merge_area_floor,
            "merge_gap_px": merge_gap,
            "selected_component_labels": [int(item["label"]) for item in selected],
            "bbox_area_fraction": round(bbox_area_fraction, 6),
            "component_area_fraction": round(component_area_fraction, 6),
            "fill_ratio": round(fill_ratio, 6),
            "mask_score": round(mask_score, 6),
            "fill_score": round(fill_score, 6),
            "area_score": round(area_score, 6),
        },
    )
