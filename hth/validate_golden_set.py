#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def bbox_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union


def edge_error(a: list[int], b: list[int]) -> int:
    return max(abs(x - y) for x, y in zip(a, b))


def actual_bbox(record: dict[str, Any]) -> list[int] | None:
    for keys in (
        ("page_left_px", "page_top_px", "page_right_px", "page_bottom_px"),
        ("content_left_px", "content_top_px", "content_right_px", "content_bottom_px"),
    ):
        if all(key in record for key in keys):
            return [int(record[key]) for key in keys]
    value = record.get("physical_document_bbox")
    return value if isinstance(value, list) and len(value) == 4 else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    golden = json.loads(args.golden.read_text(encoding="utf-8"))
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    records = {
        int(record["global_ordinal"]): record
        for record in analysis.get("records", [])
    }

    minimum_iou = float(golden["acceptance"]["minimum_iou"])
    maximum_edge_error = int(golden["acceptance"]["maximum_edge_error_px"])
    require_layout = bool(golden["acceptance"]["require_layout_match"])

    results = []
    failures = 0
    skips = 0

    for expected in golden["pages"]:
        ordinal = int(expected["global_ordinal"])
        record = records.get(ordinal)
        expected_box = expected.get("physical_document_bbox")

        if record is None:
            results.append({"global_ordinal": ordinal, "status": "fail", "reasons": ["missing_record"]})
            failures += 1
            continue

        if expected_box is None:
            results.append({"global_ordinal": ordinal, "status": "skip", "reasons": ["golden_bbox_not_populated"]})
            skips += 1
            continue

        found_box = actual_bbox(record)
        reasons = []
        iou = None
        error = None

        if found_box is None:
            reasons.append("missing_actual_bbox")
        else:
            iou = bbox_iou(expected_box, found_box)
            error = edge_error(expected_box, found_box)
            if iou < minimum_iou:
                reasons.append(f"iou_below_threshold:{iou:.4f}")
            if error > maximum_edge_error:
                reasons.append(f"edge_error_exceeded:{error}")

        expected_layout = expected["layout_type"]
        found_layout = record.get("layout_type") or record.get("layout") or "unknown"
        if require_layout and found_layout != expected_layout:
            reasons.append(f"layout_mismatch:{found_layout}!={expected_layout}")

        status = "pass" if not reasons else "fail"
        failures += int(status == "fail")
        results.append({
            "global_ordinal": ordinal,
            "status": status,
            "expected_layout": expected_layout,
            "actual_layout": found_layout,
            "expected_bbox": expected_box,
            "actual_bbox": found_box,
            "iou": None if iou is None else round(iou, 6),
            "maximum_edge_error_px": error,
            "reasons": reasons,
        })

    report = {
        "schema_version": "0.1",
        "collection_id": golden.get("collection_id", ""),
        "golden_page_count": len(golden["pages"]),
        "pass_count": sum(item["status"] == "pass" for item in results),
        "fail_count": failures,
        "skip_count": skips,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
