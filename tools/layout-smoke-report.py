#!/usr/bin/env python3
"""Summarize decision-useful health metrics from paired Kraken layout evaluation.

Native segmentation JSON remains the detailed research artifact. Counts are
diagnostics, not accuracy measurements without layout ground truth.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def _valid_points(points: object, width: int, height: int, minimum: int) -> bool:
    return (
        isinstance(points, list)
        and len(points) >= minimum
        and all(
            isinstance(point, list)
            and len(point) == 2
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in point)
            and 0 <= point[0] < width
            and 0 <= point[1] < height
            for point in points
        )
    )


def inspect_segmentation(payload: dict, size: list[int]) -> dict:
    """Return only signals used for triage, comparison, or acceptance gates."""
    width, height = size
    if payload.get("type") != "baselines" or width <= 0 or height <= 0:
        raise ValueError("Expected a baseline segmentation and positive image size")
    lines = payload.get("lines")
    regions_by_type = payload.get("regions")
    if not isinstance(lines, list) or not isinstance(regions_by_type, dict):
        raise ValueError("Missing native lines or regions")
    regions = [region for group in regions_by_type.values() for region in group]
    region_ids = {region.get("id") for region in regions}
    invalid_regions = sum(
        not _valid_points(region.get("boundary"), width, height, 3)
        for region in regions
    )
    invalid_lines = sum(
        not _valid_points(line.get("baseline"), width, height, 2)
        or not _valid_points(line.get("boundary"), width, height, 3)
        for line in lines
    )
    orphan_lines = sum(
        not line.get("regions")
        or any(region_id not in region_ids for region_id in line["regions"])
        for line in lines
    )
    line_orders = payload.get("line_orders") or []
    line_ids = [line.get("id") for line in lines]
    reading_order_complete = bool(line_ids) and bool(line_orders) and any(
        isinstance(order, list)
        and len(order) == len(line_ids)
        and len(set(order)) == len(line_ids)
        and set(order) == set(line_ids)
        for order in line_orders
    )
    return {
        "lines": len(lines),
        "regions_by_type": dict(sorted((kind, len(group)) for kind, group in regions_by_type.items())),
        "invalid_line_geometry": invalid_lines,
        "invalid_region_geometry": invalid_regions,
        "orphan_lines": orphan_lines,
        "reading_order_present": bool(line_orders),
        "reading_order_complete": reading_order_complete,
    }


def summarize(
    inputs: dict,
    source_dir: Path,
    normalized_dir: Path,
    model_sha256: str,
    engine_version: str,
    device: str,
    threads: int,
    execution: dict | None = None,
) -> dict:
    if len(model_sha256) != 64 or any(char not in "0123456789abcdef" for char in model_sha256):
        raise ValueError("Full lowercase model SHA-256 required")
    if not engine_version or not device or threads <= 0:
        raise ValueError("Engine version, device, and positive thread count are required")
    views = inputs.get("views", ["source", "normalized"])
    if views not in (["source"], ["source", "normalized"]):
        raise ValueError(f"Unsupported layout input views: {views}")
    rows = []
    for page in inputs["pages"]:
        ordinal = int(page["global_ordinal"])
        name = f"fs_{ordinal:04d}.json"
        source = inspect_segmentation(
            json.loads((source_dir / name).read_text(encoding="utf-8")), page["source_size"]
        )
        row = {
            "global_ordinal": ordinal,
            "source_pixel_sha256": page["source_pixel_sha256"],
            "source": source,
        }
        if "normalized" in views:
            normalized = inspect_segmentation(
                json.loads((normalized_dir / name).read_text(encoding="utf-8")), page["normalized_size"]
            )
            row.update({
                "normalized_pixel_sha256": page["normalized_pixel_sha256"],
                "photometric_route": page["photometric_route"],
                "normalized": normalized,
                "line_count_delta": normalized["lines"] - source["lines"],
            })
        rows.append(row)
    if not rows:
        raise ValueError("No paired pages")
    summary = {}
    for view in views:
        counts = [row[view]["lines"] for row in rows]
        regions = Counter()
        for row in rows:
            regions.update(row[view]["regions_by_type"])
        summary[view] = {
            "pages": len(rows),
            "total_lines": sum(counts),
            "median_lines_per_page": statistics.median(counts),
            "total_regions_by_type": dict(sorted(regions.items())),
            "pages_without_lines": sum(count == 0 for count in counts),
            "pages_without_reading_order": sum(not row[view]["reading_order_present"] for row in rows),
            "pages_without_complete_reading_order": sum(not row[view]["reading_order_complete"] for row in rows),
            "invalid_line_geometry": sum(row[view]["invalid_line_geometry"] for row in rows),
            "invalid_region_geometry": sum(row[view]["invalid_region_geometry"] for row in rows),
            "orphan_lines": sum(row[view]["orphan_lines"] for row in rows),
        }
    report = {
        "schema_version": "1.0",
        "purpose": "layout-evaluation-health-not-accuracy",
        "evaluation_mode": inputs.get("evaluation_mode", "full"),
        "golden_set_page_count": inputs.get("golden_set_page_count", len(rows)),
        "views": views,
        "golden_set_id": inputs["golden_set_id"],
        "golden_set_sha256": inputs["golden_set_sha256"],
        "source_release": inputs["source_release"],
        "results_commit": inputs["results_commit"],
        "base_normalization_result_identity": inputs["base_normalization_result_identity"],
        "final_normalization_result_identity": inputs["final_normalization_result_identity"],
        "model_sha256": model_sha256,
        "execution": {
            "engine": "kraken",
            "version": engine_version,
            "device": device,
            "threads": threads,
            "command": "segment -bl",
        },
        "summary": summary,
        "paired_pages_with_line_count_change": (
            sum(row["line_count_delta"] != 0 for row in rows) if "normalized" in views else None
        ),
        "photometric_routes": (
            dict(sorted(Counter(row["photometric_route"] for row in rows).items()))
            if "normalized" in views else None
        ),
        "pages": rows,
    }
    if execution is not None:
        report["batch_execution"] = execution["batches"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-inputs", type=Path, required=True)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--normalized-results", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--github-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.paired_inputs.read_text(encoding="utf-8"))
    execution = json.loads(args.execution.read_text(encoding="utf-8")) if args.execution else None
    report = summarize(
        inputs, args.source_results, args.normalized_results,
        args.model_sha256, args.engine_version, args.device, args.threads, execution,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.github_summary:
        source = report["summary"]["source"]
        normalized = report["summary"].get("normalized")
        with args.github_summary.open("a", encoding="utf-8") as handle:
            handle.write(f"\n### {report['golden_set_id']} layout {report['evaluation_mode']} — diagnostic, not accuracy\n\n")
            if normalized is None:
                handle.write("Source only: matching verified normalized evidence is unavailable for this older Golden Set.\n\n")
                handle.write("| Signal | Source |\n| --- | ---: |\n")
            else:
                handle.write("| Signal | Source | Final normalized |\n| --- | ---: | ---: |\n")
            for label, key in (
                ("Pages", "pages"),
                ("Baselines", "total_lines"),
                ("Text regions", "total_regions_by_type"),
                ("Invalid line geometry", "invalid_line_geometry"),
                ("Orphan lines", "orphan_lines"),
            ):
                left = source[key].get("text", 0) if key == "total_regions_by_type" else source[key]
                if normalized is None:
                    handle.write(f"| {label} | {left} |\n")
                else:
                    right = normalized[key].get("text", 0) if key == "total_regions_by_type" else normalized[key]
                    handle.write(f"| {label} | {left} | {right} |\n")
            handle.write(
                "\nReading order: not emitted by this Kraken baseline-segmentation command; "
                "no reading-order judgment is made. Counts are review triggers, not accuracy scores.\n"
            )
    print(json.dumps({"summary": report["summary"], "paired_pages_with_line_count_change": report["paired_pages_with_line_count_change"]}, indent=2))


if __name__ == "__main__":
    main()
