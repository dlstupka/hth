#!/usr/bin/env python3
"""Generate isolated, registered physical-document geometry candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import cv2

from geometry.common import document_mask, resize_for_analysis, scale_bbox
from geometry.model import Candidate
from geometry.registry import (
    detector_catalog,
    detector_names,
    run_registered_detectors,
    summarize_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dimension", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fail-on",
        choices=("never", "page-error", "detector-error"),
        default="never",
        help=(
            "Exit policy after writing output. Default 'never' makes detector "
            "plugins diagnostic-only; global setup/read/write failures still abort."
        ),
    )
    return parser.parse_args()


def run_detectors(image_path: Path, maximum: int) -> list[Candidate]:
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    original_height, original_width = original.shape[:2]
    image, scale = resize_for_analysis(original, maximum)
    mask, mask_diag = document_mask(image)
    candidates = run_registered_detectors(image_bgr=image, mask=mask)

    inverse = 1.0 / scale
    for candidate in candidates:
        candidate.diagnostics["mask"] = mask_diag
        candidate.diagnostics["analysis_scale"] = round(scale, 8)
        if candidate.bbox is not None:
            candidate.bbox = scale_bbox(
                candidate.bbox, inverse, original_width, original_height
            )
        if candidate.corners is not None:
            candidate.corners = [
                [
                    round(max(0.0, min(original_width, point[0] * inverse)), 3),
                    round(max(0.0, min(original_height, point[1] * inverse)), 3),
                ]
                for point in candidate.corners
            ]

    return candidates


def path_for_record(record: dict[str, Any], image_root: Path) -> Path:
    for key in ("extracted_file", "raw_file", "image_file", "output_file", "path"):
        value = record.get(key)
        if value:
            candidate = image_root / str(value)
            if candidate.exists():
                return candidate

    ordinal = int(record["global_ordinal"])
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"):
        candidate = image_root / "raw" / f"fs_{ordinal:04d}{suffix}"
        if candidate.exists():
            return candidate
        candidate = image_root / f"fs_{ordinal:04d}{suffix}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No image found for global ordinal {ordinal}")


def _page_status(candidates: list[Candidate]) -> str:
    statuses = {candidate.status for candidate in candidates}
    if statuses == {"error"}:
        return "error"
    if "error" in statuses:
        return "partial"
    return "complete"


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists; pass --overwrite: {args.output}")

    # These are global contract failures and intentionally remain fatal.
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    manifest_records = {
        int(record["global_ordinal"]): record for record in manifest.get("records", [])
    }

    page_counts: Counter[str] = Counter()
    method_counts: dict[str, Counter[str]] = {
        method: Counter() for method in detector_names()
    }
    method_elapsed_ms: dict[str, list[float]] = {
        method: [] for method in detector_names()
    }
    method_confidence: dict[str, list[float]] = {
        method: [] for method in detector_names()
    }

    for record in analysis.get("records", []):
        ordinal = int(record["global_ordinal"])
        manifest_record = manifest_records.get(ordinal, {"global_ordinal": ordinal})
        record.pop("geometry_candidate_error", None)

        try:
            image_path = path_for_record(manifest_record, args.image_root)
            candidates = run_detectors(image_path, args.max_dimension)
            record["geometry_candidates"] = [asdict(candidate) for candidate in candidates]
            record["geometry_candidate_status"] = _page_status(candidates)
            record["geometry_candidate_summary"] = summarize_candidates(candidates)

            for candidate in candidates:
                method_counts[candidate.method][candidate.status] += 1
                elapsed = candidate.diagnostics.get("elapsed_ms")
                if isinstance(elapsed, (int, float)):
                    method_elapsed_ms[candidate.method].append(float(elapsed))
                if candidate.status == "ok":
                    method_confidence[candidate.method].append(float(candidate.confidence))
        except Exception as exc:
            # A page input/preparation failure prevents every detector from
            # running, but it is still recorded and does not discard other pages.
            record["geometry_candidates"] = []
            record["geometry_candidate_status"] = "error"
            record["geometry_candidate_error"] = {
                "reason": "page_preparation_exception",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            }

        page_counts[record["geometry_candidate_status"]] += 1

    detector_error_count = sum(counts["error"] for counts in method_counts.values())
    page_error_count = page_counts["error"]
    summary = {
        # Keep the original keys for downstream compatibility.
        "processed": page_counts["complete"] + page_counts["partial"],
        "errors": page_error_count,
        "methods": detector_names(),
        # New resilience/observability fields.
        "page_status_counts": {
            "complete": page_counts["complete"],
            "partial": page_counts["partial"],
            "error": page_counts["error"],
        },
        "detector_error_count": detector_error_count,
        "detectors": detector_catalog(),
        "method_status_counts": {
            method: {
                "ok": counts["ok"],
                "no_candidate": counts["no_candidate"],
                "error": counts["error"],
            }
            for method, counts in method_counts.items()
        },
        "detector_performance": {
            method: {
                "runs": len(method_elapsed_ms[method]),
                "elapsed_ms_total": round(sum(method_elapsed_ms[method]), 3),
                "elapsed_ms_average": round(
                    sum(method_elapsed_ms[method]) / len(method_elapsed_ms[method]), 3
                ) if method_elapsed_ms[method] else None,
                "confidence_average": round(
                    sum(method_confidence[method]) / len(method_confidence[method]), 6
                ) if method_confidence[method] else None,
            }
            for method in detector_names()
        },
        "fail_on": args.fail_on,
    }
    analysis["geometry_candidate_summary"] = summary

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.fail_on == "detector-error" and detector_error_count:
        return 1
    if args.fail_on == "page-error" and page_error_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
