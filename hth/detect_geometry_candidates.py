#!/usr/bin/env python3
"""Generate registered physical-document geometry candidates for HTH pages."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import cv2

from geometry.common import document_mask, resize_for_analysis, scale_bbox
from geometry.registry import detector_names, run_registered_detectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dimension", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_detectors(image_path: Path, maximum: int) -> list[dict[str, Any]]:
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

    return [asdict(candidate) for candidate in candidates]


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


def main() -> int:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists; pass --overwrite: {args.output}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    manifest_records = {
        int(record["global_ordinal"]): record for record in manifest.get("records", [])
    }

    processed = 0
    errors = 0
    for record in analysis.get("records", []):
        ordinal = int(record["global_ordinal"])
        manifest_record = manifest_records.get(ordinal, {"global_ordinal": ordinal})
        try:
            image_path = path_for_record(manifest_record, args.image_root)
            record["geometry_candidates"] = run_detectors(image_path, args.max_dimension)
            record["geometry_candidate_status"] = "complete"
            processed += 1
        except Exception as exc:
            record["geometry_candidates"] = []
            record["geometry_candidate_status"] = "error"
            record["geometry_candidate_error"] = str(exc)
            errors += 1

    analysis["geometry_candidate_summary"] = {
        "processed": processed,
        "errors": errors,
        "methods": detector_names(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(analysis["geometry_candidate_summary"], indent=2, ensure_ascii=False))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
