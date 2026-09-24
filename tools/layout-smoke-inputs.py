#!/usr/bin/env python3
"""Materialize verified Golden Set layout inputs.

This is a bounded research tool, not a layout result schema or a new cache.
It reuses the published normalization algorithms and fails if their pixels do
not match the pinned Results manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hth.assess_photometric_methods import apply_method
from hth.normalize_document_images import _pixel_sha256


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pages(payload: dict) -> dict[int, dict]:
    rows = payload.get("pages") or []
    if not isinstance(rows, list):
        raise ValueError("Manifest pages must be a list")
    result = {int(row["global_ordinal"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Manifest contains duplicate page ordinals")
    return result


def materialize(
    freeze_path: Path,
    images_root: Path,
    base_manifest_path: Path,
    photometric_manifest_path: Path,
    method_assessment_path: Path,
    final_manifest_path: Path,
    output: Path,
    results_commit: str,
    source_only: bool = False,
) -> dict:
    freeze = _load(freeze_path)
    base = _load(base_manifest_path) if not source_only else None
    photometric = _load(photometric_manifest_path) if not source_only else None
    methods = _load(method_assessment_path) if not source_only else None
    final = _load(final_manifest_path) if not source_only else None
    golden_set_path = ROOT / freeze["golden_set_path"]
    if not golden_set_path.is_file() or _sha256(golden_set_path) != freeze["golden_set_sha256"]:
        raise ValueError("Frozen Golden Set digest mismatch")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}")
    if not source_only and (len(results_commit) != 40 or any(c not in "0123456789abcdef" for c in results_commit)):
        raise ValueError("A full lowercase Results commit SHA is required")

    ordinals = [int(value) for value in freeze["membership"]["global_ordinals"]]
    if len(ordinals) != int(freeze["membership"]["page_count"]) or len(set(ordinals)) != len(ordinals):
        raise ValueError("Golden Set membership is inconsistent")
    source_images = {
        int(row["global_ordinal"]): row for row in freeze["image_bundle"]["images"]
    }
    if len(source_images) != len(freeze["image_bundle"]["images"]):
        raise ValueError("Image bundle contains duplicate page ordinals")
    if not ordinals or not all(n in source_images for n in ordinals):
        raise ValueError("Golden Set page is missing from the source bundle")
    if not source_only:
        base_pages, photometric_pages, final_pages = (
            _pages(payload) for payload in (base, photometric, final)
        )
        if base.get("status") != "complete" or photometric.get("status") != "complete" or final.get("status") != "complete":
            raise ValueError("All normalization manifests must be complete")
        if photometric.get("base_normalization_result_identity") != base.get("canonical_result_identity"):
            raise ValueError("Photometric manifest does not consume the selected base normalization")
        if methods.get("recommended_method_id") != (photometric.get("method") or {}).get("id"):
            raise ValueError("Photometric method selection differs from the applied method")
        if not all(n in base_pages and n in photometric_pages and n in final_pages for n in ordinals):
            raise ValueError("Golden Set page is missing from normalization evidence")

    source_output = output / "source"
    normalized_output = output / "normalized"
    source_output.mkdir(parents=True)
    if not source_only:
        normalized_output.mkdir()
    rows = []
    for ordinal in ordinals:
        source_record = source_images[ordinal]
        source = images_root / source_record["path"]
        if not source.is_file() or _sha256(source) != source_record["sha256"]:
            raise ValueError(f"Source image digest mismatch on page {ordinal}")
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not decode source page {ordinal}")
        source_target = source_output / f"fs_{ordinal:04d}.png"
        shutil.copyfile(source, source_target)
        row = {
            "global_ordinal": ordinal,
            "source_file": source_target.relative_to(output).as_posix(),
            "source_sha256": source_record["sha256"],
            "source_pixel_sha256": _pixel_sha256(image),
            "source_size": [int(image.shape[1]), int(image.shape[0])],
        }
        if source_only:
            rows.append(row)
            continue
        base_record = base_pages[ordinal]
        photometric_record = photometric_pages[ordinal]
        final_record = final_pages[ordinal]
        if source_record["sha256"] != base_record["source_sha256"]:
            raise ValueError(f"Source release differs from base normalization on page {ordinal}")
        if row["source_pixel_sha256"] != base_record["source_pixel_sha256"]:
            raise ValueError(f"Source pixels differ from base normalization on page {ordinal}")
        if base_record.get("transform_decision") != "preserve":
            raise ValueError(f"Page {ordinal} requires a geometric transform not materialized by this smoke")
        left = int(base_record["crop_left"])
        top = int(base_record["crop_top"])
        right = int(base_record["crop_right_exclusive"])
        bottom = int(base_record["crop_bottom_exclusive"])
        if not (0 <= left < right <= image.shape[1] and 0 <= top < bottom <= image.shape[0]):
            raise ValueError(f"Invalid crop on page {ordinal}")
        base_image = image[top:bottom, left:right].copy()
        base_hash = _pixel_sha256(base_image)
        if base_hash != base_record["output_pixel_sha256"]:
            raise ValueError(f"Canonical crop pixels differ on page {ordinal}")
        if photometric_record["input_pixel_sha256"] != base_hash:
            raise ValueError(f"Photometric input differs from canonical crop on page {ordinal}")
        route = photometric_record["route"]
        if route == "apply":
            normalized = apply_method(
                base_image,
                photometric["method"],
                methods["config"]["background_field"],
            )
        elif route == "preserve":
            normalized = base_image
        else:
            raise ValueError(f"Unsupported photometric route on page {ordinal}: {route}")
        final_hash = _pixel_sha256(normalized)
        if final_hash != photometric_record["output_pixel_sha256"] or final_hash != final_record["output_pixel_sha256"]:
            raise ValueError(f"Final normalized pixels differ from durable evidence on page {ordinal}")
        normalized_target = normalized_output / source_target.name
        if not cv2.imwrite(str(normalized_target), normalized, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise ValueError(f"Could not write final normalized page {ordinal}")
        rows.append({
            **row,
            "normalized_file": normalized_target.relative_to(output).as_posix(),
            "normalized_pixel_sha256": final_hash,
            "photometric_route": route,
            "normalized_size": [int(normalized.shape[1]), int(normalized.shape[0])],
        })
    result = {
        "schema_version": "1.0",
        "purpose": "layout-evaluation-inputs",
        "views": ["source"] if source_only else ["source", "normalized"],
        "golden_set_id": freeze["golden_set_id"],
        "golden_set_sha256": freeze["golden_set_sha256"],
        "source_release": freeze["source_release"],
        "results_commit": None if source_only else results_commit,
        "base_normalization_result_identity": None if source_only else base["canonical_result_identity"],
        "final_normalization_result_identity": None if source_only else final["binarization_result_identity"],
        "pages": rows,
    }
    (output / "paired-inputs.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--photometric-manifest", type=Path, required=True)
    parser.add_argument("--method-assessment", type=Path, required=True)
    parser.add_argument("--final-manifest", type=Path, required=True)
    parser.add_argument("--results-commit", required=True)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(
        args.freeze, args.images_root, args.base_manifest,
        args.photometric_manifest, args.method_assessment,
        args.final_manifest, args.output, args.results_commit, args.source_only,
    )
    print(f"Verified {len(result['pages'])} Golden Set images at {args.output}")


if __name__ == "__main__":
    main()
