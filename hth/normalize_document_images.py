#!/usr/bin/env python3
"""Create canonical document images from proven crop and transform policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hth.canonical_build_evidence import (
    NORMALIZATION_SCOPE,
    PREPROCESS_SCOPE,
    canonical_hash,
    load_evidence_store,
    validate_published_results,
)
from hth.preprocess import canonical_image, extension, ordered_images
from hth.orientation_deskew import evaluate_hough_policy
from hth.normalization_report import (
    render_summary,
    should_render_review,
    write_contact_sheet,
    write_reports,
)


SCHEMA_VERSION = "1.0"
POLICY_ID = "axis-aligned-detector-envelope-v1"


def _load_transform_policy(
    path: Path | None,
    preprocess_evidence: dict[str, Any],
    detector_selection: dict[str, Any],
    image_manifest: dict[str, Any],
    page_analysis: dict[str, Any],
    prior_normalization_manifest_path: Path | None,
) -> tuple[dict[str, Any] | None, dict[int, str]]:
    if path is None:
        return None, {}
    policy = _read_json(path)
    claimed_identity = str(policy.get("policy_identity") or "")
    identity_payload = dict(policy)
    identity_payload.pop("policy_identity", None)
    if len(claimed_identity) != 64 or canonical_hash(identity_payload) != claimed_identity:
        raise ValueError("Transform policy identity does not match its contents")
    if policy.get("policy_type") != "orientation-deskew-normalization":
        raise ValueError("Transform policy has an unsupported policy_type")
    if policy.get("policy_id") != "hough-lines-conservative-v1":
        raise ValueError("Transform policy is not a supported conservative Hough policy")
    if policy.get("status") != "recommended-for-validation":
        raise ValueError("Transform policy is not recommended for validation")
    if (policy.get("gross_orientation") or {}).get("action") != "preserve":
        raise ValueError("Automatic gross-orientation changes are not supported")
    deskew = policy.get("deskew") or {}
    if deskew.get("estimator") != "hough-lines":
        raise ValueError("Transform policy does not use the supported Hough estimator")
    if deskew.get("canvas") != "expanded-white" or deskew.get("interpolation") != "linear":
        raise ValueError("Transform policy has an unsupported resampling contract")
    compatibility = policy.get("compatibility") or {}
    current_preprocess = str((preprocess_evidence.get("canonical_result") or {}).get("identity") or "")
    expected = str(compatibility.get("canonical_preprocess_result_identity") or "")
    if compatibility.get("detector") != detector_selection.get("detector"):
        raise ValueError("Transform policy was assessed with a different document detector")
    if compatibility.get("parameter_identity_sha256") != detector_selection.get("parameter_identity_sha256"):
        raise ValueError("Transform policy was assessed with different detector parameters")
    if compatibility.get("base_normalization_policy_id") != POLICY_ID:
        raise ValueError("Transform policy was assessed against a different crop policy")
    if not expected:
        raise ValueError("Transform policy has no canonical preprocess result identity")
    if expected == current_preprocess:
        return policy, {}

    # A CBE result identity may change even when the exact input pixels to
    # conservative deskew have not. Only reuse the recommendation if its
    # published, CBE-verified normalization basis matches every current source
    # image and selected crop. The pixel digest is rechecked after cropping.
    if prior_normalization_manifest_path is None or not prior_normalization_manifest_path.is_file():
        raise ValueError("Transform policy was assessed against a different canonical preprocess result; no prior normalization basis is available")
    results_root = prior_normalization_manifest_path.parent.parent
    store = load_evidence_store(
        prior_normalization_manifest_path.parent / "canonical-build-evidence.json",
        scope=NORMALIZATION_SCOPE,
    )
    authoritative = str(store.get("authoritative_identity") or "")
    record = (store.get("records") or {}).get(authoritative)
    if not authoritative or not isinstance(record, dict):
        raise ValueError("Prior normalization basis has no authoritative Canonical Build Evidence")
    validate_published_results(record, results_root)
    prior = _read_json(prior_normalization_manifest_path)
    if (prior.get("canonical_preprocess") or {}).get("canonical_result_identity") != expected:
        raise ValueError("Prior normalization basis does not match the assessed preprocess result")
    if prior.get("canonical_result_identity") != (policy.get("evidence") or {}).get("canonical_normalization_result_identity"):
        raise ValueError("Prior normalization basis does not match the assessed normalization result")
    prior_detector = prior.get("detector_selection") or {}
    if (prior_detector.get("detector"), prior_detector.get("parameter_identity_sha256")) != (
        detector_selection.get("detector"), detector_selection.get("parameter_identity_sha256")
    ):
        raise ValueError("Prior normalization basis used different detector parameters")
    if (prior.get("policy") or {}).get("base_policy_id") != POLICY_ID:
        raise ValueError("Prior normalization basis used a different crop policy")

    prior_pages = prior.get("pages") or []
    image_pages = image_manifest.get("records") or []
    analysis_pages = page_analysis.get("records") or []
    prior_by_ordinal = {int(page["global_ordinal"]): page for page in prior_pages}
    analysis_by_ordinal = {int(page["global_ordinal"]): page for page in analysis_pages}
    image_ordinals = {int(page["global_ordinal"]) for page in image_pages}
    if (
        not prior_pages
        or len(prior_by_ordinal) != len(prior_pages)
        or len(analysis_by_ordinal) != len(analysis_pages)
        or len(image_ordinals) != len(image_pages)
        or set(prior_by_ordinal) != image_ordinals
        or set(analysis_by_ordinal) != image_ordinals
    ):
        raise ValueError("Prior normalization basis has a different or ambiguous page population")
    crop_hashes: dict[int, str] = {}
    for image_page in image_pages:
        ordinal = int(image_page["global_ordinal"])
        prior_page = prior_by_ordinal[ordinal]
        candidate = _candidate(analysis_by_ordinal[ordinal], str(detector_selection["detector"]))
        crop_hash = str(prior_page.get("base_crop_pixel_sha256") or "")
        if (
            prior_page.get("source_sha256") != image_page.get("sha256")
            or prior_page.get("detector_corners") != candidate.get("corners")
            or len(crop_hash) != 64
        ):
            raise ValueError(f"Prior normalization basis differs from current source or crop on page {ordinal}")
        crop_hashes[ordinal] = crop_hash
    return policy, crop_hashes


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pixel_sha256(image: np.ndarray) -> str:
    header = f"{image.dtype}:{','.join(str(value) for value in image.shape)}\n".encode("ascii")
    return hashlib.sha256(header + np.ascontiguousarray(image).tobytes()).hexdigest()


def _find_image(root: Path, ordinal: int) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"):
        for candidate in (root / f"fs_{ordinal:04d}{suffix}", root / "raw" / f"fs_{ordinal:04d}{suffix}"):
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No source image found for page {ordinal}")


def axis_aligned_bounds(corners: Any, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(points).all():
        raise ValueError("Document corners must be finite")
    height, width = image_shape[:2]
    left = max(0, int(math.floor(float(points[:, 0].min()))))
    top = max(0, int(math.floor(float(points[:, 1].min()))))
    right = min(width, int(math.ceil(float(points[:, 0].max()))))
    bottom = min(height, int(math.ceil(float(points[:, 1].max()))))
    if right <= left or bottom <= top:
        raise ValueError("Axis-aligned document crop is empty")
    return left, top, right, bottom


def axis_aligned_crop(image: np.ndarray, corners: Any) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    bounds = axis_aligned_bounds(corners, image.shape)
    left, top, right, bottom = bounds
    return image[top:bottom, left:right].copy(), bounds


def _candidate(record: dict[str, Any], detector: str) -> dict[str, Any]:
    candidates = record.get("geometry_candidates") or []
    match = next((item for item in candidates if item.get("method") == detector), None)
    if not isinstance(match, dict) or match.get("status") != "ok" or not match.get("corners"):
        raise ValueError(f"Page {record.get('global_ordinal')} has no successful {detector} geometry")
    return match


def load_authoritative_preprocess_evidence(store_path: Path, results_root: Path) -> dict[str, Any]:
    store = load_evidence_store(store_path, scope=PREPROCESS_SCOPE)
    identity = str(store.get("authoritative_identity") or "")
    if not identity:
        raise ValueError("Canonical preprocess evidence has no authoritative identity")
    evidence = (store.get("records") or {}).get(identity)
    if not isinstance(evidence, dict):
        raise ValueError("Canonical preprocess evidence does not contain its authoritative record")
    validate_published_results(evidence, results_root)
    return evidence


def materialize_canonical_images(source_root: Path, manifest_path: Path, output: Path) -> dict[str, Any]:
    """Reconstruct only canonical raw images and prove them against the published manifest."""
    manifest = _read_json(manifest_path)
    records = manifest.get("records") or []
    if not isinstance(records, list) or not records:
        raise ValueError("Canonical image manifest has no records")
    by_docx: dict[str, dict[int, dict[str, Any]]] = {}
    for record in records:
        source_docx = str(record.get("source_docx") or "")
        source_ordinal = int(record.get("source_ordinal") or 0)
        if not source_docx or source_ordinal <= 0:
            raise ValueError("Canonical image manifest contains an invalid source identity")
        if source_ordinal in by_docx.setdefault(source_docx, {}):
            raise ValueError(f"Canonical image manifest duplicates {source_docx} image {source_ordinal}")
        by_docx[source_docx][source_ordinal] = record

    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Canonical image materialization target is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    materialized: list[int] = []
    for source_docx, expected_by_ordinal in sorted(by_docx.items()):
        docx = source_root / source_docx
        if not docx.is_file():
            matches = list(source_root.rglob(source_docx))
            if len(matches) != 1:
                raise FileNotFoundError(f"Cannot uniquely resolve canonical source DOCX: {source_docx}")
            docx = matches[0]
        remaining = dict(expected_by_ordinal)
        for source_ordinal, (relationship_id, media_path, embedded_data, crop) in enumerate(ordered_images(docx), 1):
            record = remaining.pop(source_ordinal, None)
            if record is None:
                continue
            if str(record.get("relationship_id") or "") != relationship_id:
                raise ValueError(f"Page {record.get('global_ordinal')} relationship identity changed")
            if str(record.get("media_path") or "") != media_path:
                raise ValueError(f"Page {record.get('global_ordinal')} embedded media identity changed")
            if int(record.get("embedded_bytes") or 0) != len(embedded_data):
                raise ValueError(f"Page {record.get('global_ordinal')} embedded byte count changed")
            if str(record.get("embedded_sha256") or "") != hashlib.sha256(embedded_data).hexdigest():
                raise ValueError(f"Page {record.get('global_ordinal')} embedded image SHA-256 changed")
            expected_crop = tuple(int(record.get(f"word_crop_{side}") or 0) for side in ("left", "top", "right", "bottom"))
            if tuple(crop) != expected_crop:
                raise ValueError(f"Page {record.get('global_ordinal')} Word crop contract changed")
            data, detected_format, width, height, mode = canonical_image(embedded_data, crop)
            ordinal = int(record["global_ordinal"])
            if hashlib.sha256(data).hexdigest() != str(record.get("sha256") or ""):
                raise ValueError(f"Page {ordinal} reconstructed canonical image SHA-256 does not match")
            if (width, height, mode, detected_format) != (
                int(record.get("width_px") or 0),
                int(record.get("height_px") or 0),
                str(record.get("mode") or ""),
                str(record.get("detected_format") or ""),
            ):
                raise ValueError(f"Page {ordinal} reconstructed canonical image metadata does not match")
            target = output / f"fs_{ordinal:04d}{extension(detected_format)}"
            target.write_bytes(data)
            materialized.append(ordinal)
        if remaining:
            missing = ", ".join(str(value.get("global_ordinal")) for value in remaining.values())
            raise ValueError(f"Source DOCX {source_docx} is missing canonical pages: {missing}")
    ordinals = sorted(materialized)
    expected_ordinals = sorted(int(record["global_ordinal"]) for record in records)
    if ordinals != expected_ordinals:
        raise ValueError("Materialized canonical image population does not match the published manifest")
    return {
        "collection_id": manifest.get("collection_id"),
        "page_count": len(ordinals),
        "first_page": ordinals[0],
        "last_page": ordinals[-1],
    }


def normalize(
    golden_set_path: Path | None,
    image_root: Path,
    manifest_path: Path,
    analysis_path: Path,
    preprocess_evidence: dict[str, Any],
    output: Path,
    *,
    status: str = "artifact-only",
    contact_sheet_every: int = 1,
    transform_policy_path: Path | None = None,
    prior_normalization_manifest_path: Path | None = None,
) -> dict[str, Any]:
    golden_set = _read_json(golden_set_path) if golden_set_path else None
    manifest = _read_json(manifest_path)
    analysis = _read_json(analysis_path)
    selection = analysis.get("document_detector") or {}
    detector = str(selection.get("detector") or "").strip()
    if not detector:
        raise ValueError("Canonical page analysis has no resolved document_detector")
    transform_policy, prior_crop_hashes = _load_transform_policy(
        transform_policy_path,
        preprocess_evidence,
        selection,
        manifest,
        analysis,
        prior_normalization_manifest_path,
    )

    manifest_by_ordinal = {int(item["global_ordinal"]): item for item in manifest.get("records") or []}
    analysis_by_ordinal = {int(item["global_ordinal"]): item for item in analysis.get("records") or []}
    evidence_pages = {
        int(item["global_ordinal"]): item
        for item in (preprocess_evidence.get("canonical_result") or {}).get("pages") or []
    }
    if golden_set is not None:
        pages = [
            item for item in golden_set.get("pages") or []
            if item.get("review_status") == "approved" and item.get("image_sha256")
        ]
        target_type = "golden-set"
        collection_id = str(golden_set.get("collection_id") or "Golden Set")
    else:
        pages = list(manifest.get("records") or [])
        target_type = "collection"
        collection_id = str(manifest.get("collection_id") or "Collection")
    if not pages:
        raise ValueError("Normalization target has no image records")

    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Normalization output target is not empty: {output}")
    normalized_root = output / "normalized"
    contacts_root = output / "contact-sheets"
    normalized_root.mkdir(parents=True, exist_ok=True)
    contacts_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    geometry_records: list[dict[str, Any]] = []
    review_ordinals: list[int] = []

    ordered_pages = sorted(pages, key=lambda item: int(item["global_ordinal"]))
    for page_index, page in enumerate(ordered_pages):
        ordinal = int(page["global_ordinal"])
        image_path = _find_image(image_root, ordinal)
        source_sha256 = _sha256(image_path)
        manifest_record = manifest_by_ordinal.get(ordinal) or {}
        evidence_page = evidence_pages.get(ordinal) or {}
        expected_hashes = {
            "canonical image manifest": str(manifest_record.get("sha256") or ""),
            "Canonical Build Evidence": str(evidence_page.get("canonical_image_sha256") or ""),
        }
        if golden_set is not None:
            expected_hashes["Golden Set"] = str(page.get("image_sha256") or "")
        for source, expected in expected_hashes.items():
            if expected != source_sha256:
                raise ValueError(f"Page {ordinal} {source} image SHA-256 does not match the materialized image")

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not decode canonical source image: {image_path}")
        analysis_record = analysis_by_ordinal.get(ordinal) or {}
        candidate = _candidate(analysis_record, detector)
        base_crop, bounds = axis_aligned_crop(image, candidate["corners"])
        base_crop_hash = _pixel_sha256(base_crop)
        if prior_crop_hashes and prior_crop_hashes[ordinal] != base_crop_hash:
            raise ValueError(f"Prior normalization basis has different cropped pixels on page {ordinal}")
        if transform_policy is None:
            normalized = base_crop
            transformation = {
                "estimator": None,
                "decision": "preserve",
                "reason": "axis-aligned-only",
                "estimated_correction_degrees": 0.0,
                "applied_correction_degrees": 0.0,
            }
        else:
            normalized, transformation = evaluate_hough_policy(base_crop, transform_policy)
        target = normalized_root / f"fs_{ordinal:04d}.png"
        if not cv2.imwrite(str(target), normalized, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise ValueError(f"Could not write normalized image: {target}")
        round_trip = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
        if round_trip is None or not np.array_equal(round_trip, normalized):
            raise ValueError(f"Normalized image failed lossless pixel verification: {target}")

        left, top, right, bottom = bounds
        row = {
            "global_ordinal": ordinal,
            "source_file": image_path.name,
            "source_sha256": source_sha256,
            "source_pixel_sha256": _pixel_sha256(image),
            "output_file": target.relative_to(output).as_posix(),
            "output_sha256": _sha256(target),
            "output_pixel_sha256": _pixel_sha256(normalized),
            "base_crop_pixel_sha256": base_crop_hash,
            "base_crop_width": int(base_crop.shape[1]),
            "base_crop_height": int(base_crop.shape[0]),
            "transform_decision": transformation["decision"],
            "deskew_correction_degrees": transformation["applied_correction_degrees"],
            "transformation": transformation,
            "crop_left": left,
            "crop_top": top,
            "crop_right_exclusive": right,
            "crop_bottom_exclusive": bottom,
            "source_width": int(image.shape[1]),
            "source_height": int(image.shape[0]),
            "output_width": int(normalized.shape[1]),
            "output_height": int(normalized.shape[0]),
            "detector_confidence": candidate.get("confidence"),
            "detector_corners": candidate["corners"],
            "canonical_preprocess_page_result_sha256": evidence_page.get("canonical_page_result_sha256"),
        }
        rows.append(row)
        geometry_records.append({"global_ordinal": ordinal, "geometry_candidate": candidate})
        if should_render_review(
            page_index,
            len(ordered_pages),
            contact_sheet_every,
            str(transformation["decision"]),
        ):
            write_contact_sheet(
                contacts_root / f"fs_{ordinal:04d}.jpg",
                image,
                normalized,
                bounds,
                ordinal,
                collection_id,
            )
            review_ordinals.append(ordinal)

    effective_identity = str(preprocess_evidence["effective_build_identity"])
    canonical_result_identity = str(preprocess_evidence["canonical_result"]["identity"])
    normalization_identity = canonical_hash({
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY_ID,
        "target": {
            "type": target_type,
            "identity": _sha256(golden_set_path) if golden_set_path else canonical_hash(manifest),
        },
        "canonical_preprocess_effective_build_identity": effective_identity,
        "canonical_preprocess_result_identity": canonical_result_identity,
        "detector": detector,
        "parameter_identity_sha256": selection.get("parameter_identity_sha256"),
        "transform_policy_identity": transform_policy.get("policy_identity") if transform_policy else None,
    })
    result_identity = canonical_hash([
        {
            "global_ordinal": row["global_ordinal"],
            "source_sha256": row["source_sha256"],
            "output_pixel_sha256": row["output_pixel_sha256"],
            "crop": [row["crop_left"], row["crop_top"], row["crop_right_exclusive"], row["crop_bottom_exclusive"]],
            "transformation": row["transformation"],
        }
        for row in rows
    ])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "normalization_type": "document-crop",
        "status": status,
        "policy": {
            "id": (
                f"{POLICY_ID}+{transform_policy['policy_id']}" if transform_policy else POLICY_ID
            ),
            "base_policy_id": POLICY_ID,
            "transform_policy_id": transform_policy.get("policy_id") if transform_policy else None,
            "transform_policy_identity": transform_policy.get("policy_identity") if transform_policy else None,
            "operation": (
                "axis-aligned crop followed by gated conservative Hough deskew"
                if transform_policy else "axis-aligned crop of detector quadrilateral envelope"
            ),
            "resampling": "conditional linear rotation on expanded white canvas" if transform_policy else "none",
            "output_format": "lossless PNG",
        },
        "collection": {"id": collection_id},
        **({
            "golden_set": {"id": collection_id, "sha256": _sha256(golden_set_path)},
        } if golden_set_path else {}),
        "target": {
            "type": target_type,
            "sha256": _sha256(golden_set_path) if golden_set_path else canonical_hash(manifest),
        },
        "canonical_preprocess": {
            "effective_build_identity": effective_identity,
            "canonical_result_identity": canonical_result_identity,
            "activity": "REUSED",
            "domain_result": "SKIP",
        },
        "detector_selection": selection,
        "normalization_identity": normalization_identity,
        "canonical_result_identity": result_identity,
        "page_count": len(rows),
        "ordinals": [row["global_ordinal"] for row in rows],
        "pages": rows,
    }
    payload["transform_summary"] = {
        "policy_requested": transform_policy is not None,
        "pages_transformed": sum(row["transform_decision"] == "apply" for row in rows),
        "pages_preserved": sum(row["transform_decision"] != "apply" for row in rows),
    }
    if prior_crop_hashes:
        payload["transform_policy_compatibility"] = {
            "mode": "verified-identical-base-crops",
            "assessed_preprocess_result_identity": (transform_policy.get("compatibility") or {}).get("canonical_preprocess_result_identity"),
            "current_preprocess_result_identity": canonical_result_identity,
            "pages_verified": len(prior_crop_hashes),
        }
    (output / "normalization-manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "preprocess-evidence.json").write_text(json.dumps(preprocess_evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "geometry-evidence.json").write_text(json.dumps({"document_detector": selection, "records": geometry_records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if transform_policy is not None:
        (output / "normalization-policy.json").write_text(
            json.dumps(transform_policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    write_reports(output, payload, rows, review_ordinals, contact_sheet_every)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-root", type=Path)
    source.add_argument("--source-root", type=Path, help="Immutable source DOCX directory to reconstruct canonical images")
    parser.add_argument("--materialized-image-root", type=Path)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-summary", type=Path)
    parser.add_argument("--status", default="artifact-only")
    parser.add_argument("--contact-sheet-every", type=int, default=1)
    parser.add_argument(
        "--transform-policy",
        type=Path,
        help="Machine-readable recommendation produced by the normalization assessment flow",
    )
    args = parser.parse_args(argv)
    if args.contact_sheet_every < 0:
        parser.error("--contact-sheet-every must be zero or positive")
    evidence = load_authoritative_preprocess_evidence(
        args.results_root / "metadata/canonical-build-evidence.json",
        args.results_root,
    )
    image_root = args.image_root
    if args.source_root:
        if args.materialized_image_root is None:
            parser.error("--materialized-image-root is required with --source-root")
        materialize_canonical_images(
            args.source_root,
            args.results_root / "metadata/image_manifest.json",
            args.materialized_image_root,
        )
        image_root = args.materialized_image_root
    assert image_root is not None
    payload = normalize(
        args.golden_set,
        image_root,
        args.results_root / "metadata/image_manifest.json",
        args.results_root / "analysis/page-analysis.json",
        evidence,
        args.output,
        status=args.status,
        contact_sheet_every=args.contact_sheet_every,
        transform_policy_path=args.transform_policy,
        prior_normalization_manifest_path=args.results_root / "normalization/normalization-manifest.json",
    )
    if args.github_summary:
        with args.github_summary.open("a", encoding="utf-8") as handle:
            handle.write(render_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
