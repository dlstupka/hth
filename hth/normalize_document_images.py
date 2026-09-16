#!/usr/bin/env python3
"""Create pixel-preserving axis-aligned document crops from canonical geometry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hth.canonical_build_evidence import (
    PREPROCESS_SCOPE,
    canonical_hash,
    load_evidence_store,
    validate_published_results,
)


SCHEMA_VERSION = "1.0"
POLICY_ID = "axis-aligned-detector-envelope-v1"


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


def _display_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _fit_panel(image: np.ndarray, width: int = 700, height: int = 620) -> np.ndarray:
    image = _display_image(image)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def _contact_sheet(original: np.ndarray, normalized: np.ndarray, bounds: tuple[int, int, int, int], ordinal: int) -> np.ndarray:
    overlay = _display_image(original.copy())
    left, top, right, bottom = bounds
    cv2.rectangle(overlay, (left, top), (right - 1, bottom - 1), (0, 0, 255), max(2, round(min(original.shape[:2]) / 500)))
    panels = []
    for image, label in ((overlay, "Canonical source + crop boundary"), (normalized, "Axis-aligned normalized image")):
        panel = _fit_panel(image)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (20, 20, 20), -1)
        cv2.putText(panel, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    sheet = np.concatenate(panels, axis=1)
    cv2.putText(sheet, f"GS0002 page {ordinal}", (12, sheet.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
    return sheet


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "global_ordinal", "source_sha256", "output_sha256", "output_pixel_sha256",
        "crop_left", "crop_top", "crop_right_exclusive", "crop_bottom_exclusive",
        "source_width", "source_height", "output_width", "output_height", "detector_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _summary(payload: dict[str, Any]) -> str:
    detector = payload["detector_selection"]
    return "\n".join([
        "# GS0002 Axis-Aligned Normalization",
        "",
        "> Artifact-only validation of the first production normalization policy. No detector inference, rotation, perspective warp, resizing, enhancement, or publication was performed.",
        "",
        "## Result",
        "",
        f"- Status: `{payload['status']}`",
        f"- Policy: `{payload['policy']['id']}`",
        f"- Pages normalized: `{payload['page_count']}`",
        f"- Detector evidence: `{detector.get('detector')}` / `{detector.get('parameter_set_id')}`",
        f"- Canonical preprocess build: `{payload['canonical_preprocess']['effective_build_identity']}`",
        f"- Canonical preprocess result: `{payload['canonical_preprocess']['canonical_result_identity']}`",
        f"- Normalization identity: `{payload['normalization_identity']}`",
        f"- Canonical normalization result: `{payload['canonical_result_identity']}`",
        "",
        "## Review artifact",
        "",
        "Open `index.html` after extracting the artifact. The red rectangle is the exact half-open crop boundary applied to the canonical source pixels.",
        "",
    ])


def _write_html(path: Path, payload: dict[str, Any]) -> None:
    cards = "".join(
        f'<article><h2>Page {ordinal}</h2><a href="contact-sheets/fs_{ordinal:04d}.jpg">'
        f'<img loading="lazy" src="contact-sheets/fs_{ordinal:04d}.jpg" alt="Normalization review for page {ordinal}"></a></article>'
        for ordinal in payload["ordinals"]
    )
    policy = html.escape(payload["policy"]["id"])
    path.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GS0002 Axis-Aligned Normalization</title><style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#111820;color:#e6edf3}}main{{max-width:1500px;margin:auto}}
article{{margin:2rem 0;padding:1rem;background:#18212b;border:1px solid #34404c;border-radius:.5rem}}img{{width:100%;height:auto}}
code{{background:#26313c;padding:.15rem .35rem;border-radius:.25rem}}
</style></head><body><main><h1>GS0002 Axis-Aligned Normalization</h1>
<p>Policy: <code>{policy}</code>. Pixels were cropped only; no geometric resampling was applied.</p>{cards}</main></body></html>""", encoding="utf-8")


def normalize(
    golden_set_path: Path,
    image_root: Path,
    manifest_path: Path,
    analysis_path: Path,
    preprocess_evidence: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    golden_set = _read_json(golden_set_path)
    manifest = _read_json(manifest_path)
    analysis = _read_json(analysis_path)
    selection = analysis.get("document_detector") or {}
    detector = str(selection.get("detector") or "").strip()
    if not detector:
        raise ValueError("Canonical page analysis has no resolved document_detector")

    manifest_by_ordinal = {int(item["global_ordinal"]): item for item in manifest.get("records") or []}
    analysis_by_ordinal = {int(item["global_ordinal"]): item for item in analysis.get("records") or []}
    evidence_pages = {
        int(item["global_ordinal"]): item
        for item in (preprocess_evidence.get("canonical_result") or {}).get("pages") or []
    }
    pages = [
        item for item in golden_set.get("pages") or []
        if item.get("review_status") == "approved" and item.get("image_sha256")
    ]
    if not pages:
        raise ValueError("Golden Set has no approved image records")

    normalized_root = output / "normalized"
    contacts_root = output / "contact-sheets"
    normalized_root.mkdir(parents=True, exist_ok=True)
    contacts_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    geometry_records: list[dict[str, Any]] = []

    for page in sorted(pages, key=lambda item: int(item["global_ordinal"])):
        ordinal = int(page["global_ordinal"])
        image_path = _find_image(image_root, ordinal)
        source_sha256 = _sha256(image_path)
        manifest_record = manifest_by_ordinal.get(ordinal) or {}
        evidence_page = evidence_pages.get(ordinal) or {}
        expected_hashes = {
            "Golden Set": str(page.get("image_sha256") or ""),
            "canonical image manifest": str(manifest_record.get("sha256") or ""),
            "Canonical Build Evidence": str(evidence_page.get("canonical_image_sha256") or ""),
        }
        for source, expected in expected_hashes.items():
            if expected != source_sha256:
                raise ValueError(f"Page {ordinal} {source} image SHA-256 does not match the materialized image")

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not decode canonical source image: {image_path}")
        analysis_record = analysis_by_ordinal.get(ordinal) or {}
        candidate = _candidate(analysis_record, detector)
        normalized, bounds = axis_aligned_crop(image, candidate["corners"])
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
        sheet = _contact_sheet(image, normalized, bounds, ordinal)
        if not cv2.imwrite(str(contacts_root / f"fs_{ordinal:04d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ValueError(f"Could not write normalization contact sheet for page {ordinal}")

    effective_identity = str(preprocess_evidence["effective_build_identity"])
    canonical_result_identity = str(preprocess_evidence["canonical_result"]["identity"])
    normalization_identity = canonical_hash({
        "schema_version": SCHEMA_VERSION,
        "policy": POLICY_ID,
        "golden_set_sha256": _sha256(golden_set_path),
        "canonical_preprocess_effective_build_identity": effective_identity,
        "canonical_preprocess_result_identity": canonical_result_identity,
        "detector": detector,
        "parameter_identity_sha256": selection.get("parameter_identity_sha256"),
    })
    result_identity = canonical_hash([
        {
            "global_ordinal": row["global_ordinal"],
            "source_sha256": row["source_sha256"],
            "output_pixel_sha256": row["output_pixel_sha256"],
            "crop": [row["crop_left"], row["crop_top"], row["crop_right_exclusive"], row["crop_bottom_exclusive"]],
        }
        for row in rows
    ])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "normalization_type": "document-crop",
        "status": "artifact-only",
        "policy": {
            "id": POLICY_ID,
            "operation": "axis-aligned crop of detector quadrilateral envelope",
            "resampling": "none",
            "output_format": "lossless PNG",
        },
        "golden_set": {"id": golden_set.get("collection_id"), "sha256": _sha256(golden_set_path)},
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
    (output / "normalization-manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "preprocess-evidence.json").write_text(json.dumps(preprocess_evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "geometry-evidence.json").write_text(json.dumps({"document_detector": selection, "records": geometry_records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(output / "normalization-manifest.csv", rows)
    (output / "summary.md").write_text(_summary(payload), encoding="utf-8")
    _write_html(output / "index.html", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-summary", type=Path)
    args = parser.parse_args(argv)
    evidence = load_authoritative_preprocess_evidence(
        args.results_root / "metadata/canonical-build-evidence.json",
        args.results_root,
    )
    payload = normalize(
        args.golden_set,
        args.image_root,
        args.results_root / "metadata/image_manifest.json",
        args.results_root / "analysis/page-analysis.json",
        evidence,
        args.output,
    )
    if args.github_summary:
        with args.github_summary.open("a", encoding="utf-8") as handle:
            handle.write(_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
