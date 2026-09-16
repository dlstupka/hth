#!/usr/bin/env python3
"""Assess gross orientation and conservative deskew on canonical document crops."""

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

from hth.canonical_build_evidence import canonical_hash
from hth.normalize_document_images import (
    _find_image,
    _pixel_sha256,
    _sha256,
    materialize_canonical_images,
)
from hth.orientation_deskew import estimate_hough_lines, rotate_expand


SCHEMA_VERSION = "2.0"
ESTIMATORS = ("projection-profile", "hough-lines")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _require_current_schema(payload: dict[str, Any], label: str) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{label} must be regenerated with schema {SCHEMA_VERSION}; "
            "development artifacts are not migrated or accepted as production evidence"
        )


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text.lower()):
        raise ValueError(f"{label} must be a 64-character SHA-256 identity")
    return text


def _normalization_scope(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return displayable provenance for collection and Golden Set manifests."""
    scope = manifest.get("collection") or manifest.get("golden_set")
    if isinstance(scope, dict) and scope.get("id"):
        return scope
    return {"id": "unknown"}


def _approved_golden_ordinals(path: Path | None) -> list[int]:
    if path is None:
        return []
    payload = _read_json(path)
    return sorted({
        int(page["global_ordinal"])
        for page in payload.get("pages") or []
        if page.get("review_status") == "approved"
    })


def _area_retention(page: dict[str, Any]) -> float:
    source_area = int(page["source_width"]) * int(page["source_height"])
    output_area = int(page["output_width"]) * int(page["output_height"])
    return output_area / source_area if source_area else 0.0


def _crop_asymmetry(page: dict[str, Any]) -> float:
    left = int(page["crop_left"])
    top = int(page["crop_top"])
    right = int(page["source_width"]) - int(page["crop_right_exclusive"])
    bottom = int(page["source_height"]) - int(page["crop_bottom_exclusive"])
    width = max(1, int(page["source_width"]))
    height = max(1, int(page["source_height"]))
    return abs(left - right) / width + abs(top - bottom) / height


def select_sample(
    normalization_manifest: dict[str, Any],
    config: dict[str, Any],
    golden_ordinals: list[int] | None = None,
) -> dict[str, Any]:
    pages = normalization_manifest.get("pages") or []
    if not isinstance(pages, list) or not pages:
        raise ValueError("Normalization manifest has no page records")
    by_ordinal = {int(page["global_ordinal"]): page for page in pages}
    if len(by_ordinal) != len(pages):
        raise ValueError("Normalization manifest contains duplicate page ordinals")
    sampling = config.get("sampling") or {}
    cadence = max(1, int(sampling.get("cadence") or 25))
    reasons: dict[int, set[str]] = {}

    def include(ordinal: int, reason: str) -> None:
        if ordinal in by_ordinal:
            reasons.setdefault(ordinal, set()).add(reason)

    ordered = sorted(by_ordinal)
    include(ordered[0], "first-page")
    include(ordered[-1], "last-page")
    for index, ordinal in enumerate(ordered):
        if index % cadence == 0:
            include(ordinal, f"cadence-{cadence}")
    for ordinal in golden_ordinals or []:
        include(int(ordinal), "golden-set")

    confidence_count = max(0, int(sampling.get("lowest_detector_confidence") or 0))
    confidence_pages = sorted(
        pages,
        key=lambda page: (
            float(page.get("detector_confidence")) if page.get("detector_confidence") is not None else -1.0,
            int(page["global_ordinal"]),
        ),
    )[:confidence_count]
    for page in confidence_pages:
        include(int(page["global_ordinal"]), "low-detector-confidence")

    retention_count = max(0, int(sampling.get("lowest_area_retention") or 0))
    for page in sorted(pages, key=lambda item: (_area_retention(item), int(item["global_ordinal"])))[:retention_count]:
        include(int(page["global_ordinal"]), "low-area-retention")

    asymmetry_count = max(0, int(sampling.get("largest_crop_asymmetry") or 0))
    for page in sorted(pages, key=lambda item: (-_crop_asymmetry(item), int(item["global_ordinal"])))[:asymmetry_count]:
        include(int(page["global_ordinal"]), "crop-asymmetry")

    selected = [
        {
            "global_ordinal": ordinal,
            "reasons": sorted(reasons[ordinal]),
            "detector_confidence": by_ordinal[ordinal].get("detector_confidence"),
            "area_retention": _area_retention(by_ordinal[ordinal]),
            "crop_asymmetry": _crop_asymmetry(by_ordinal[ordinal]),
        }
        for ordinal in sorted(reasons)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "assessment_type": "orientation-deskew-sample-plan",
        "normalization_identity": normalization_manifest.get("normalization_identity"),
        "canonical_normalization_result_identity": normalization_manifest.get("canonical_result_identity"),
        "collection": _normalization_scope(normalization_manifest),
        "population_page_count": len(pages),
        "sample_page_count": len(selected),
        "sampling": sampling,
        "pages": selected,
        "sample_identity": canonical_hash(selected),
    }


def prepare_sample(
    normalization_manifest_path: Path,
    config_path: Path,
    output_path: Path,
    golden_set_path: Path | None = None,
) -> dict[str, Any]:
    manifest = _read_json(normalization_manifest_path)
    config = _read_json(config_path)
    payload = select_sample(manifest, config, _approved_golden_ordinals(golden_set_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def materialize_sample(
    source_root: Path,
    canonical_manifest_path: Path,
    normalization_manifest_path: Path,
    sample_plan_path: Path,
    output: Path,
) -> dict[str, Any]:
    canonical_manifest = _read_json(canonical_manifest_path)
    normalization_manifest = _read_json(normalization_manifest_path)
    sample_plan = _read_json(sample_plan_path)
    ordinals = sorted({int(page["global_ordinal"]) for page in sample_plan.get("pages") or []})
    if not ordinals:
        raise ValueError("Orientation/deskew sample plan is empty")
    wanted = set(ordinals)
    source_records = [
        record for record in canonical_manifest.get("records") or []
        if int(record["global_ordinal"]) in wanted
    ]
    normalized_by_ordinal = {
        int(page["global_ordinal"]): page
        for page in normalization_manifest.get("pages") or []
        if int(page["global_ordinal"]) in wanted
    }
    if len(source_records) != len(ordinals) or len(normalized_by_ordinal) != len(ordinals):
        raise ValueError("Sample ordinals are not fully represented in the canonical manifests")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Orientation/deskew sample output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    canonical_root = output / "canonical-source"
    sample_manifest = output / "sample-canonical-image-manifest.json"
    sample_manifest.write_text(json.dumps({
        **{key: value for key, value in canonical_manifest.items() if key != "records"},
        "records": source_records,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    materialize_canonical_images(source_root, sample_manifest, canonical_root)

    normalized_root = output / "normalized"
    normalized_root.mkdir()
    rows: list[dict[str, Any]] = []
    for ordinal in ordinals:
        record = normalized_by_ordinal[ordinal]
        source = _find_image(canonical_root, ordinal)
        if _sha256(source) != str(record.get("source_sha256") or ""):
            raise ValueError(f"Page {ordinal} reconstructed source does not match normalization evidence")
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not decode reconstructed page {ordinal}")
        if (image.shape[1], image.shape[0]) != (int(record["source_width"]), int(record["source_height"])):
            raise ValueError(f"Page {ordinal} reconstructed dimensions do not match normalization evidence")
        left = int(record["crop_left"])
        top = int(record["crop_top"])
        right = int(record["crop_right_exclusive"])
        bottom = int(record["crop_bottom_exclusive"])
        cropped = image[top:bottom, left:right].copy()
        if (cropped.shape[1], cropped.shape[0]) != (int(record["output_width"]), int(record["output_height"])):
            raise ValueError(f"Page {ordinal} crop dimensions do not match normalization evidence")
        if _pixel_sha256(cropped) != str(record.get("output_pixel_sha256") or ""):
            raise ValueError(f"Page {ordinal} crop pixels do not match the canonical normalization result")
        target = normalized_root / f"fs_{ordinal:04d}.png"
        if not cv2.imwrite(str(target), cropped, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise ValueError(f"Could not write sampled normalized page {ordinal}")
        round_trip = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
        if round_trip is None or not np.array_equal(round_trip, cropped):
            raise ValueError(f"Sampled normalized page {ordinal} failed lossless round-trip validation")
        rows.append({
            "global_ordinal": ordinal,
            "source_sha256": record["source_sha256"],
            "canonical_normalized_pixel_sha256": record["output_pixel_sha256"],
            "materialized_pixel_sha256": _pixel_sha256(cropped),
            "output_file": target.relative_to(output).as_posix(),
        })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "verified",
        "sample_identity": sample_plan.get("sample_identity"),
        "canonical_normalization_result_identity": normalization_manifest.get("canonical_result_identity"),
        "page_count": len(rows),
        "pages": rows,
    }
    (output / "materialization-evidence.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _display(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _analysis_gray(image: np.ndarray, maximum_dimension: int = 1400) -> np.ndarray:
    gray = _gray(image)
    scale = min(1.0, maximum_dimension / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray


def _foreground(gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    if float(np.median(blurred)) >= 127.0:
        return cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def _rotate_same(image: np.ndarray, angle: float, interpolation: int, border_value: int = 0) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def _projection_score(mask: np.ndarray, angle: float) -> float:
    rotated = _rotate_same(mask, angle, cv2.INTER_NEAREST, 0)
    margin_y = max(1, rotated.shape[0] // 50)
    margin_x = max(1, rotated.shape[1] // 50)
    central = rotated[margin_y:-margin_y, margin_x:-margin_x]
    profile = np.count_nonzero(central, axis=1).astype(np.float64)
    return float(np.var(profile) / (np.mean(profile) + 1.0))


def estimate_projection_profile(
    image: np.ndarray,
    maximum_degrees: float,
    deadband_degrees: float,
    coarse_step: float,
    fine_step: float,
) -> dict[str, Any]:
    mask = _foreground(_analysis_gray(image))
    coarse_angles = np.arange(-maximum_degrees, maximum_degrees + coarse_step / 2.0, coarse_step)
    coarse_scores = [(float(angle), _projection_score(mask, float(angle))) for angle in coarse_angles]
    coarse_best = max(coarse_scores, key=lambda item: (item[1], -abs(item[0])))
    fine_start = max(-maximum_degrees, coarse_best[0] - coarse_step)
    fine_end = min(maximum_degrees, coarse_best[0] + coarse_step)
    fine_angles = np.arange(fine_start, fine_end + fine_step / 2.0, fine_step)
    scored = [(float(angle), _projection_score(mask, float(angle))) for angle in fine_angles]
    best_angle, best_score = max(scored, key=lambda item: (item[1], -abs(item[0])))
    zero_score = _projection_score(mask, 0.0)
    alternatives = [score for angle, score in scored if abs(angle - best_angle) >= max(0.15, fine_step * 2)]
    second_score = max(alternatives) if alternatives else zero_score
    improvement = max(0.0, (best_score - zero_score) / max(best_score, 1e-9))
    peak = max(0.0, (best_score - second_score) / max(best_score, 1e-9))
    confidence = min(1.0, math.sqrt(improvement * max(peak, 1e-9)) * 5.0)
    boundary_limited = abs(best_angle) >= maximum_degrees - fine_step / 2.0
    applied = 0.0 if abs(best_angle) < deadband_degrees else best_angle
    return {
        "estimated_correction_degrees": round(best_angle, 6),
        "applied_correction_degrees": round(applied, 6),
        "confidence": round(confidence, 6),
        "zero_score": round(zero_score, 6),
        "best_score": round(best_score, 6),
        "relative_improvement": round(improvement, 6),
        "boundary_limited": boundary_limited,
    }


def orientation_axis_scores(image: np.ndarray) -> dict[str, Any]:
    mask = _foreground(_analysis_gray(image, 900))
    raw: dict[str, float] = {}
    for degrees in (0, 90, 180, 270):
        candidate = np.rot90(mask, degrees // 90)
        profile = np.count_nonzero(candidate, axis=1).astype(np.float64)
        raw[str(degrees)] = float(np.var(profile) / (np.mean(profile) + 1.0))
    total = sum(raw.values()) or 1.0
    scores = {key: value / total for key, value in raw.items()}
    horizontal = scores["0"] + scores["180"]
    vertical = scores["90"] + scores["270"]
    axis = "horizontal-text-axis" if horizontal >= vertical else "vertical-text-axis"
    confidence = abs(horizontal - vertical) / max(horizontal + vertical, 1e-9)
    return {
        "scores": {key: round(value, 6) for key, value in scores.items()},
        "preferred_axis": axis,
        "axis_confidence": round(confidence, 6),
        "upright_vs_upside_down": "indeterminate-without-semantic-evidence",
    }


def _sharpness(image: np.ndarray) -> float:
    return float(cv2.Laplacian(_gray(image), cv2.CV_64F).var())


def _candidate_metrics(source: np.ndarray, output: np.ndarray, angle: float) -> dict[str, Any]:
    source_mask = _foreground(_analysis_gray(source, 900))
    output_mask = _foreground(_analysis_gray(output, 900))
    source_foreground = max(1, int(np.count_nonzero(source_mask)))
    output_foreground = int(np.count_nonzero(output_mask))
    band = max(1, min(output_mask.shape) // 100)
    border = np.zeros_like(output_mask, dtype=bool)
    border[:band, :] = True
    border[-band:, :] = True
    border[:, :band] = True
    border[:, -band:] = True
    foreground = output_mask > 0
    return {
        "applied_correction_degrees": round(float(angle), 6),
        "output_width": int(output.shape[1]),
        "output_height": int(output.shape[0]),
        "expanded_area_ratio": round((output.shape[0] * output.shape[1]) / (source.shape[0] * source.shape[1]), 8),
        "foreground_retention_proxy": round(output_foreground / source_foreground, 8),
        "foreground_border_fraction": round(float(np.count_nonzero(foreground & border)) / max(1, output_foreground), 8),
        "sharpness_ratio": round(_sharpness(output) / max(_sharpness(source), 1e-9), 8),
    }


def _fit_panel(image: np.ndarray, width: int, height: int, label: str) -> np.ndarray:
    image = _display(image)
    header = 42
    usable_height = height - header
    scale = min(width / image.shape[1], usable_height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = header + (usable_height - resized.shape[0]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.rectangle(panel, (0, 0), (width, header), (20, 20, 20), -1)
    cv2.putText(panel, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _contact_sheet(
    image: np.ndarray,
    projection: np.ndarray,
    hough: np.ndarray,
    projection_result: dict[str, Any],
    hough_result: dict[str, Any],
    ordinal: int,
    width: int,
    height: int,
) -> np.ndarray:
    candidates = [
        (image, "Canonical crop: 0 deg"),
        (cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), "Orientation candidate: 90 CW"),
        (cv2.rotate(image, cv2.ROTATE_180), "Orientation candidate: 180"),
        (cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), "Orientation candidate: 270 CW"),
        (projection, f"Projection correction: {projection_result['applied_correction_degrees']:+.2f} deg"),
        (hough, f"Hough correction: {hough_result['applied_correction_degrees']:+.2f} deg"),
    ]
    panels = [_fit_panel(candidate, width, height, label) for candidate, label in candidates]
    rows = [np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)]
    sheet = np.concatenate(rows, axis=0)
    cv2.putText(sheet, f"HTH-0001 page {ordinal}", (12, sheet.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
    return sheet


def _write_csv(path: Path, pages: list[dict[str, Any]]) -> None:
    fields = [
        "global_ordinal", "sample_reasons", "preferred_axis", "axis_confidence",
        "projection_angle", "projection_confidence", "hough_angle", "hough_confidence",
        "estimator_delta_degrees", "estimator_status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for page in pages:
            writer.writerow({
                "global_ordinal": page["global_ordinal"],
                "sample_reasons": ";".join(page["sample_reasons"]),
                "preferred_axis": page["orientation"]["preferred_axis"],
                "axis_confidence": page["orientation"]["axis_confidence"],
                "projection_angle": page["estimators"]["projection-profile"]["applied_correction_degrees"],
                "projection_confidence": page["estimators"]["projection-profile"]["confidence"],
                "hough_angle": page["estimators"]["hough-lines"]["applied_correction_degrees"],
                "hough_confidence": page["estimators"]["hough-lines"]["confidence"],
                "estimator_delta_degrees": page["estimator_delta_degrees"],
                "estimator_status": page["estimator_status"],
            })


def _summary(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# HTH Orientation / Deskew Assessment",
        "",
        "> Diagnostic evidence only. Gross orientation has no semantic 0-vs-180 truth, and deskew resamples pixels. No production normalization policy is selected or published by this run.",
        "",
        "## Provenance",
        "",
        f"- Collection: `{payload['collection']['id']}`",
        f"- Canonical normalization result: `{payload['canonical_normalization_result_identity']}`",
        f"- Population: `{payload['population_page_count']}` pages",
        f"- Stratified sample: `{payload['sample_page_count']}` pages",
        f"- Sample identity: `{payload['sample_identity']}`",
        "",
        "## Aggregate comparison",
        "",
        f"- Horizontal text-axis preference: `{aggregate['horizontal_axis_pages']}` pages",
        f"- Vertical text-axis preference: `{aggregate['vertical_axis_pages']}` pages",
        f"- Projection corrections outside deadband: `{aggregate['projection_applied_pages']}` pages",
        f"- Hough corrections outside deadband: `{aggregate['hough_applied_pages']}` pages",
        f"- Estimators agree within tolerance: `{aggregate['agreement_pages']}` pages",
        f"- Estimator conflicts requiring review: `{aggregate['conflict_pages']}` pages",
        "",
        "| Estimator | Mean absolute correction | Maximum absolute correction | Mean confidence |",
        "|---|---:|---:|---:|",
    ]
    for estimator in ESTIMATORS:
        item = aggregate[estimator]
        lines.append(
            f"| {estimator} | {item['mean_absolute_correction_degrees']:.3f} deg | "
            f"{item['maximum_absolute_correction_degrees']:.3f} deg | {item['mean_confidence']:.3f} |"
        )
    lines.extend([
        "",
        "## Highest-priority visual review",
        "",
        "| Page | Projection | Hough | Delta | Status |",
        "|---:|---:|---:|---:|---|",
    ])
    for page in payload["priority_review_pages"]:
        lines.append(
            f"| {page['global_ordinal']} | {page['projection_angle']:+.3f} deg | "
            f"{page['hough_angle']:+.3f} deg | {page['delta']:.3f} deg | {page['status']} |"
        )
    lines.extend([
        "",
        "## Review artifact",
        "",
        "Open `index.html`. Each contact sheet contains the canonical crop, all four gross-orientation candidates, and both conservative deskew candidates.",
        "",
    ])
    return "\n".join(lines)


def _write_html(path: Path, payload: dict[str, Any]) -> None:
    cards = "".join(
        f'<article><h2>Page {page["global_ordinal"]}</h2><p>{html.escape(", ".join(page["sample_reasons"]))}</p>'
        f'<a href="contact-sheets/fs_{page["global_ordinal"]:04d}.jpg"><img loading="lazy" '
        f'src="contact-sheets/fs_{page["global_ordinal"]:04d}.jpg" alt="Orientation and deskew comparison for page {page["global_ordinal"]}"></a></article>'
        for page in payload["pages"]
    )
    path.write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HTH Orientation / Deskew Assessment</title><style>
body{{font-family:system-ui,sans-serif;margin:2rem;background:#111820;color:#e6edf3}}main{{max-width:1700px;margin:auto}}
article{{margin:2rem 0;padding:1rem;background:#18212b;border:1px solid #34404c;border-radius:.5rem}}img{{width:100%;height:auto}}
code{{background:#26313c;padding:.15rem .35rem;border-radius:.25rem}}.warning{{padding:1rem;border-left:4px solid #d29922;background:#2b2415}}
</style></head><body><main><h1>HTH Orientation / Deskew Assessment</h1>
<p class="warning">Diagnostic evidence only. No orientation or deskew transform has been approved for production.</p>
<p>Canonical normalization result: <code>{html.escape(str(payload['canonical_normalization_result_identity']))}</code></p>
{cards}</main></body></html>""", encoding="utf-8")


def assess(
    image_root: Path,
    normalization_manifest_path: Path,
    sample_plan_path: Path,
    config_path: Path,
    output: Path,
) -> dict[str, Any]:
    manifest = _read_json(normalization_manifest_path)
    sample = _read_json(sample_plan_path)
    config = _read_json(config_path)
    if sample.get("canonical_normalization_result_identity") != manifest.get("canonical_result_identity"):
        raise ValueError("Sample plan does not match the canonical normalization result")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Orientation/deskew assessment output is not empty: {output}")
    contacts = output / "contact-sheets"
    contacts.mkdir(parents=True, exist_ok=True)
    deskew = config.get("deskew") or {}
    review = config.get("review") or {}
    maximum = float(deskew.get("maximum_correction_degrees") or 3.0)
    deadband = float(deskew.get("deadband_degrees") or 0.15)
    coarse_step = float(deskew.get("projection_coarse_step_degrees") or 0.25)
    fine_step = float(deskew.get("projection_fine_step_degrees") or 0.05)
    agreement_tolerance = float(deskew.get("agreement_tolerance_degrees") or 0.25)
    conflict_threshold = float(deskew.get("conflict_threshold_degrees") or 0.75)
    panel_width = int(review.get("panel_width") or 520)
    panel_height = int(review.get("panel_height") or 420)
    jpeg_quality = int(review.get("jpeg_quality") or 92)
    reasons_by_ordinal = {
        int(page["global_ordinal"]): list(page.get("reasons") or [])
        for page in sample.get("pages") or []
    }
    rows: list[dict[str, Any]] = []
    for ordinal in sorted(reasons_by_ordinal):
        source = _find_image(image_root, ordinal)
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not decode canonical normalized sample page {ordinal}")
        orientation = orientation_axis_scores(image)
        projection = estimate_projection_profile(image, maximum, deadband, coarse_step, fine_step)
        hough = estimate_hough_lines(image, maximum, deadband)
        projection_image = rotate_expand(image, float(projection["applied_correction_degrees"]))
        hough_image = rotate_expand(image, float(hough["applied_correction_degrees"]))
        projection["output_metrics"] = _candidate_metrics(image, projection_image, float(projection["applied_correction_degrees"]))
        hough["output_metrics"] = _candidate_metrics(image, hough_image, float(hough["applied_correction_degrees"]))
        delta = abs(float(projection["applied_correction_degrees"]) - float(hough["applied_correction_degrees"]))
        status = "agree" if delta <= agreement_tolerance else "conflict" if delta >= conflict_threshold else "review"
        sheet = _contact_sheet(
            image,
            projection_image,
            hough_image,
            projection,
            hough,
            ordinal,
            panel_width,
            panel_height,
        )
        if not cv2.imwrite(str(contacts / f"fs_{ordinal:04d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
            raise ValueError(f"Could not write orientation/deskew contact sheet for page {ordinal}")
        rows.append({
            "global_ordinal": ordinal,
            "sample_reasons": reasons_by_ordinal[ordinal],
            "source_file": source.name,
            "source_sha256": _sha256(source),
            "source_pixel_sha256": _pixel_sha256(image),
            "orientation": orientation,
            "estimators": {"projection-profile": projection, "hough-lines": hough},
            "estimator_delta_degrees": round(delta, 6),
            "estimator_status": status,
        })

    aggregate: dict[str, Any] = {
        "horizontal_axis_pages": sum(page["orientation"]["preferred_axis"] == "horizontal-text-axis" for page in rows),
        "vertical_axis_pages": sum(page["orientation"]["preferred_axis"] == "vertical-text-axis" for page in rows),
        "projection_applied_pages": sum(abs(page["estimators"]["projection-profile"]["applied_correction_degrees"]) > 0 for page in rows),
        "hough_applied_pages": sum(abs(page["estimators"]["hough-lines"]["applied_correction_degrees"]) > 0 for page in rows),
        "agreement_pages": sum(page["estimator_status"] == "agree" for page in rows),
        "conflict_pages": sum(page["estimator_status"] == "conflict" for page in rows),
        "review_pages": sum(page["estimator_status"] == "review" for page in rows),
    }
    for estimator in ESTIMATORS:
        values = [abs(float(page["estimators"][estimator]["applied_correction_degrees"])) for page in rows]
        confidences = [float(page["estimators"][estimator]["confidence"]) for page in rows]
        aggregate[estimator] = {
            "mean_absolute_correction_degrees": sum(values) / len(values),
            "maximum_absolute_correction_degrees": max(values),
            "mean_confidence": sum(confidences) / len(confidences),
            "boundary_limited_pages": sum(bool(page["estimators"][estimator]["boundary_limited"]) for page in rows),
        }
    priority = sorted(
        rows,
        key=lambda page: (
            page["estimator_status"] != "conflict",
            -float(page["estimator_delta_degrees"]),
            -max(abs(float(page["estimators"][name]["applied_correction_degrees"])) for name in ESTIMATORS),
        ),
    )[:20]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "assessment_type": "orientation-deskew-comparison",
        "status": "diagnostic-only",
        "decision": "manual-review-required",
        "truth_limitations": [
            "No independently approved 0-vs-180 semantic orientation truth is available.",
            "Deskew estimators are image-derived associations, not geometric ground truth.",
            "Any non-zero deskew resamples pixels and requires explicit production approval.",
        ],
        "collection": _normalization_scope(manifest),
        "base_normalization_policy": manifest.get("policy"),
        "canonical_preprocess": manifest.get("canonical_preprocess"),
        "detector_selection": manifest.get("detector_selection"),
        "canonical_normalization_result_identity": manifest.get("canonical_result_identity"),
        "population_page_count": sample.get("population_page_count"),
        "sample_page_count": len(rows),
        "sample_identity": sample.get("sample_identity"),
        "config": config,
        "aggregate": aggregate,
        "priority_review_pages": [
            {
                "global_ordinal": page["global_ordinal"],
                "projection_angle": page["estimators"]["projection-profile"]["applied_correction_degrees"],
                "hough_angle": page["estimators"]["hough-lines"]["applied_correction_degrees"],
                "delta": page["estimator_delta_degrees"],
                "status": page["estimator_status"],
            }
            for page in priority
        ],
        "pages": rows,
    }
    payload["assessment_identity"] = canonical_hash({
        "assessment_type": payload["assessment_type"],
        "canonical_normalization_result_identity": payload["canonical_normalization_result_identity"],
        "sample_identity": payload["sample_identity"],
        "config": payload["config"],
        "pages": payload["pages"],
    })
    (output / "assessment.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(output / "assessment.csv", rows)
    summary = _summary(payload)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    _write_html(output / "index.html", payload)
    return payload


def recommend_policy(
    assessment_path: Path,
    output_path: Path,
    github_summary: Path | None = None,
) -> dict[str, Any]:
    """Convert diagnostic evidence into a bounded, explicitly activated policy candidate."""
    assessment = _read_json(assessment_path)
    _require_current_schema(assessment, "Orientation/deskew assessment")
    if assessment.get("assessment_type") != "orientation-deskew-comparison":
        raise ValueError("Recommendation input is not an orientation/deskew assessment")
    if assessment.get("status") != "diagnostic-only":
        raise ValueError("Recommendation input is not completed diagnostic evidence")
    assessment_identity = _require_sha256(assessment.get("assessment_identity"), "Assessment identity")
    expected_assessment_identity = canonical_hash({
        "assessment_type": assessment["assessment_type"],
        "canonical_normalization_result_identity": assessment.get("canonical_normalization_result_identity"),
        "sample_identity": assessment.get("sample_identity"),
        "config": assessment.get("config"),
        "pages": assessment.get("pages"),
    })
    if assessment_identity != expected_assessment_identity:
        raise ValueError("Assessment identity does not match its contents")
    _require_sha256(
        assessment.get("canonical_normalization_result_identity"),
        "Canonical normalization result identity",
    )
    _require_sha256(assessment.get("sample_identity"), "Assessment sample identity")
    preprocess = assessment.get("canonical_preprocess")
    detector = assessment.get("detector_selection")
    base_policy = assessment.get("base_normalization_policy")
    if not isinstance(preprocess, dict) or not isinstance(detector, dict) or not isinstance(base_policy, dict):
        raise ValueError("Assessment is missing current canonical normalization provenance; regenerate it")
    _require_sha256(preprocess.get("canonical_result_identity"), "Canonical preprocess result identity")
    if not str(detector.get("detector") or ""):
        raise ValueError("Assessment has no authoritative detector identity")
    _require_sha256(detector.get("parameter_identity_sha256"), "Detector parameter identity")
    if base_policy.get("id") != "axis-aligned-detector-envelope-v1":
        raise ValueError("Assessment was not generated from the current canonical crop policy")
    pages = assessment.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Assessment contains no evaluated pages")
    if int(assessment.get("sample_page_count") or 0) != len(pages):
        raise ValueError("Assessment page count does not match its evidence records")
    config = assessment.get("config") or {}
    recommendation = config.get("recommendation") or {}
    deskew = recommendation.get("deskew") or {}
    aggregate = assessment.get("aggregate") or {}
    hough = aggregate.get("hough-lines") or {}
    projection = aggregate.get("projection-profile") or {}
    required = {
        "policy_id": recommendation.get("policy_id"),
        "minimum_sample_pages": recommendation.get("minimum_sample_pages"),
        "minimum_mean_hough_confidence": recommendation.get("minimum_mean_hough_confidence"),
        "minimum_hough_over_projection_confidence_gap": recommendation.get("minimum_hough_over_projection_confidence_gap"),
        "maximum_hough_boundary_limited_pages": recommendation.get("maximum_hough_boundary_limited_pages"),
        "gross_orientation_action": recommendation.get("gross_orientation_action"),
        "minimum_absolute_correction_degrees": deskew.get("minimum_absolute_correction_degrees"),
        "maximum_absolute_correction_degrees": deskew.get("maximum_absolute_correction_degrees"),
        "minimum_confidence": deskew.get("minimum_confidence"),
        "minimum_line_count": deskew.get("minimum_line_count"),
        "maximum_weighted_mad_degrees": deskew.get("maximum_weighted_mad_degrees"),
        "canvas": deskew.get("canvas"),
        "interpolation": deskew.get("interpolation"),
    }
    missing = [name for name, value in required.items() if value is None or value == ""]
    if missing:
        raise ValueError(f"Assessment recommendation configuration is incomplete: {', '.join(missing)}")
    if recommendation["gross_orientation_action"] != "preserve":
        raise ValueError("Automatic gross-orientation changes are outside the current evidence contract")
    if deskew["canvas"] != "expanded-white" or deskew["interpolation"] != "linear":
        raise ValueError("Recommendation requests an unsupported resampling contract")

    confidence_gap = float(hough.get("mean_confidence") or 0.0) - float(projection.get("mean_confidence") or 0.0)
    evidence_gates = {
        "sample_size": int(assessment.get("sample_page_count") or 0) >= int(recommendation.get("minimum_sample_pages") or 0),
        "hough_mean_confidence": float(hough.get("mean_confidence") or 0.0) >= float(recommendation.get("minimum_mean_hough_confidence") or 0.0),
        "hough_confidence_advantage": confidence_gap >= float(recommendation.get("minimum_hough_over_projection_confidence_gap") or 0.0),
        "hough_not_boundary_limited": int(hough.get("boundary_limited_pages") or 0) <= int(recommendation.get("maximum_hough_boundary_limited_pages") or 0),
        "observed_corrections_within_policy_bound": float(hough.get("maximum_absolute_correction_degrees") or 0.0) <= float(deskew["maximum_absolute_correction_degrees"]),
    }
    eligible = all(evidence_gates.values())
    selected: list[dict[str, Any]] = []
    for page in assessment.get("pages") or []:
        estimate = (page.get("estimators") or {}).get("hough-lines") or {}
        angle = float(estimate.get("estimated_correction_degrees") or 0.0)
        mad = estimate.get("weighted_mad_degrees")
        checks = (
            abs(angle) >= float(deskew["minimum_absolute_correction_degrees"]),
            abs(angle) <= float(deskew["maximum_absolute_correction_degrees"]),
            float(estimate.get("confidence") or 0.0) >= float(deskew["minimum_confidence"]),
            int(estimate.get("line_count") or 0) >= int(deskew["minimum_line_count"]),
            mad is not None and float(mad) <= float(deskew["maximum_weighted_mad_degrees"]),
            not bool(estimate.get("boundary_limited")),
        )
        if all(checks):
            selected.append({
                "global_ordinal": int(page["global_ordinal"]),
                "correction_degrees": angle,
                "confidence": estimate.get("confidence"),
            })

    policy = {
        "schema_version": "1.0",
        "policy_type": "orientation-deskew-normalization",
        "policy_id": str(recommendation["policy_id"]),
        "status": "recommended-for-validation" if eligible else "insufficient-evidence",
        "activation": "explicit-normalization-workflow-selection-required",
        "plain_language": {
            "recommendation": (
                "Keep every page in its existing gross orientation and straighten only clearly tilted pages using conservative line evidence."
                if eligible else
                "Keep the current axis-aligned normalization until more representative evidence is available."
            ),
            "pixel_change": "Only pages passing every safety gate are resampled; all other pages remain byte-for-byte equivalent at the pixel-array level.",
            "review": "The assessment contact sheets remain the human-readable evidence for this recommendation.",
        },
        "gross_orientation": {
            "action": str(recommendation.get("gross_orientation_action") or "preserve"),
            "degrees": 0,
            "reason": "The assessment has no independent semantic truth for upright versus upside-down pages.",
        },
        "deskew": {
            "estimator": "hough-lines",
            "minimum_absolute_correction_degrees": float(deskew["minimum_absolute_correction_degrees"]),
            "maximum_absolute_correction_degrees": float(deskew["maximum_absolute_correction_degrees"]),
            "minimum_confidence": float(deskew["minimum_confidence"]),
            "minimum_line_count": int(deskew["minimum_line_count"]),
            "maximum_weighted_mad_degrees": float(deskew["maximum_weighted_mad_degrees"]),
            "canvas": str(deskew.get("canvas") or "expanded-white"),
            "interpolation": str(deskew.get("interpolation") or "linear"),
        },
        "compatibility": {
            "canonical_preprocess_result_identity": preprocess.get("canonical_result_identity"),
            "detector": detector.get("detector"),
            "parameter_identity_sha256": detector.get("parameter_identity_sha256"),
            "base_normalization_policy_id": base_policy.get("id"),
        },
        "evidence": {
            "assessment_identity": assessment_identity,
            "canonical_normalization_result_identity": assessment.get("canonical_normalization_result_identity"),
            "sample_page_count": assessment.get("sample_page_count"),
            "population_page_count": assessment.get("population_page_count"),
            "evidence_gates": evidence_gates,
            "confidence_gap": round(confidence_gap, 6),
            "sample_pages_passing_transform_gates": selected,
        },
    }
    policy["policy_identity"] = canonical_hash(policy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gate_lines = [f"- {'Passed' if passed else 'Needs more evidence'}: `{name}`" for name, passed in evidence_gates.items()]
    summary = "\n".join([
        "## Normalization recommendation",
        "",
        f"**{'Recommended for a validation run' if eligible else 'Not ready for validation'}:** {policy['plain_language']['recommendation']}",
        "",
        f"- Suggested policy: `{policy['policy_id']}`",
        "- Gross page orientation: preserve as-is",
        f"- Expected sample impact: `{len(selected)}` of `{assessment.get('sample_page_count')}` sampled pages",
        "- Activation: a researcher must explicitly choose the recommended policy when starting normalization",
        "",
        "### Automated evidence checks",
        "",
        *gate_lines,
        "",
        "The evidence and machine-readable policy were preserved automatically. The run artifact adds contact sheets for optional visual review; no production pixels were changed.",
        "",
        "**Next step:** review the recommendation, then start collection normalization. Its default prepared-recommendation option is the explicit approval to apply this policy; choose axis-aligned-only to preserve cropped pixels.",
        "",
    ])
    (output_path.parent / "recommendation.md").write_text(summary, encoding="utf-8")
    existing_summary = output_path.parent / "summary.md"
    if existing_summary.is_file():
        with existing_summary.open("a", encoding="utf-8") as handle:
            handle.write("\n" + summary)
    if github_summary:
        with github_summary.open("a", encoding="utf-8") as handle:
            handle.write("\n" + summary)
    return policy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Select a stratified canonical crop sample")
    prepare.add_argument("--normalization-manifest", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--golden-set", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    materialize = commands.add_parser("materialize", help="Reconstruct and verify sampled canonical crops")
    materialize.add_argument("--source-root", type=Path, required=True)
    materialize.add_argument("--canonical-image-manifest", type=Path, required=True)
    materialize.add_argument("--normalization-manifest", type=Path, required=True)
    materialize.add_argument("--sample-plan", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Generate orientation and deskew diagnostic evidence")
    evaluate.add_argument("--image-root", type=Path, required=True)
    evaluate.add_argument("--normalization-manifest", type=Path, required=True)
    evaluate.add_argument("--sample-plan", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--github-summary", type=Path)
    recommend = commands.add_parser("recommend", help="Create a safe machine-readable normalization policy candidate")
    recommend.add_argument("--assessment", type=Path, required=True)
    recommend.add_argument("--output", type=Path, required=True)
    recommend.add_argument("--github-summary", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_sample(args.normalization_manifest, args.config, args.output, args.golden_set)
    elif args.command == "materialize":
        materialize_sample(
            args.source_root,
            args.canonical_image_manifest,
            args.normalization_manifest,
            args.sample_plan,
            args.output,
        )
    elif args.command == "evaluate":
        payload = assess(args.image_root, args.normalization_manifest, args.sample_plan, args.config, args.output)
        if args.github_summary:
            with args.github_summary.open("a", encoding="utf-8") as handle:
                handle.write(_summary(payload))
    else:
        recommend_policy(args.assessment, args.output, args.github_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
