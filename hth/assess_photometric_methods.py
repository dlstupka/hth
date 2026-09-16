#!/usr/bin/env python3
"""Compare deterministic photometric corrections on persisted paper-page candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hth.assess_photometric import estimate_photometric_condition
from hth.canonical_build_evidence import canonical_hash
from hth.normalize_document_images import _find_image, _pixel_sha256, _sha256


SCHEMA_VERSION = "1.0"
ASSESSMENT_TYPE = "photometric-method-comparison"
PHOTOMETRIC_IDENTITY_FIELDS = (
    "assessment_type",
    "canonical_normalization_result_identity",
    "sample_identity",
    "config",
    "pages",
)
METHOD_IDENTITY_FIELDS = (
    "assessment_type",
    "canonical_normalization_result_identity",
    "photometric_assessment_identity",
    "sample_identity",
    "config",
    "pages",
)
SAFETY_CHECKS = {
    "background_span_reduction",
    "detail_preservation",
    "endpoint_clipping",
    "median_luminance_shift",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image[:, :, :3].copy()


def _validate_photometric_assessment(photometric: dict[str, Any]) -> None:
    if photometric.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported photometric assessment schema")
    if photometric.get("assessment_type") != "photometric-uniformity-comparison":
        raise ValueError("Not a photometric assessment")
    if photometric.get("status") != "diagnostic-only":
        raise ValueError("Photometric assessment is not complete diagnostic evidence")
    try:
        expected = canonical_hash({key: photometric[key] for key in PHOTOMETRIC_IDENTITY_FIELDS})
    except KeyError as exc:
        raise ValueError(f"Photometric assessment is missing identity field: {exc.args[0]}") from exc
    if photometric.get("assessment_identity") != expected:
        raise ValueError("Photometric assessment identity does not match its contents")


def _validate_method_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION or config.get("assessment_type") != ASSESSMENT_TYPE:
        raise ValueError("Unsupported photometric method configuration")
    methods = config.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("Photometric method configuration has no methods")
    identifiers = [str(method.get("id") or "") for method in methods]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Photometric method IDs must be present and unique")
    for method in methods:
        if method.get("mode") not in {"subtract", "divide"}:
            raise ValueError(f"Unsupported photometric method mode: {method.get('mode')}")
        strength = float(method.get("strength", -1.0))
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"Photometric method strength must be in (0, 1]: {method.get('id')}")
    recommendation = config.get("recommendation") or {}
    if recommendation.get("require_one_method_safe_for_every_candidate") is not True:
        raise ValueError("Photometric method recommendation must require one method to pass every candidate")
    if recommendation.get("selection_rule") != "minimum-strength-then-highest-score":
        raise ValueError(f"Unsupported photometric method selection rule: {recommendation.get('selection_rule')}")


def _derive_recommendation(
    config: dict[str, Any], pages: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any] | None]:
    if not pages:
        raise ValueError("Photometric method assessment has no candidate pages")
    configured = {str(method["id"]): method for method in config["methods"]}
    method_summary = []
    globally_safe = []
    for method_id, method in configured.items():
        results = []
        for page in pages:
            variants = page.get("variants") or []
            if {str(item.get("method_id") or "") for item in variants} != set(configured):
                raise ValueError(f"Page {page.get('global_ordinal')} does not contain the configured method set")
            result = next(item for item in variants if item["method_id"] == method_id)
            checks = result.get("checks") or {}
            if set(checks) != SAFETY_CHECKS:
                raise ValueError(f"Method safety checks are incomplete for page {page.get('global_ordinal')}: {method_id}")
            if bool(result.get("safe")) != all(bool(value) for value in checks.values()):
                raise ValueError(f"Method safety result is inconsistent for page {page.get('global_ordinal')}: {method_id}")
            results.append(result)
        summary = {
            "method_id": method_id,
            "mode": method["mode"],
            "strength": float(method["strength"]),
            "safe_page_count": sum(bool(item["safe"]) for item in results),
            "mean_background_span_reduction_fraction": round(float(np.mean([item["background_span_reduction_fraction"] for item in results])), 6),
            "mean_high_frequency_correlation": round(float(np.mean([item["high_frequency_correlation"] for item in results])), 6),
            "mean_score": round(float(np.mean([item["score"] for item in results])), 6),
        }
        method_summary.append(summary)
        if summary["safe_page_count"] == len(pages):
            globally_safe.append(method_id)
    safe_summaries = [item for item in method_summary if item["method_id"] in globally_safe]
    recommended = min(
        safe_summaries,
        key=lambda item: (item["strength"], -item["mean_score"], item["method_id"]),
        default=None,
    )
    return method_summary, globally_safe, recommended


def prepare_sample(photometric: dict[str, Any], normalization: dict[str, Any]) -> dict[str, Any]:
    _validate_photometric_assessment(photometric)
    if photometric.get("canonical_normalization_result_identity") != normalization.get("canonical_result_identity"):
        raise ValueError("Photometric evidence does not match canonical normalization")
    candidates = [
        page for page in photometric.get("pages") or []
        if (page.get("estimate") or {}).get("decision") == "correction-candidate"
        and (page.get("estimate") or {}).get("archetype") == "paper-page"
    ]
    if not candidates:
        raise ValueError("Photometric assessment has no actionable paper-page candidates")
    pages = [{
        "global_ordinal": int(page["global_ordinal"]),
        "reasons": list((page.get("estimate") or {}).get("decision_reasons") or []),
    } for page in candidates]
    pages.sort(key=lambda page: page["global_ordinal"])
    ordinals = [page["global_ordinal"] for page in pages]
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("Photometric assessment has duplicate correction candidates")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "assessment_type": "photometric-method-sample-plan",
        "canonical_normalization_result_identity": normalization.get("canonical_result_identity"),
        "photometric_assessment_identity": photometric.get("assessment_identity"),
        "population_page_count": len(normalization.get("pages") or []),
        "sample_page_count": len(pages),
        "pages": pages,
    }
    payload["sample_identity"] = canonical_hash(payload)
    return payload


def _background_field(luminance: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    rows = int(cfg.get("grid_rows") or 8)
    columns = int(cfg.get("grid_columns") or 8)
    percentile = float(cfg.get("background_percentile") or 85.0)
    grid = np.empty((rows, columns), dtype=np.float32)
    for row in range(rows):
        top, bottom = round(row * luminance.shape[0] / rows), round((row + 1) * luminance.shape[0] / rows)
        for column in range(columns):
            left, right = round(column * luminance.shape[1] / columns), round((column + 1) * luminance.shape[1] / columns)
            grid[row, column] = float(np.percentile(luminance[top:bottom, left:right], percentile))
    field = cv2.resize(grid, (luminance.shape[1], luminance.shape[0]), interpolation=cv2.INTER_CUBIC)
    sigma = max(1.0, max(luminance.shape) * float(cfg.get("smoothing_sigma_fraction") or 0.02))
    return np.clip(cv2.GaussianBlur(field, (0, 0), sigmaX=sigma, sigmaY=sigma), 1.0, 255.0)


def apply_method(image: np.ndarray, method: dict[str, Any], field_config: dict[str, Any]) -> np.ndarray:
    color = _bgr(image)
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0].astype(np.float32)
    field = _background_field(luminance, field_config)
    target = float(np.median(field))
    strength = float(method["strength"])
    if method["mode"] == "subtract":
        corrected = luminance - strength * (field - target)
    elif method["mode"] == "divide":
        multiplier = 1.0 + strength * ((target / field) - 1.0)
        corrected = luminance * multiplier
    else:
        raise ValueError(f"Unknown photometric method mode: {method['mode']}")
    output_lab = lab.copy()
    output_lab[:, :, 0] = np.uint8(np.clip(np.rint(corrected), 0, 255))
    output = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    if image.ndim == 2:
        return cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    if image.shape[2] == 4:
        return np.dstack((output, image[:, :, 3]))
    return output


def _analysis_luminance(image: np.ndarray, maximum_dimension: int = 1400) -> np.ndarray:
    luminance = cv2.cvtColor(_bgr(image), cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) / 255.0
    scale = min(1.0, maximum_dimension / max(luminance.shape))
    return cv2.resize(luminance, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else luminance


def _detail_correlation(original: np.ndarray, corrected: np.ndarray) -> float:
    first, second = _analysis_luminance(original), _analysis_luminance(corrected)
    first -= cv2.GaussianBlur(first, (0, 0), 3.0)
    second -= cv2.GaussianBlur(second, (0, 0), 3.0)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return 1.0 if denominator <= 1e-12 else float(np.sum(first * second) / denominator)


def evaluate_variant(
    original: np.ndarray,
    corrected: np.ndarray,
    original_condition: dict[str, Any],
    photometric_config: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    condition = estimate_photometric_condition(corrected, photometric_config)
    before_span = float(original_condition["background_luminance_span"])
    after_span = float(condition["background_luminance_span"])
    reduction = (before_span - after_span) / max(before_span, 1e-9)
    detail = _detail_correlation(original, corrected)
    original_l = _analysis_luminance(original)
    corrected_l = _analysis_luminance(corrected)
    original_endpoints = float(np.mean((original_l <= 2 / 255) | (original_l >= 253 / 255)))
    corrected_endpoints = float(np.mean((corrected_l <= 2 / 255) | (corrected_l >= 253 / 255)))
    clipping_delta = max(0.0, corrected_endpoints - original_endpoints)
    median_shift = abs(float(np.median(corrected_l) - np.median(original_l)))
    checks = {
        "background_span_reduction": reduction >= float(gates["minimum_background_span_reduction_fraction"]),
        "detail_preservation": detail >= float(gates["minimum_high_frequency_correlation"]),
        "endpoint_clipping": clipping_delta <= float(gates["maximum_new_endpoint_clipping_fraction"]),
        "median_luminance_shift": median_shift <= float(gates["maximum_absolute_median_luminance_shift"]),
    }
    score = reduction * 0.65 + detail * 0.35 - clipping_delta * 4.0 - median_shift * 0.5
    return {
        "safe": all(checks.values()),
        "checks": checks,
        "background_span_before": round(before_span, 6),
        "background_span_after": round(after_span, 6),
        "background_span_reduction_fraction": round(reduction, 6),
        "high_frequency_correlation": round(detail, 6),
        "new_endpoint_clipping_fraction": round(clipping_delta, 6),
        "absolute_median_luminance_shift": round(median_shift, 6),
        "score": round(score, 6),
        "condition_after": condition,
        "output_pixel_sha256": _pixel_sha256(corrected),
    }


def _contact_sheet(original: np.ndarray, variants: list[tuple[str, np.ndarray, dict[str, Any]]], ordinal: int) -> np.ndarray:
    panels = [("baseline", original, None), *variants]
    width, height = 520, 520
    canvas = np.full((2 * (height + 55), 3 * width, 3), 255, dtype=np.uint8)
    for index, (name, image, result) in enumerate(panels):
        display = _bgr(image)
        scale = min(width / display.shape[1], height / display.shape[0])
        display = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        row, column = divmod(index, 3)
        top, left = row * (height + 55), column * width
        canvas[top + 55:top + 55 + display.shape[0], left:left + display.shape[1]] = display
        label = f"Page {ordinal} | {name}"
        if result is not None:
            label += f" | {'PASS' if result['safe'] else 'HOLD'} | reduction {result['background_span_reduction_fraction']:.2f}"
        cv2.putText(canvas, label, (left + 8, top + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def _summary(payload: dict[str, Any]) -> str:
    lines = [
        "# HTH Photometric Method Assessment", "",
        "> Diagnostic comparison only. Canonical normalized pixels were not changed.", "",
        f"- Candidate pages: `{payload['sample_page_count']}`",
        f"- Globally safe methods: `{len(payload['globally_safe_methods'])}`",
        f"- Recommended method: `{payload.get('recommended_method_id') or 'none'}`", "",
        "| Method | Safe pages | Mean reduction | Mean detail correlation |",
        "|---|---:|---:|---:|",
    ]
    for method in payload["method_summary"]:
        lines.append(
            f"| {method['method_id']} | {method['safe_page_count']}/{payload['sample_page_count']} | "
            f"{method['mean_background_span_reduction_fraction']:.3f} | {method['mean_high_frequency_correlation']:.4f} |"
        )
    lines.extend(["", "Open `index.html` to compare every candidate and deterministic variant.", ""])
    return "\n".join(lines)


def assess(
    image_root: Path,
    photometric_path: Path,
    normalization_path: Path,
    sample_path: Path,
    photometric_config_path: Path,
    method_config_path: Path,
    output: Path,
) -> dict[str, Any]:
    photometric = _read_json(photometric_path)
    normalization = _read_json(normalization_path)
    sample = _read_json(sample_path)
    photometric_config = _read_json(photometric_config_path)
    config = _read_json(method_config_path)
    _validate_method_config(config)
    expected_sample = prepare_sample(photometric, normalization)
    if sample != expected_sample:
        raise ValueError("Photometric method sample does not match persisted candidate evidence")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Photometric method output is not empty: {output}")
    contacts, variants_root = output / "contact-sheets", output / "variants"
    contacts.mkdir(parents=True, exist_ok=True)
    variants_root.mkdir(parents=True, exist_ok=True)
    by_ordinal = {int(page["global_ordinal"]): page for page in photometric.get("pages") or []}
    pages = []
    for planned in sample["pages"]:
        ordinal = int(planned["global_ordinal"])
        source = _find_image(image_root, ordinal)
        original = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if original is None:
            raise ValueError(f"Could not decode candidate page {ordinal}")
        original_condition = by_ordinal[ordinal]["estimate"]
        page_variants = []
        preview_variants = []
        page_root = variants_root / f"fs_{ordinal:04d}"
        page_root.mkdir(parents=True, exist_ok=True)
        for method in config["methods"]:
            corrected = apply_method(original, method, config["background_field"])
            result = evaluate_variant(original, corrected, original_condition, photometric_config, config["safety_gates"])
            target = page_root / f"{method['id']}.png"
            if not cv2.imwrite(str(target), corrected, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
                raise ValueError(f"Could not write {method['id']} output for page {ordinal}")
            round_trip = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
            if round_trip is None or not np.array_equal(round_trip, corrected):
                raise ValueError(f"Lossless round-trip failed for {method['id']} page {ordinal}")
            result.update({"method_id": method["id"], "output_file": target.relative_to(output).as_posix()})
            page_variants.append(result)
            preview_variants.append((method["id"], corrected, result))
        sheet = _contact_sheet(original, preview_variants, ordinal)
        if not cv2.imwrite(str(contacts / f"fs_{ordinal:04d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ValueError(f"Could not write contact sheet for page {ordinal}")
        pages.append({
            "global_ordinal": ordinal,
            "source_sha256": _sha256(source),
            "source_pixel_sha256": _pixel_sha256(original),
            "variants": page_variants,
        })
    method_summary, globally_safe, recommended = _derive_recommendation(config, pages)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "assessment_type": ASSESSMENT_TYPE,
        "status": "diagnostic-only",
        "canonical_normalization_result_identity": normalization.get("canonical_result_identity"),
        "photometric_assessment_identity": photometric.get("assessment_identity"),
        "sample_identity": sample.get("sample_identity"),
        "sample_page_count": len(pages),
        "config": config,
        "globally_safe_methods": globally_safe,
        "recommended_method_id": recommended["method_id"] if recommended else None,
        "method_summary": method_summary,
        "pages": pages,
    }
    payload["assessment_identity"] = canonical_hash({key: payload[key] for key in METHOD_IDENTITY_FIELDS})
    (output / "assessment.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (output / "assessment.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("global_ordinal", "method_id", "safe", "background_span_reduction_fraction", "high_frequency_correlation", "new_endpoint_clipping_fraction", "absolute_median_luminance_shift", "output_pixel_sha256"))
        writer.writeheader()
        for page in pages:
            for variant in page["variants"]:
                writer.writerow({"global_ordinal": page["global_ordinal"], **{key: variant[key] for key in writer.fieldnames if key != "global_ordinal"}})
    summary = _summary(payload)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    cards = "".join(f'<article><h2>Page {page["global_ordinal"]}</h2><img src="contact-sheets/fs_{page["global_ordinal"]:04d}.jpg"></article>' for page in pages)
    (output / "index.html").write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>HTH Photometric Method Assessment</title><style>body{{font-family:system-ui;background:#111820;color:#e6edf3;margin:2rem}}img{{max-width:100%}}</style></head><body><h1>HTH Photometric Method Assessment</h1>{cards}</body></html>', encoding="utf-8")
    return payload


def recommend(assessment_path: Path, output_path: Path) -> dict[str, Any]:
    assessment = _read_json(assessment_path)
    if assessment.get("schema_version") != SCHEMA_VERSION or assessment.get("status") != "diagnostic-only":
        raise ValueError("Photometric method assessment is not complete diagnostic evidence")
    try:
        expected = canonical_hash({key: assessment[key] for key in METHOD_IDENTITY_FIELDS})
    except KeyError as exc:
        raise ValueError(f"Photometric method assessment is missing identity field: {exc.args[0]}") from exc
    if assessment.get("assessment_type") != ASSESSMENT_TYPE or assessment.get("assessment_identity") != expected:
        raise ValueError("Photometric method assessment identity does not match its contents")
    config = assessment.get("config") or {}
    _validate_method_config(config)
    pages = assessment.get("pages") or []
    method_summary, globally_safe, recommended = _derive_recommendation(config, pages)
    method_id = recommended["method_id"] if recommended else None
    if assessment.get("method_summary") != method_summary or assessment.get("globally_safe_methods") != globally_safe or assessment.get("recommended_method_id") != method_id:
        raise ValueError("Photometric method recommendation fields do not match page evidence")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "policy_type": "photometric-method-validation",
        "policy_id": config.get("recommendation", {}).get("policy_id"),
        "status": "validation-candidate" if method_id else "manual-review-required",
        "action": "validate-method" if method_id else "withhold",
        "recommended_method_id": method_id,
        "compatibility": {
            "canonical_normalization_result_identity": assessment.get("canonical_normalization_result_identity"),
            "photometric_assessment_identity": assessment.get("photometric_assessment_identity"),
        },
        "evidence": {"assessment_identity": assessment.get("assessment_identity"), "globally_safe_methods": globally_safe},
    }
    payload["policy_identity"] = canonical_hash(payload)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    statement = (
        f"Method `{method_id}` passed every deterministic safety gate on every correction candidate. Review it before any production integration."
        if method_id else "No single method passed every safety gate on every candidate; preserve current pixels."
    )
    summary = f"## Photometric method recommendation\n\n**{statement}**\n\n- Status: `{payload['status']}`\n- Action: `{payload['action']}`\n"
    (output_path.parent / "recommendation.md").write_text(summary, encoding="utf-8")
    with (output_path.parent / "summary.md").open("a", encoding="utf-8") as handle:
        handle.write("\n" + summary)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--photometric-assessment", type=Path, required=True)
    prepare.add_argument("--normalization-manifest", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate")
    for name in ("image-root", "photometric-assessment", "normalization-manifest", "sample-plan", "photometric-config", "method-config", "output"):
        evaluate.add_argument(f"--{name}", type=Path, required=True)
    evaluate.add_argument("--github-summary", type=Path)
    recommendation = commands.add_parser("recommend")
    recommendation.add_argument("--assessment", type=Path, required=True)
    recommendation.add_argument("--output", type=Path, required=True)
    recommendation.add_argument("--github-summary", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_sample(_read_json(args.photometric_assessment), _read_json(args.normalization_manifest))
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    elif args.command == "evaluate":
        payload = assess(args.image_root, args.photometric_assessment, args.normalization_manifest, args.sample_plan, args.photometric_config, args.method_config, args.output)
        if args.github_summary:
            with args.github_summary.open("a", encoding="utf-8") as handle:
                handle.write(_summary(payload))
    else:
        recommend(args.assessment, args.output)
        if args.github_summary:
            with args.github_summary.open("a", encoding="utf-8") as handle:
                handle.write((args.output.parent / "recommendation.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
