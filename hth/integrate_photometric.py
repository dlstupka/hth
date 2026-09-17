#!/usr/bin/env python3
"""Apply one fully validated photometric method to the canonical collection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any
import zipfile

import cv2
import numpy as np

from hth.assess_photometric import estimate_photometric_condition
from hth.assess_photometric_methods import (
    _bgr,
    _validate_photometric_assessment,
    apply_method,
    evaluate_variant,
)
from hth.canonical_build_evidence import canonical_hash
from hth.normalize_document_images import _find_image, _pixel_sha256, _sha256
from hth.validate_photometric_method import (
    VALIDATION_IDENTITY_FIELDS,
    VALIDATION_TYPE,
    _derive_result,
    _validate_method_inputs,
)


SCHEMA_VERSION = "1.0"
INTEGRATION_TYPE = "photometric-background-field-integration"
RESULT_IDENTITY_FIELDS = (
    "integration_type",
    "base_normalization_result_identity",
    "photometric_assessment_identity",
    "method_assessment_identity",
    "method_policy_identity",
    "validation_identity",
    "integration_policy_identity",
    "method",
    "pages",
)

RELEASE_TYPE = "canonical-photometric-normalized-collection"
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _validate_validation(validation: dict[str, Any]) -> None:
    if (
        validation.get("schema_version") != SCHEMA_VERSION
        or validation.get("validation_type") != VALIDATION_TYPE
        or validation.get("status") != "diagnostic-only"
    ):
        raise ValueError("Unsupported photometric method validation")
    expected = canonical_hash({key: validation[key] for key in VALIDATION_IDENTITY_FIELDS})
    if validation.get("validation_identity") != expected:
        raise ValueError("Photometric method validation identity does not match its contents")
    aggregate, gates, status, action = _derive_result(
        validation["config"], validation["pages"], int(validation["held_out_population_page_count"])
    )
    if (
        validation.get("aggregate") != aggregate
        or validation.get("gates") != gates
        or validation.get("recommendation_status") != status
        or validation.get("recommendation_action") != action
    ):
        raise ValueError("Photometric method validation recommendation does not match page evidence")
    if status != "integration-candidate" or action != "prepare-integration" or not all(gates.values()):
        raise ValueError("Photometric method validation does not authorize integration")


def _validate_integration_policy(policy: dict[str, Any], validation: dict[str, Any]) -> None:
    claimed = str(policy.get("policy_identity") or "")
    body = {key: value for key, value in policy.items() if key != "policy_identity"}
    if len(claimed) != 64 or claimed != canonical_hash(body):
        raise ValueError("Photometric integration policy identity does not match its contents")
    if policy.get("policy_type") != "photometric-method-integration":
        raise ValueError("Unsupported photometric integration policy")
    if policy.get("status") != "integration-candidate" or policy.get("action") != "prepare-integration":
        raise ValueError("Photometric integration policy does not authorize integration")
    if policy.get("method") != validation.get("method"):
        raise ValueError("Photometric integration policy selects a different method")
    compatibility = policy.get("compatibility") or {}
    expected = {
        "canonical_normalization_result_identity": validation.get("canonical_normalization_result_identity"),
        "photometric_assessment_identity": validation.get("photometric_assessment_identity"),
        "method_assessment_identity": validation.get("method_assessment_identity"),
        "method_policy_identity": validation.get("method_policy_identity"),
    }
    if compatibility != expected:
        raise ValueError("Photometric integration policy compatibility does not match validation")
    evidence = policy.get("evidence") or {}
    if evidence.get("validation_identity") != validation.get("validation_identity"):
        raise ValueError("Photometric integration policy does not reference authoritative validation")
    if evidence.get("gates") != validation.get("gates") or evidence.get("aggregate") != validation.get("aggregate"):
        raise ValueError("Photometric integration policy evidence does not match validation")


def _validate_inputs(
    normalization: dict[str, Any],
    photometric: dict[str, Any],
    method_assessment: dict[str, Any],
    method_policy: dict[str, Any],
    validation: dict[str, Any],
    integration_policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[int, tuple[str, dict[str, Any]]]]:
    normalization_identity = str(normalization.get("canonical_result_identity") or "")
    pages = normalization.get("pages") or []
    if len(normalization_identity) != 64 or not pages:
        raise ValueError("Canonical normalization manifest is incomplete")
    normalization_ordinals = [int(page["global_ordinal"]) for page in pages]
    if len(normalization_ordinals) != len(set(normalization_ordinals)):
        raise ValueError("Canonical normalization manifest has duplicate page ordinals")

    _validate_photometric_assessment(photometric)
    if photometric.get("canonical_normalization_result_identity") != normalization_identity:
        raise ValueError("Photometric assessment does not match canonical normalization")
    method = _validate_method_inputs(method_assessment, method_policy, normalization_identity)
    if method_assessment.get("photometric_assessment_identity") != photometric.get("assessment_identity"):
        raise ValueError("Photometric method assessment does not match photometric evidence")

    _validate_validation(validation)
    if validation.get("canonical_normalization_result_identity") != normalization_identity:
        raise ValueError("Photometric validation does not match canonical normalization")
    if validation.get("photometric_assessment_identity") != photometric.get("assessment_identity"):
        raise ValueError("Photometric validation does not match photometric evidence")
    if validation.get("method_assessment_identity") != method_assessment.get("assessment_identity"):
        raise ValueError("Photometric validation does not match method evidence")
    if validation.get("method_policy_identity") != method_policy.get("policy_identity"):
        raise ValueError("Photometric validation does not match method policy")
    if validation.get("method") != method:
        raise ValueError("Photometric validation uses a different method")
    _validate_integration_policy(integration_policy, validation)

    references: dict[int, tuple[str, dict[str, Any]]] = {}
    for page in method_assessment.get("pages") or []:
        ordinal = int(page["global_ordinal"])
        if ordinal in references:
            raise ValueError(f"Duplicate development evidence for page {ordinal}")
        variants = [item for item in page.get("variants") or [] if item.get("method_id") == method["id"]]
        if len(variants) != 1:
            raise ValueError(f"Development page {ordinal} lacks exactly one selected-method variant")
        references[ordinal] = ("development", {
            "global_ordinal": ordinal,
            "route": "apply",
            "source_pixel_sha256": page.get("source_pixel_sha256"),
            "output_pixel_sha256": variants[0].get("output_pixel_sha256"),
            "method_result": variants[0],
        })
    for page in validation.get("pages") or []:
        ordinal = int(page["global_ordinal"])
        if ordinal in references:
            raise ValueError(f"Development and held-out evidence overlap on page {ordinal}")
        references[ordinal] = ("held-out", page)
    if set(references) != set(normalization_ordinals):
        missing = sorted(set(normalization_ordinals) - set(references))
        extra = sorted(set(references) - set(normalization_ordinals))
        raise ValueError(f"Photometric evidence does not partition the canonical population; missing={missing}, extra={extra}")
    return method, references


def prepare_plan(
    normalization: dict[str, Any],
    photometric: dict[str, Any],
    method_assessment: dict[str, Any],
    method_policy: dict[str, Any],
    validation: dict[str, Any],
    integration_policy: dict[str, Any],
) -> dict[str, Any]:
    method, references = _validate_inputs(
        normalization, photometric, method_assessment, method_policy, validation, integration_policy
    )
    pages = [{
        "global_ordinal": ordinal,
        "reasons": ["complete-production-population", f"{source}-evidence"],
        "expected_route": reference["route"],
    } for ordinal, (source, reference) in sorted(references.items())]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "assessment_type": "photometric-integration-plan",
        "canonical_normalization_result_identity": normalization.get("canonical_result_identity"),
        "photometric_assessment_identity": photometric.get("assessment_identity"),
        "method_assessment_identity": method_assessment.get("assessment_identity"),
        "method_policy_identity": method_policy.get("policy_identity"),
        "validation_identity": validation.get("validation_identity"),
        "integration_policy_identity": integration_policy.get("policy_identity"),
        "method_id": method["id"],
        "population_page_count": len(pages),
        "sample_page_count": len(pages),
        "pages": pages,
    }
    payload["sample_identity"] = canonical_hash(payload)
    return payload


def _contact_sheet(original: np.ndarray, output: np.ndarray, ordinal: int, route: str) -> np.ndarray:
    width, height, header = 560, 620, 58
    canvas = np.full((height, width * 2, 3), 255, dtype=np.uint8)
    for index, (label, image) in enumerate((("canonical", original), (route, output))):
        display = _bgr(image)
        scale = min(width / display.shape[1], (height - header) / display.shape[0])
        display = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        left = index * width + (width - display.shape[1]) // 2
        top = header + (height - header - display.shape[0]) // 2
        canvas[top:top + display.shape[0], left:left + display.shape[1]] = display
        cv2.putText(canvas, label, (index * width + 10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Page {ordinal} | {route}", (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def _summary(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    return "\n".join([
        "# HTH Photometric Integration", "",
        "> Selective background-field correction. No contrast, tonal-range, sharpening, denoising, or binarization transform was applied.", "",
        f"- Method: `{payload['method']['id']}`",
        f"- Input pages: `{aggregate['page_count']}`",
        f"- Corrected pages: `{aggregate['corrected_pages']}`",
        f"- Preserved pages: `{aggregate['preserved_pages']}`",
        f"- Safe corrected pages: `{aggregate['safe_corrected_pages']}`",
        f"- Base normalization result: `{payload['base_normalization_result_identity']}`",
        f"- Photometric result: `{payload['photometric_result_identity']}`", "",
        "All pages continue downstream. Preserved pages retain their canonical pixels; corrected pages reproduce previously validated outputs.", "",
    ])


def integrate(
    image_root: Path,
    normalization: dict[str, Any],
    photometric: dict[str, Any],
    method_assessment: dict[str, Any],
    method_policy: dict[str, Any],
    validation: dict[str, Any],
    integration_policy: dict[str, Any],
    plan: dict[str, Any],
    output: Path,
    review_every: int = 25,
) -> dict[str, Any]:
    expected_plan = prepare_plan(
        normalization, photometric, method_assessment, method_policy, validation, integration_policy
    )
    if plan != expected_plan:
        raise ValueError("Photometric integration plan does not match authoritative evidence")
    method, references = _validate_inputs(
        normalization, photometric, method_assessment, method_policy, validation, integration_policy
    )
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Photometric integration output is not empty: {output}")
    images = output / "photometric-normalized"
    contacts = output / "contact-sheets"
    images.mkdir(parents=True, exist_ok=True)
    contacts.mkdir(parents=True, exist_ok=True)

    normalization_by_ordinal = {int(page["global_ordinal"]): page for page in normalization["pages"]}
    photometric_config = photometric["config"]
    method_config = method_assessment["config"]
    rows: list[dict[str, Any]] = []
    ordered = [int(page["global_ordinal"]) for page in plan["pages"]]
    for index, ordinal in enumerate(ordered):
        source = _find_image(image_root, ordinal)
        original = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if original is None:
            raise ValueError(f"Could not decode canonical normalized page {ordinal}")
        base = normalization_by_ordinal[ordinal]
        input_pixel_hash = _pixel_sha256(original)
        if input_pixel_hash != str(base.get("output_pixel_sha256") or ""):
            raise ValueError(f"Page {ordinal} input pixels do not match canonical normalization")

        condition = estimate_photometric_condition(original, photometric_config)
        if condition["decision"] == "correction-candidate" and condition.get("correction_eligible") is not True:
            raise ValueError(f"Page {ordinal} candidate violates correction eligibility")
        apply = condition["decision"] == "correction-candidate" and condition["archetype"] == "paper-page" and condition["correction_eligible"] is True
        route = "apply" if apply else "preserve"
        evidence_source, reference = references[ordinal]
        if route != reference.get("route"):
            raise ValueError(f"Page {ordinal} production route does not match {evidence_source} evidence")
        if input_pixel_hash != str(reference.get("source_pixel_sha256") or ""):
            raise ValueError(f"Page {ordinal} input pixels do not match {evidence_source} evidence")

        method_result = None
        if apply:
            result_image = apply_method(original, method, method_config["background_field"])
            method_result = evaluate_variant(
                original, result_image, condition, photometric_config, method_config["safety_gates"]
            )
            if method_result.get("safe") is not True:
                raise ValueError(f"Page {ordinal} failed production photometric safety gates")
        else:
            result_image = original
        output_pixel_hash = _pixel_sha256(result_image)
        if output_pixel_hash != str(reference.get("output_pixel_sha256") or ""):
            raise ValueError(f"Page {ordinal} output pixels do not reproduce {evidence_source} evidence")

        target = images / f"fs_{ordinal:04d}.png"
        if not cv2.imwrite(str(target), result_image, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise ValueError(f"Could not write photometric output page {ordinal}")
        round_trip = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
        if round_trip is None or not np.array_equal(round_trip, result_image):
            raise ValueError(f"Page {ordinal} failed lossless output round-trip")

        review = apply or index == 0 or index == len(ordered) - 1 or (review_every > 0 and index % review_every == 0)
        review_file = None
        if review:
            review_file = f"contact-sheets/fs_{ordinal:04d}.jpg"
            sheet = _contact_sheet(original, result_image, ordinal, route)
            if not cv2.imwrite(str(output / review_file), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise ValueError(f"Could not write photometric review sheet for page {ordinal}")
        rows.append({
            "global_ordinal": ordinal,
            "route": route,
            "pipeline_action": "corrected-and-continue" if apply else "preserve-and-continue",
            "evidence_source": evidence_source,
            "archetype": condition["archetype"],
            "decision_before": condition["decision"],
            "decision_reasons": condition["decision_reasons"],
            "correction_eligible": condition["correction_eligible"],
            "correction_exclusion_reasons": condition["correction_exclusion_reasons"],
            "input_pixel_sha256": input_pixel_hash,
            "output_pixel_sha256": output_pixel_hash,
            "output_sha256": _sha256(target),
            "output_width": int(result_image.shape[1]),
            "output_height": int(result_image.shape[0]),
            "method_safe": method_result.get("safe") if method_result else None,
            "background_span_reduction_fraction": method_result.get("background_span_reduction_fraction") if method_result else None,
            "high_frequency_correlation": method_result.get("high_frequency_correlation") if method_result else None,
            "review_artifact": review_file,
        })

    aggregate = {
        "page_count": len(rows),
        "corrected_pages": sum(page["route"] == "apply" for page in rows),
        "preserved_pages": sum(page["route"] == "preserve" for page in rows),
        "safe_corrected_pages": sum(page["method_safe"] is True for page in rows),
        "development_evidence_pages": sum(page["evidence_source"] == "development" for page in rows),
        "held_out_evidence_pages": sum(page["evidence_source"] == "held-out" for page in rows),
    }
    if aggregate["safe_corrected_pages"] != aggregate["corrected_pages"]:
        raise ValueError("Not every production correction passed safety gates")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "integration_type": INTEGRATION_TYPE,
        "status": "complete",
        "base_normalization_result_identity": normalization.get("canonical_result_identity"),
        "photometric_assessment_identity": photometric.get("assessment_identity"),
        "method_assessment_identity": method_assessment.get("assessment_identity"),
        "method_policy_identity": method_policy.get("policy_identity"),
        "validation_identity": validation.get("validation_identity"),
        "integration_policy_identity": integration_policy.get("policy_identity"),
        "method": method,
        "aggregate": aggregate,
        "pages": rows,
    }
    payload["photometric_result_identity"] = canonical_hash({key: payload[key] for key in RESULT_IDENTITY_FIELDS})
    (output / "photometric-normalization-manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (output / "photometric-normalization-manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "global_ordinal", "route", "pipeline_action", "evidence_source", "archetype", "decision_before",
            "input_pixel_sha256", "output_pixel_sha256", "output_sha256", "method_safe",
            "background_span_reduction_fraction", "high_frequency_correlation",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for page in rows:
            writer.writerow({key: page[key] for key in fields})
    summary = _summary(payload)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    cards = "".join(
        f'<article><h2>Page {page["global_ordinal"]}: {page["route"]}</h2><img loading="lazy" src="{page["review_artifact"]}"></article>'
        for page in rows if page["review_artifact"]
    )
    (output / "index.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>HTH Photometric Integration</title><style>body{{font-family:system-ui;background:#111820;color:#e6edf3;margin:2rem}}article{{margin:2rem 0}}img{{max-width:100%}}</style></head><body><h1>HTH Photometric Integration</h1><p>Selective illumination/background correction only.</p>{cards}</body></html>',
        encoding="utf-8",
    )
    return payload


def _add_evidence_arguments(parser: argparse.ArgumentParser) -> None:
    for name in (
        "normalization-manifest", "photometric-assessment", "method-assessment", "method-policy",
        "validation", "integration-policy",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)


def package_release(collection: Path, asset: Path, tag: str) -> dict[str, Any]:
    """Create a byte-stable Zip64 release asset and its external provenance record."""
    manifest = _read_json(collection / "photometric-normalization-manifest.json")
    identity = str(manifest.get("photometric_result_identity") or "")
    if manifest.get("status") != "complete" or len(identity) != 64:
        raise ValueError("Photometric collection has no complete result identity")
    expected_tag = f"HTH-PHOTOMETRIC-{identity}"
    if tag != expected_tag:
        raise ValueError(f"Release tag must be {expected_tag}")
    files = sorted(
        path for path in collection.rglob("*")
        if path.is_file() and path.name != "release.json"
    )
    if not files:
        raise ValueError("Photometric collection is empty")
    asset.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(asset, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in files:
            relative = Path(collection.name) / path.relative_to(collection)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
    digest = hashlib.sha256()
    with asset.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "schema_version": SCHEMA_VERSION,
        "release_type": RELEASE_TYPE,
        "photometric_result_identity": identity,
        "tag": tag,
        "asset": asset.name,
        "asset_sha256": digest.hexdigest(),
        "asset_size": asset.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    _add_evidence_arguments(prepare)
    prepare.add_argument("--output", type=Path, required=True)
    execute = commands.add_parser("integrate")
    _add_evidence_arguments(execute)
    execute.add_argument("--image-root", type=Path, required=True)
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--review-every", type=int, default=25)
    execute.add_argument("--github-summary", type=Path)
    package = commands.add_parser("package")
    package.add_argument("--collection", type=Path, required=True)
    package.add_argument("--asset", type=Path, required=True)
    package.add_argument("--tag", required=True)
    package.add_argument("--output-record", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "package":
        payload = package_release(args.collection, args.asset, args.tag)
        args.output_record.parent.mkdir(parents=True, exist_ok=True)
        args.output_record.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0
    inputs = [
        _read_json(args.normalization_manifest), _read_json(args.photometric_assessment),
        _read_json(args.method_assessment), _read_json(args.method_policy),
        _read_json(args.validation), _read_json(args.integration_policy),
    ]
    if args.command == "prepare":
        payload = prepare_plan(*inputs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        if args.review_every < 0:
            parser.error("--review-every must be zero or positive")
        payload = integrate(args.image_root, *inputs, _read_json(args.plan), args.output, args.review_every)
        if args.github_summary:
            with args.github_summary.open("a", encoding="utf-8") as handle:
                handle.write(_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
