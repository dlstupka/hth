#!/usr/bin/env python3
"""Validate one persisted photometric method on deterministic held-out pages."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hth.assess_photometric import estimate_photometric_condition
from hth.assess_photometric_methods import (
    METHOD_IDENTITY_FIELDS,
    _bgr,
    _derive_recommendation,
    _validate_method_config,
    _validate_photometric_assessment,
    apply_method,
    evaluate_variant,
)
from hth.canonical_build_evidence import canonical_hash
from hth.normalize_document_images import _find_image, _pixel_sha256, _sha256


SCHEMA_VERSION = "1.0"
VALIDATION_TYPE = "photometric-method-held-out-validation"
VALIDATION_IDENTITY_FIELDS = (
    "validation_type",
    "canonical_normalization_result_identity",
    "photometric_assessment_identity",
    "method_assessment_identity",
    "method_policy_identity",
    "sample_identity",
    "held_out_population_page_count",
    "config",
    "method",
    "pages",
)
PROTECTED_ARCHETYPES = {"dark-polarity-frame", "mixed-polarity-page"}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _systematic(items: list[int], count: int) -> list[int]:
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)
    indexes = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)] if count > 1 else [len(items) // 2]
    return [items[index] for index in indexes]


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION or config.get("validation_type") != VALIDATION_TYPE:
        raise ValueError("Unsupported photometric method validation configuration")
    sampling = config.get("sampling") or {}
    if sampling.get("scope") != "all-held-out-pages" or sampling.get("exclude_method_development_pages") is not True:
        raise ValueError("Photometric method validation must cover all non-development pages")
    gates = config.get("gates") or {}
    if float(gates.get("minimum_held_out_population_fraction") or 0.0) != 1.0:
        raise ValueError("Photometric method validation must require complete held-out coverage")
    if gates.get("require_every_candidate_safe") is not True or gates.get("require_every_protected_control_preserved") is not True:
        raise ValueError("Photometric validation must fail closed for candidates and protected controls")


def _validate_method_inputs(
    method_assessment: dict[str, Any], method_policy: dict[str, Any], normalization_identity: str
) -> dict[str, Any]:
    if method_assessment.get("schema_version") != SCHEMA_VERSION or method_assessment.get("assessment_type") != "photometric-method-comparison":
        raise ValueError("Unsupported photometric method assessment")
    if method_assessment.get("status") != "diagnostic-only" or method_assessment.get("canonical_normalization_result_identity") != normalization_identity:
        raise ValueError("Photometric method assessment does not match canonical normalization")
    expected = canonical_hash({key: method_assessment[key] for key in METHOD_IDENTITY_FIELDS})
    if method_assessment.get("assessment_identity") != expected:
        raise ValueError("Photometric method assessment identity does not match its contents")
    config = method_assessment.get("config") or {}
    _validate_method_config(config)
    summaries, safe_methods, recommended = _derive_recommendation(config, method_assessment.get("pages") or [])
    method_id = recommended["method_id"] if recommended else None
    if method_assessment.get("method_summary") != summaries or method_assessment.get("globally_safe_methods") != safe_methods or method_assessment.get("recommended_method_id") != method_id:
        raise ValueError("Photometric method recommendation fields do not match page evidence")
    claimed_policy_identity = method_policy.get("policy_identity")
    policy_body = {key: value for key, value in method_policy.items() if key != "policy_identity"}
    if claimed_policy_identity != canonical_hash(policy_body):
        raise ValueError("Photometric method policy identity does not match its contents")
    compatibility = method_policy.get("compatibility") or {}
    if method_policy.get("action") != "validate-method" or method_policy.get("status") != "validation-candidate":
        raise ValueError("Photometric method policy does not authorize validation")
    if method_policy.get("recommended_method_id") != method_id:
        raise ValueError("Photometric method policy does not select the evidence-backed method")
    if compatibility.get("canonical_normalization_result_identity") != normalization_identity:
        raise ValueError("Photometric method policy does not match canonical normalization")
    if compatibility.get("photometric_assessment_identity") != method_assessment.get("photometric_assessment_identity"):
        raise ValueError("Photometric method policy does not match photometric evidence")
    if (method_policy.get("evidence") or {}).get("assessment_identity") != method_assessment.get("assessment_identity"):
        raise ValueError("Photometric method policy does not reference the authoritative assessment")
    return next(method for method in config["methods"] if method["id"] == method_id)


def prepare_sample(
    normalization: dict[str, Any],
    photometric: dict[str, Any],
    method_assessment: dict[str, Any],
    method_policy: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    _validate_config(config)
    _validate_photometric_assessment(photometric)
    normalization_identity = str(normalization.get("canonical_result_identity") or "")
    if photometric.get("canonical_normalization_result_identity") != normalization_identity:
        raise ValueError("Photometric assessment does not match canonical normalization")
    method = _validate_method_inputs(method_assessment, method_policy, normalization_identity)
    if method_assessment.get("photometric_assessment_identity") != photometric.get("assessment_identity"):
        raise ValueError("Method assessment does not match the authoritative photometric assessment")

    development = {int(page["global_ordinal"]) for page in method_assessment.get("pages") or []}
    sampling = config.get("sampling") or {}
    population = sorted(
        int(page["global_ordinal"])
        for page in normalization.get("pages") or []
        if int(page["global_ordinal"]) not in development
    )
    reasons: dict[int, set[str]] = {}

    def include(ordinal: int, reason: str) -> None:
        if ordinal not in development:
            reasons.setdefault(ordinal, set()).add(reason)

    for ordinal in population:
        include(ordinal, "complete-held-out-population")

    known = sorted(photometric.get("pages") or [], key=lambda page: int(page["global_ordinal"]))
    protected = [page for page in known if (page.get("estimate") or {}).get("archetype") in PROTECTED_ARCHETYPES]
    if sampling.get("include_all_protected_archetypes") is True:
        for page in protected:
            include(int(page["global_ordinal"]), "known-protected-control")
    review = [int(page["global_ordinal"]) for page in known if (page.get("estimate") or {}).get("decision") == "review"]
    preserve = [int(page["global_ordinal"]) for page in known if (page.get("estimate") or {}).get("decision") == "preserve" and (page.get("estimate") or {}).get("archetype") == "paper-page"]
    for ordinal in _systematic(review, int(sampling.get("maximum_known_review_controls") or 0)):
        include(ordinal, "known-review-control")
    for ordinal in _systematic(preserve, int(sampling.get("maximum_known_preserve_controls") or 0)):
        include(ordinal, "known-preserve-control")

    pages = [{"global_ordinal": ordinal, "reasons": sorted(page_reasons)} for ordinal, page_reasons in sorted(reasons.items())]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "assessment_type": "photometric-method-validation-sample-plan",
        "canonical_normalization_result_identity": normalization_identity,
        "photometric_assessment_identity": photometric.get("assessment_identity"),
        "method_assessment_identity": method_assessment.get("assessment_identity"),
        "method_policy_identity": method_policy.get("policy_identity"),
        "method_id": method["id"],
        "population_page_count": len(normalization.get("pages") or []),
        "held_out_population_page_count": len(population),
        "excluded_development_pages": sorted(development),
        "sample_page_count": len(pages),
        "sampling": sampling,
        "pages": pages,
    }
    payload["sample_identity"] = canonical_hash(payload)
    return payload


def _contact_sheet(original: np.ndarray, output: np.ndarray, page: dict[str, Any]) -> np.ndarray:
    width, height, header = 560, 620, 58
    canvas = np.full((height, width * 2, 3), 255, dtype=np.uint8)
    for index, (label, image) in enumerate((("canonical", original), (page["route"], output))):
        display = _bgr(image)
        scale = min(width / display.shape[1], (height - header) / display.shape[0])
        display = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        left = index * width + (width - display.shape[1]) // 2
        top = header + (height - header - display.shape[0]) // 2
        canvas[top:top + display.shape[0], left:left + display.shape[1]] = display
        cv2.putText(canvas, label, (index * width + 10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 1, cv2.LINE_AA)
    title = f"Page {page['global_ordinal']} | {page['archetype']} | {page['decision_before']} -> {page['decision_after']}"
    cv2.putText(canvas, title, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def _derive_result(config: dict[str, Any], pages: list[dict[str, Any]], held_out_population_page_count: int) -> tuple[dict[str, Any], dict[str, bool], str, str]:
    _validate_config(config)
    candidates = [page for page in pages if page["route"] == "apply"]
    protected = [page for page in pages if page["archetype"] in PROTECTED_ARCHETYPES]
    reductions = [float((page.get("method_result") or {}).get("background_span_reduction_fraction") or 0.0) for page in candidates]
    aggregate = {
        "validation_pages": len(pages),
        "held_out_population_page_count": held_out_population_page_count,
        "held_out_population_fraction": round(len(pages) / max(1, held_out_population_page_count), 6),
        "held_out_correction_candidates": len(candidates),
        "safe_correction_candidates": sum(bool((page.get("method_result") or {}).get("safe")) for page in candidates),
        "mean_candidate_background_reduction_fraction": round(float(np.mean(reductions)), 6) if reductions else 0.0,
        "routed_preserve_pages": sum(page["route"] == "preserve" for page in pages),
        "protected_control_pages": len(protected),
        "protected_controls_preserved": sum(page["route"] == "preserve" and page["source_pixel_sha256"] == page["output_pixel_sha256"] for page in protected),
    }
    gates_cfg = config.get("gates") or {}
    gates = {
        "held_out_population_coverage": aggregate["held_out_population_fraction"] >= float(gates_cfg.get("minimum_held_out_population_fraction") or 0.0),
        "held_out_candidate_count": len(candidates) >= int(gates_cfg.get("minimum_held_out_correction_candidates") or 0),
        "every_candidate_safe": not candidates or all(bool((page.get("method_result") or {}).get("safe")) for page in candidates),
        "candidate_background_improvement": bool(candidates) and aggregate["mean_candidate_background_reduction_fraction"] >= float(gates_cfg.get("minimum_mean_candidate_background_reduction_fraction") or 0.0),
        "every_protected_control_preserved": all(page["route"] == "preserve" and page["source_pixel_sha256"] == page["output_pixel_sha256"] for page in protected),
    }
    if all(gates.values()):
        return aggregate, gates, "integration-candidate", "prepare-integration"
    if not gates["held_out_population_coverage"]:
        return aggregate, gates, "more-evidence-required", "expand-validation"
    if not gates["held_out_candidate_count"]:
        return aggregate, gates, "integration-not-justified", "preserve"
    return aggregate, gates, "validation-failed", "preserve"


def evaluate(
    image_root: Path,
    normalization_path: Path,
    photometric_path: Path,
    method_assessment_path: Path,
    method_policy_path: Path,
    sample_path: Path,
    config_path: Path,
    output: Path,
) -> dict[str, Any]:
    normalization = _read_json(normalization_path)
    photometric = _read_json(photometric_path)
    method_assessment = _read_json(method_assessment_path)
    method_policy = _read_json(method_policy_path)
    sample = _read_json(sample_path)
    config = _read_json(config_path)
    expected_sample = prepare_sample(normalization, photometric, method_assessment, method_policy, config)
    if sample != expected_sample:
        raise ValueError("Photometric validation sample does not match authoritative evidence")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Photometric validation output is not empty: {output}")
    contacts = output / "contact-sheets"
    transformed = output / "candidate-outputs"
    contacts.mkdir(parents=True, exist_ok=True)
    transformed.mkdir(parents=True, exist_ok=True)
    method = _validate_method_inputs(method_assessment, method_policy, str(normalization.get("canonical_result_identity") or ""))
    method_config = method_assessment["config"]
    photometric_config = photometric["config"]
    pages = []
    for planned in sample["pages"]:
        ordinal = int(planned["global_ordinal"])
        source = _find_image(image_root, ordinal)
        original = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if original is None:
            raise ValueError(f"Could not decode held-out page {ordinal}")
        condition_before = estimate_photometric_condition(original, photometric_config)
        apply = condition_before["decision"] == "correction-candidate" and condition_before["archetype"] == "paper-page"
        page: dict[str, Any] = {
            "global_ordinal": ordinal,
            "sample_reasons": planned["reasons"],
            "source_sha256": _sha256(source),
            "source_pixel_sha256": _pixel_sha256(original),
            "archetype": condition_before["archetype"],
            "decision_before": condition_before["decision"],
            "route": "apply" if apply else "preserve",
            "condition_before": condition_before,
        }
        if apply:
            result_image = apply_method(original, method, method_config["background_field"])
            method_result = evaluate_variant(original, result_image, condition_before, photometric_config, method_config["safety_gates"])
            target = transformed / f"fs_{ordinal:04d}.png"
            if not cv2.imwrite(str(target), result_image, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
                raise ValueError(f"Could not write held-out candidate page {ordinal}")
            round_trip = cv2.imread(str(target), cv2.IMREAD_UNCHANGED)
            if round_trip is None or not np.array_equal(round_trip, result_image):
                raise ValueError(f"Held-out candidate page {ordinal} failed lossless round-trip validation")
            page["method_result"] = method_result
            page["output_file"] = target.relative_to(output).as_posix()
            page["decision_after"] = method_result["condition_after"]["decision"]
        else:
            result_image = original
            page["method_result"] = None
            page["output_file"] = None
            page["decision_after"] = condition_before["decision"]
        page["output_pixel_sha256"] = _pixel_sha256(result_image)
        review_artifact = apply or condition_before["archetype"] in PROTECTED_ARCHETYPES or any(reason.startswith("known-") for reason in planned["reasons"])
        page["review_artifact"] = f"contact-sheets/fs_{ordinal:04d}.jpg" if review_artifact else None
        if review_artifact:
            sheet = _contact_sheet(original, result_image, page)
            if not cv2.imwrite(str(contacts / f"fs_{ordinal:04d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise ValueError(f"Could not write held-out contact sheet for page {ordinal}")
        pages.append(page)

    held_out_population_page_count = int(sample["held_out_population_page_count"])
    aggregate, gates, status, action = _derive_result(config, pages, held_out_population_page_count)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "validation_type": VALIDATION_TYPE,
        "status": "diagnostic-only",
        "canonical_normalization_result_identity": normalization.get("canonical_result_identity"),
        "photometric_assessment_identity": photometric.get("assessment_identity"),
        "method_assessment_identity": method_assessment.get("assessment_identity"),
        "method_policy_identity": method_policy.get("policy_identity"),
        "sample_identity": sample.get("sample_identity"),
        "held_out_population_page_count": held_out_population_page_count,
        "config": config,
        "method": method,
        "aggregate": aggregate,
        "gates": gates,
        "recommendation_status": status,
        "recommendation_action": action,
        "pages": pages,
    }
    payload["validation_identity"] = canonical_hash({key: payload[key] for key in VALIDATION_IDENTITY_FIELDS})
    output.joinpath("validation.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with output.joinpath("validation.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("global_ordinal", "archetype", "decision_before", "route", "decision_after", "safe", "background_reduction", "detail_correlation", "source_pixel_sha256", "output_pixel_sha256")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for page in pages:
            result = page.get("method_result") or {}
            writer.writerow({
                "global_ordinal": page["global_ordinal"], "archetype": page["archetype"],
                "decision_before": page["decision_before"], "route": page["route"], "decision_after": page["decision_after"],
                "safe": result.get("safe", ""), "background_reduction": result.get("background_span_reduction_fraction", ""),
                "detail_correlation": result.get("high_frequency_correlation", ""),
                "source_pixel_sha256": page["source_pixel_sha256"], "output_pixel_sha256": page["output_pixel_sha256"],
            })
    summary = _summary(payload)
    output.joinpath("summary.md").write_text(summary, encoding="utf-8")
    cards = "".join(f'<article><h2>Page {page["global_ordinal"]}: {page["route"]}</h2><img loading="lazy" src="{page["review_artifact"]}"></article>' for page in pages if page["review_artifact"])
    output.joinpath("index.html").write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>HTH Photometric Method Validation</title><style>body{{font-family:system-ui;background:#111820;color:#e6edf3;margin:2rem}}article{{margin:2rem 0}}img{{max-width:100%}}</style></head><body><h1>HTH Photometric Method Validation</h1>{cards}</body></html>', encoding="utf-8")
    return payload


def _summary(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# HTH Photometric Method Held-Out Validation", "",
        "> Diagnostic validation only. Canonical normalized pixels were not changed.", "",
        f"- Method: `{payload['method']['id']}`",
        f"- Validation pages: `{aggregate['validation_pages']}`",
        f"- Held-out population coverage: `{aggregate['validation_pages']}/{aggregate['held_out_population_page_count']}`",
        f"- Held-out correction candidates: `{aggregate['held_out_correction_candidates']}`",
        f"- Safe correction candidates: `{aggregate['safe_correction_candidates']}`",
        f"- Mean candidate background reduction: `{aggregate['mean_candidate_background_reduction_fraction']:.3f}`",
        f"- Protected controls preserved: `{aggregate['protected_controls_preserved']}/{aggregate['protected_control_pages']}`", "",
        "## Automated gates", "",
    ]
    lines.extend(f"- {'Passed' if value else 'Failed'}: `{name}`" for name, value in payload["gates"].items())
    lines.extend(["", "## Recommendation", "", f"- Status: `{payload['recommendation_status']}`", f"- Action: `{payload['recommendation_action']}`", "", "Open `index.html` to inspect every held-out route and candidate output.", ""])
    return "\n".join(lines)


def recommend(validation_path: Path, output_path: Path) -> dict[str, Any]:
    validation = _read_json(validation_path)
    if validation.get("schema_version") != SCHEMA_VERSION or validation.get("validation_type") != VALIDATION_TYPE or validation.get("status") != "diagnostic-only":
        raise ValueError("Unsupported photometric method validation")
    expected = canonical_hash({key: validation[key] for key in VALIDATION_IDENTITY_FIELDS})
    if validation.get("validation_identity") != expected:
        raise ValueError("Photometric method validation identity does not match its contents")
    aggregate, gates, status, action = _derive_result(validation["config"], validation["pages"], int(validation["held_out_population_page_count"]))
    if validation.get("aggregate") != aggregate or validation.get("gates") != gates or validation.get("recommendation_status") != status or validation.get("recommendation_action") != action:
        raise ValueError("Photometric method validation recommendation does not match page evidence")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "policy_type": "photometric-method-integration",
        "policy_id": (validation.get("config") or {}).get("recommendation", {}).get("policy_id"),
        "status": status,
        "action": action,
        "method": validation.get("method"),
        "compatibility": {
            "canonical_normalization_result_identity": validation.get("canonical_normalization_result_identity"),
            "photometric_assessment_identity": validation.get("photometric_assessment_identity"),
            "method_assessment_identity": validation.get("method_assessment_identity"),
            "method_policy_identity": validation.get("method_policy_identity"),
        },
        "evidence": {"validation_identity": validation.get("validation_identity"), "gates": gates, "aggregate": aggregate},
    }
    payload["policy_identity"] = canonical_hash(payload)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("normalization-manifest", "photometric-assessment", "method-assessment", "method-policy", "config", "output"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    validation = commands.add_parser("evaluate")
    for name in ("image-root", "normalization-manifest", "photometric-assessment", "method-assessment", "method-policy", "sample-plan", "config", "output"):
        validation.add_argument(f"--{name}", type=Path, required=True)
    validation.add_argument("--github-summary", type=Path)
    policy = commands.add_parser("recommend")
    policy.add_argument("--validation", type=Path, required=True)
    policy.add_argument("--output", type=Path, required=True)
    policy.add_argument("--github-summary", type=Path)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        payload = prepare_sample(_read_json(args.normalization_manifest), _read_json(args.photometric_assessment), _read_json(args.method_assessment), _read_json(args.method_policy), _read_json(args.config))
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    elif args.command == "evaluate":
        payload = evaluate(args.image_root, args.normalization_manifest, args.photometric_assessment, args.method_assessment, args.method_policy, args.sample_plan, args.config, args.output)
        if args.github_summary:
            with args.github_summary.open("a", encoding="utf-8") as handle:
                handle.write(_summary(payload))
    else:
        payload = recommend(args.validation, args.output)
        if args.github_summary:
            with args.github_summary.open("a", encoding="utf-8") as handle:
                handle.write(f"\n## Photometric integration decision\n\n- Status: `{payload['status']}`\n- Action: `{payload['action']}`\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
