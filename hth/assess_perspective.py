#!/usr/bin/env python3
"""Assess whether canonical normalized pages need projective correction."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hth.canonical_build_evidence import canonical_hash
from hth.normalize_document_images import _find_image, _pixel_sha256, _sha256


SCHEMA_VERSION = "1.0"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _family_fit(samples: list[tuple[float, float, float]], cfg: dict[str, Any]) -> dict[str, Any]:
    minimum_lines = int(cfg["minimum_line_count"])
    if len(samples) < minimum_lines:
        return {"line_count": len(samples), "convergence_degrees": 0.0, "confidence": 0.0}
    positions = np.asarray([item[0] for item in samples], dtype=np.float64)
    angles = np.asarray([item[1] for item in samples], dtype=np.float64)
    weights = np.asarray([item[2] for item in samples], dtype=np.float64)
    median = float(np.median(angles))
    keep = np.abs(angles - median) <= float(cfg["maximum_family_deviation_degrees"])
    positions, angles, weights = positions[keep], angles[keep], weights[keep]
    if len(positions) < minimum_lines:
        return {"line_count": len(positions), "convergence_degrees": 0.0, "confidence": 0.0}
    design = np.column_stack((np.ones(len(positions)), positions))
    weighted = np.sqrt(weights)
    intercept, slope = np.linalg.lstsq(design * weighted[:, None], angles * weighted, rcond=None)[0]
    residuals = angles - (intercept + slope * positions)
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    bins = len(set(np.clip((positions * 5).astype(int), 0, 4)))
    line_factor = min(1.0, len(positions) / float(cfg["full_confidence_line_count"]))
    coverage_factor = min(1.0, bins / 4.0)
    dispersion_factor = max(0.0, 1.0 - mad / float(cfg["maximum_family_deviation_degrees"]))
    return {
        "line_count": int(len(positions)),
        "convergence_degrees": round(abs(float(slope)), 6),
        "confidence": round(line_factor * coverage_factor * dispersion_factor, 6),
        "residual_mad_degrees": round(mad, 6),
    }


def estimate_line_convergence(image: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Measure convergence of independently detected horizontal and vertical line families."""
    cfg = config.get("estimator") or {}
    gray = _gray(image)
    maximum_dimension = int(cfg.get("maximum_analysis_dimension") or 1400)
    scale = min(1.0, maximum_dimension / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(gray, int(cfg.get("canny_low") or 60), int(cfg.get("canny_high") or 180))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 720.0,
        threshold=max(30, int(gray.shape[1] * 0.025)),
        minLineLength=max(30, int(gray.shape[1] * float(cfg.get("minimum_line_length_fraction") or 0.08))),
        maxLineGap=max(8, int(gray.shape[1] * 0.015)),
    )
    horizontal: list[tuple[float, float, float]] = []
    vertical: list[tuple[float, float, float]] = []
    maximum_angle = float(cfg.get("maximum_axis_angle_degrees") or 15.0)
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            dx, dy = float(x2 - x1), float(y2 - y1)
            length = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx))
            while angle <= -90.0:
                angle += 180.0
            while angle > 90.0:
                angle -= 180.0
            if abs(angle) <= maximum_angle:
                horizontal.append((((y1 + y2) / 2.0) / gray.shape[0], angle, length))
            if abs(abs(angle) - 90.0) <= maximum_angle:
                vertical_angle = angle - 90.0 if angle > 0 else angle + 90.0
                vertical.append((((x1 + x2) / 2.0) / gray.shape[1], vertical_angle, length))
    fit_config = {
        "minimum_line_count": int(cfg.get("minimum_line_count") or 12),
        "full_confidence_line_count": int(cfg.get("full_confidence_line_count") or 40),
        "maximum_family_deviation_degrees": float(cfg.get("maximum_family_deviation_degrees") or 3.0),
    }
    families = {
        "horizontal": _family_fit(horizontal, fit_config),
        "vertical": _family_fit(vertical, fit_config),
    }
    confident = [value for value in families.values() if value["confidence"] >= float(cfg.get("minimum_confidence") or 0.6)]
    convergence = max((float(value["convergence_degrees"]) for value in confident), default=0.0)
    required_families = int(cfg.get("required_confirming_families") or 2)
    confirmed_convergence = (
        min(float(value["convergence_degrees"]) for value in confident)
        if len(confident) >= required_families else 0.0
    )
    confidence = max((float(value["confidence"]) for value in families.values()), default=0.0)
    preserve_maximum = float(cfg.get("preserve_maximum_degrees") or 0.35)
    correction_minimum = float(cfg.get("correction_minimum_degrees") or 0.75)
    if not confident:
        decision = "inconclusive"
    elif len(confident) >= required_families and confirmed_convergence >= correction_minimum:
        decision = "correction-candidate"
    elif convergence <= preserve_maximum:
        decision = "preserve"
    else:
        decision = "review"
    return {
        "decision": decision,
        "maximum_convergence_degrees": round(convergence, 6),
        "confirmed_convergence_degrees": round(confirmed_convergence, 6),
        "confirming_family_count": len(confident),
        "confidence": round(confidence, 6),
        "families": families,
    }


def _display(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image[:, :, :3].copy()


def _contact_sheet(image: np.ndarray, result: dict[str, Any], ordinal: int) -> np.ndarray:
    display = _display(image)
    scale = min(1.0, 1000 / display.shape[1], 760 / display.shape[0])
    display = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((display.shape[0] + 70, display.shape[1], 3), 255, dtype=np.uint8)
    canvas[70:] = display
    text = f"Page {ordinal}: {result['decision']} | convergence {result['maximum_convergence_degrees']:.3f} deg | confidence {result['confidence']:.3f}"
    cv2.putText(canvas, text, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def _summary(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# HTH Perspective Assessment",
        "",
        "> Diagnostic evidence only. No projective transform was applied.",
        "",
        "## Result",
        "",
        f"- Collection: `{payload['collection']['id']}`",
        f"- Population: `{payload['population_page_count']}` pages",
        f"- Stratified sample: `{payload['sample_page_count']}` pages",
        f"- Preserve evidence: `{aggregate['preserve_pages']}` pages",
        f"- Correction candidates: `{aggregate['correction_candidate_pages']}` pages",
        f"- Review: `{aggregate['review_pages']}` pages",
        f"- Inconclusive: `{aggregate['inconclusive_pages']}` pages",
        f"- Maximum observed single-family convergence: `{aggregate['maximum_convergence_degrees']:.3f} deg`",
        "",
        "## Highest-priority visual review",
        "",
        "| Page | Decision | Convergence | Confidence |",
        "|---:|---|---:|---:|",
    ]
    for page in payload["priority_review_pages"]:
        lines.append(f"| {page['global_ordinal']} | {page['decision']} | {page['convergence_degrees']:.3f} deg | {page['confidence']:.3f} |")
    lines.extend(["", "Open [index.html](index.html) from the artifact to inspect the sampled pages.", ""])
    return "\n".join(lines)


def assess(image_root: Path, manifest_path: Path, sample_path: Path, config_path: Path, output: Path) -> dict[str, Any]:
    manifest, sample, config = _read_json(manifest_path), _read_json(sample_path), _read_json(config_path)
    if sample.get("canonical_normalization_result_identity") != manifest.get("canonical_result_identity"):
        raise ValueError("Perspective sample does not match the canonical normalization result")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Perspective assessment output is not empty: {output}")
    contacts = output / "contact-sheets"
    contacts.mkdir(parents=True, exist_ok=True)
    reasons = {int(page["global_ordinal"]): page.get("reasons") or [] for page in sample.get("pages") or []}
    pages = []
    for ordinal in sorted(reasons):
        source = _find_image(image_root, ordinal)
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not decode canonical normalized page {ordinal}")
        estimate = estimate_line_convergence(image, config)
        sheet = _contact_sheet(image, estimate, ordinal)
        if not cv2.imwrite(str(contacts / f"fs_{ordinal:04d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ValueError(f"Could not write perspective contact sheet for page {ordinal}")
        pages.append({
            "global_ordinal": ordinal,
            "sample_reasons": reasons[ordinal],
            "source_sha256": _sha256(source),
            "source_pixel_sha256": _pixel_sha256(image),
            "estimate": estimate,
        })
    counts = {name: sum(page["estimate"]["decision"] == name for page in pages) for name in ("preserve", "correction-candidate", "review", "inconclusive")}
    priority = sorted(pages, key=lambda page: (-float(page["estimate"]["maximum_convergence_degrees"]), -float(page["estimate"]["confidence"])))[:20]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "assessment_type": "perspective-convergence-comparison",
        "status": "diagnostic-only",
        "collection": manifest.get("collection"),
        "canonical_normalization_result_identity": manifest.get("canonical_result_identity"),
        "population_page_count": sample.get("population_page_count"),
        "sample_page_count": len(pages),
        "sample_identity": sample.get("sample_identity"),
        "config": config,
        "aggregate": {
            "preserve_pages": counts["preserve"],
            "correction_candidate_pages": counts["correction-candidate"],
            "review_pages": counts["review"],
            "inconclusive_pages": counts["inconclusive"],
            "maximum_convergence_degrees": max(float(page["estimate"]["maximum_convergence_degrees"]) for page in pages),
        },
        "priority_review_pages": [{
            "global_ordinal": page["global_ordinal"],
            "decision": page["estimate"]["decision"],
            "convergence_degrees": page["estimate"]["maximum_convergence_degrees"],
            "confidence": page["estimate"]["confidence"],
        } for page in priority],
        "pages": pages,
    }
    payload["assessment_identity"] = canonical_hash({key: payload[key] for key in ("assessment_type", "canonical_normalization_result_identity", "sample_identity", "config", "pages")})
    (output / "assessment.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (output / "assessment.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("global_ordinal", "decision", "convergence_degrees", "confidence"))
        writer.writeheader()
        for page in pages:
            writer.writerow({"global_ordinal": page["global_ordinal"], "decision": page["estimate"]["decision"], "convergence_degrees": page["estimate"]["maximum_convergence_degrees"], "confidence": page["estimate"]["confidence"]})
    summary = _summary(payload)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    cards = "".join(f'<article><h2>Page {page["global_ordinal"]}</h2><img loading="lazy" src="contact-sheets/fs_{page["global_ordinal"]:04d}.jpg"></article>' for page in pages)
    (output / "index.html").write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>HTH Perspective Assessment</title><style>body{{font-family:system-ui;background:#111820;color:#e6edf3;margin:2rem}}article{{margin:2rem 0}}img{{max-width:100%}}</style></head><body><h1>HTH Perspective Assessment</h1><p>Diagnostic evidence only; no pixels were changed.</p>{cards}</body></html>', encoding="utf-8")
    return payload


def recommend(assessment_path: Path, output_path: Path) -> dict[str, Any]:
    assessment = _read_json(assessment_path)
    if assessment.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported perspective assessment schema")
    if assessment.get("assessment_type") != "perspective-convergence-comparison":
        raise ValueError("Not a perspective convergence assessment")
    if assessment.get("status") != "diagnostic-only":
        raise ValueError("Perspective assessment is not complete diagnostic evidence")
    expected_identity = canonical_hash({
        key: assessment[key]
        for key in (
            "assessment_type",
            "canonical_normalization_result_identity",
            "sample_identity",
            "config",
            "pages",
        )
    })
    if assessment.get("assessment_identity") != expected_identity:
        raise ValueError("Perspective assessment identity does not match its contents")
    recommendation = (assessment.get("config") or {}).get("recommendation") or {}
    aggregate = assessment.get("aggregate") or {}
    sample_count = int(assessment.get("sample_page_count") or 0)
    candidate_count = int(aggregate.get("correction_candidate_pages") or 0)
    inconclusive_count = int(aggregate.get("inconclusive_pages") or 0)
    gates = {
        "sample_size": sample_count >= int(recommendation.get("minimum_sample_pages") or 50),
        "candidate_prevalence_below_skip_limit": candidate_count / max(1, sample_count) <= float(recommendation.get("maximum_candidate_fraction_for_skip") or 0.025),
        "inconclusive_prevalence_below_limit": inconclusive_count / max(1, sample_count) <= float(recommendation.get("maximum_inconclusive_fraction_for_skip") or 0.25),
    }
    skip = all(gates.values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "policy_type": "perspective-normalization",
        "policy_id": str(recommendation.get("policy_id") or "projective-convergence-conservative-v1"),
        "status": "skip-recommended" if skip else "manual-review-required",
        "action": "preserve" if skip else "withhold",
        "plain_language": (
            "Keep the current normalized pixels; the sampled pages do not show prevalent, credible projective convergence."
            if skip else
            "Do not apply projective correction automatically; the evidence requires focused review."
        ),
        "compatibility": {"canonical_normalization_result_identity": assessment.get("canonical_normalization_result_identity")},
        "evidence": {"assessment_identity": assessment.get("assessment_identity"), "gates": gates, "aggregate": aggregate},
    }
    payload["policy_identity"] = canonical_hash(payload)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = "\n".join(["## Perspective recommendation", "", f"**{payload['plain_language']}**", "", f"- Decision: `{payload['action']}`", f"- Status: `{payload['status']}`", "", *[f"- {'Passed' if value else 'Needs review'}: `{name}`" for name, value in gates.items()], ""])
    (output_path.parent / "recommendation.md").write_text(summary, encoding="utf-8")
    with (output_path.parent / "summary.md").open("a", encoding="utf-8") as handle:
        handle.write("\n" + summary)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    for name in ("image-root", "normalization-manifest", "sample-plan", "config", "output"):
        evaluate.add_argument(f"--{name}", type=Path, required=True)
    evaluate.add_argument("--github-summary", type=Path)
    recommendation = commands.add_parser("recommend")
    recommendation.add_argument("--assessment", type=Path, required=True)
    recommendation.add_argument("--output", type=Path, required=True)
    recommendation.add_argument("--github-summary", type=Path)
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        payload = assess(args.image_root, args.normalization_manifest, args.sample_plan, args.config, args.output)
        if args.github_summary:
            with args.github_summary.open("a", encoding="utf-8") as handle:
                handle.write(_summary(payload))
    else:
        payload = recommend(args.assessment, args.output)
        if args.github_summary:
            with args.github_summary.open("a", encoding="utf-8") as handle:
                handle.write((args.output.parent / "recommendation.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
