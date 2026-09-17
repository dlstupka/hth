#!/usr/bin/env python3
"""Assess whether canonical normalized pages need photometric correction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from hth.canonical_build_evidence import canonical_hash
from hth.normalize_document_images import _find_image, _pixel_sha256, _sha256


SCHEMA_VERSION = "1.0"
ASSESSMENT_TYPE = "photometric-uniformity-comparison"


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


def estimate_photometric_condition(image: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """Measure background uniformity, usable tonal range, clipping, and color variation."""
    cfg = config.get("estimator") or {}
    color = _bgr(image)
    maximum_dimension = int(cfg.get("maximum_analysis_dimension") or 1400)
    scale = min(1.0, maximum_dimension / max(color.shape[:2]))
    if scale < 1.0:
        color = cv2.resize(color, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0].astype(np.float32) / 255.0
    rows = int(cfg.get("grid_rows") or 8)
    columns = int(cfg.get("grid_columns") or 8)
    minimum_pixels = int(cfg.get("minimum_tile_pixels") or 64)
    percentile = float(cfg.get("background_percentile") or 85.0)
    tile_luminance: list[float] = []
    tile_chroma: list[tuple[float, float]] = []
    for row in range(rows):
        top, bottom = round(row * luminance.shape[0] / rows), round((row + 1) * luminance.shape[0] / rows)
        for column in range(columns):
            left, right = round(column * luminance.shape[1] / columns), round((column + 1) * luminance.shape[1] / columns)
            tile = luminance[top:bottom, left:right]
            if tile.size < minimum_pixels:
                continue
            threshold = float(np.percentile(tile, percentile))
            bright = tile >= threshold
            if int(np.count_nonzero(bright)) < max(8, minimum_pixels // 10):
                continue
            tile_luminance.append(float(np.median(tile[bright])))
            a = lab[top:bottom, left:right, 1][bright].astype(np.float32) - 128.0
            b = lab[top:bottom, left:right, 2][bright].astype(np.float32) - 128.0
            tile_chroma.append((float(np.median(a)), float(np.median(b))))

    valid_tiles = len(tile_luminance)
    minimum_tiles = int(cfg.get("minimum_valid_tiles") or 32)
    if valid_tiles:
        full_frame_background_span = float(np.percentile(tile_luminance, 90) - np.percentile(tile_luminance, 10))
    else:
        full_frame_background_span = 0.0
    boundary_trim = max(0, int(cfg.get("background_boundary_trim_tiles") or 1))
    complete_grid = valid_tiles == rows * columns
    if complete_grid and rows > 2 * boundary_trim and columns > 2 * boundary_trim:
        background_grid = np.asarray(tile_luminance, dtype=np.float32).reshape(rows, columns)
        interior = background_grid[
            boundary_trim:rows - boundary_trim,
            boundary_trim:columns - boundary_trim,
        ].reshape(-1)
        background_span = float(np.percentile(interior, 90) - np.percentile(interior, 10))
    else:
        # An incomplete grid cannot prove that variation is confined to framing.
        # Fail conservatively by retaining the full-frame measurement.
        background_span = full_frame_background_span
    if tile_chroma:
        chroma = np.asarray(tile_chroma, dtype=np.float32)
        chroma_variation = float(np.hypot(
            np.percentile(chroma[:, 0], 90) - np.percentile(chroma[:, 0], 10),
            np.percentile(chroma[:, 1], 90) - np.percentile(chroma[:, 1], 10),
        ))
    else:
        chroma_variation = 0.0
    tonal_span = float(np.percentile(luminance, 98) - np.percentile(luminance, 2))
    edge_fraction = float(np.mean(cv2.Canny(np.uint8(luminance * 255.0), 60, 180) > 0))
    shadow_clipping = float(np.mean(luminance <= (2.0 / 255.0)))
    highlight_clipping = float(np.mean(luminance >= (253.0 / 255.0)))

    dark_frame_minimum = float(cfg.get("dark_frame_minimum_shadow_fraction") or 0.75)
    mixed_polarity_minimum = float(cfg.get("mixed_polarity_minimum_shadow_fraction") or 0.25)
    if shadow_clipping >= dark_frame_minimum:
        archetype = "dark-polarity-frame"
    elif shadow_clipping >= mixed_polarity_minimum:
        archetype = "mixed-polarity-page"
    else:
        archetype = "paper-page"

    preserve_checks = {
        "background_uniformity": background_span <= float(cfg.get("preserve_maximum_background_span") or 0.10),
        "tonal_span": tonal_span >= float(cfg.get("preserve_minimum_tonal_span") or 0.24),
        "shadow_clipping": shadow_clipping <= float(cfg.get("preserve_maximum_shadow_clipping_fraction") or 0.005),
        "highlight_clipping": highlight_clipping <= float(cfg.get("preserve_maximum_highlight_clipping_fraction") or 0.02),
        "background_chroma_uniformity": chroma_variation <= float(cfg.get("preserve_maximum_background_chroma_variation") or 8.0),
    }
    candidate_reasons = []
    if archetype == "paper-page" and background_span >= float(cfg.get("candidate_minimum_background_span") or 0.18):
        candidate_reasons.append("uneven-background")
    if archetype == "paper-page" and tonal_span <= float(cfg.get("candidate_maximum_tonal_span") or 0.14) and edge_fraction >= 0.01:
        candidate_reasons.append("compressed-tonal-range")
    # Dense black ink and bright paper legitimately occupy the luminance endpoints.
    # Clipping remains review evidence, but is not independently sufficient to
    # recommend changing archival pixels.
    if archetype == "paper-page" and chroma_variation >= float(cfg.get("candidate_minimum_background_chroma_variation") or 16.0):
        candidate_reasons.append("uneven-color-cast")

    boundary_geometry = (
        complete_grid
        and full_frame_background_span >= float(cfg.get("boundary_geometry_minimum_full_frame_span") or 0.18)
        and background_span < float(cfg.get("candidate_minimum_background_span") or 0.18)
    )

    if valid_tiles < minimum_tiles:
        decision = "inconclusive"
    elif archetype == "dark-polarity-frame":
        decision = "preserve"
    elif archetype == "mixed-polarity-page":
        decision = "review"
    elif candidate_reasons:
        decision = "correction-candidate"
    elif boundary_geometry:
        decision = "preserve"
    elif all(preserve_checks.values()):
        decision = "preserve"
    else:
        decision = "review"
    if archetype == "dark-polarity-frame":
        decision_reasons = ["intentional-dark-polarity"]
    elif archetype == "mixed-polarity-page":
        decision_reasons = ["mixed-polarity-content"]
    elif candidate_reasons:
        decision_reasons = candidate_reasons
    elif boundary_geometry:
        decision_reasons = ["boundary-dominated-background-geometry"]
    elif decision == "review":
        decision_reasons = ["threshold-review"]
    else:
        decision_reasons = ["photometric-gates-passed"]
    severity = max(
        background_span / max(0.001, float(cfg.get("candidate_minimum_background_span") or 0.18)),
        max(0.0, float(cfg.get("preserve_minimum_tonal_span") or 0.24) - tonal_span) / 0.24,
        shadow_clipping / max(0.001, float(cfg.get("candidate_minimum_shadow_clipping_fraction") or 0.025)),
        highlight_clipping / max(0.001, float(cfg.get("candidate_minimum_highlight_clipping_fraction") or 0.08)),
        chroma_variation / max(0.001, float(cfg.get("candidate_minimum_background_chroma_variation") or 16.0)),
    )
    return {
        "decision": decision,
        "archetype": archetype,
        "candidate_reasons": candidate_reasons,
        "decision_reasons": decision_reasons,
        "valid_tile_count": valid_tiles,
        "background_luminance_span": round(background_span, 6),
        "full_frame_background_luminance_span": round(full_frame_background_span, 6),
        "boundary_geometry_detected": boundary_geometry,
        "tonal_span": round(tonal_span, 6),
        "edge_fraction": round(edge_fraction, 6),
        "shadow_clipping_fraction": round(shadow_clipping, 6),
        "highlight_clipping_fraction": round(highlight_clipping, 6),
        "background_chroma_variation": round(chroma_variation, 6),
        "severity": round(severity, 6),
        "preserve_checks": preserve_checks,
        "background_grid": [round(value, 6) for value in tile_luminance],
        "background_grid_shape": [rows, columns],
    }


def _contact_sheet(image: np.ndarray, result: dict[str, Any], ordinal: int) -> np.ndarray:
    display = _bgr(image)
    scale = min(1.0, 820 / display.shape[1], 720 / display.shape[0])
    display = cv2.resize(display, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    values = np.asarray(result["background_grid"], dtype=np.float32)
    rows, columns = result["background_grid_shape"]
    if values.size == rows * columns:
        grid = values.reshape(rows, columns)
        heat = cv2.resize(np.uint8(np.clip(grid * 255.0, 0, 255)), (display.shape[1], display.shape[0]), interpolation=cv2.INTER_NEAREST)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    else:
        heat = np.full_like(display, 220)
    canvas = np.full((display.shape[0] + 90, display.shape[1] * 2 + 12, 3), 255, dtype=np.uint8)
    canvas[90:, :display.shape[1]] = display
    canvas[90:, display.shape[1] + 12:] = heat
    label = (
        f"Page {ordinal}: {result['decision']} ({result['archetype']}) | background span {result['background_luminance_span']:.3f} | "
        f"tonal span {result['tonal_span']:.3f}"
    )
    cv2.putText(canvas, label, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Canonical normalized page", (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Bright-background tile map", (display.shape[1] + 24, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def _summary(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# HTH Photometric Assessment", "",
        "> Diagnostic evidence only. No illumination, color, contrast, or tonal transform was applied.", "",
        "## Result", "",
        f"- Collection: `{payload['collection']['id']}`",
        f"- Population: `{payload['population_page_count']}` pages",
        f"- Stratified sample: `{payload['sample_page_count']}` pages",
        f"- Preserve evidence: `{aggregate['preserve_pages']}` pages",
        f"- Correction candidates: `{aggregate['correction_candidate_pages']}` pages",
        f"- Dark-polarity frames preserved: `{aggregate['dark_polarity_frame_pages']}` pages",
        f"- Mixed-polarity pages held for review: `{aggregate['mixed_polarity_pages']}` pages",
        f"- Review: `{aggregate['review_pages']}` pages",
        f"- Inconclusive: `{aggregate['inconclusive_pages']}` pages", "",
        "## Highest-priority visual review", "",
        "| Page | Archetype | Decision | Background span | Tonal span | Reasons |",
        "|---:|---|---|---:|---:|---|",
    ]
    for page in payload["priority_review_pages"]:
        lines.append(
            f"| {page['global_ordinal']} | {page['archetype']} | {page['decision']} | {page['background_span']:.3f} | "
            f"{page['tonal_span']:.3f} | {', '.join(page['reasons']) or 'threshold review'} |"
        )
    lines.extend(["", "Download and extract the review artifact, then open `index.html` locally to inspect the sampled pages and background maps.", ""])
    return "\n".join(lines)


def assess(image_root: Path, manifest_path: Path, sample_path: Path, config_path: Path, output: Path) -> dict[str, Any]:
    manifest, sample, config = _read_json(manifest_path), _read_json(sample_path), _read_json(config_path)
    if sample.get("canonical_normalization_result_identity") != manifest.get("canonical_result_identity"):
        raise ValueError("Photometric sample does not match the canonical normalization result")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Photometric assessment output is not empty: {output}")
    contacts = output / "contact-sheets"
    contacts.mkdir(parents=True, exist_ok=True)
    reasons = {int(page["global_ordinal"]): page.get("reasons") or [] for page in sample.get("pages") or []}
    pages = []
    for ordinal in sorted(reasons):
        source = _find_image(image_root, ordinal)
        image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not decode canonical normalized page {ordinal}")
        estimate = estimate_photometric_condition(image, config)
        if not cv2.imwrite(str(contacts / f"fs_{ordinal:04d}.jpg"), _contact_sheet(image, estimate, ordinal), [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise ValueError(f"Could not write photometric contact sheet for page {ordinal}")
        pages.append({
            "global_ordinal": ordinal,
            "sample_reasons": reasons[ordinal],
            "source_sha256": _sha256(source),
            "source_pixel_sha256": _pixel_sha256(image),
            "estimate": estimate,
        })
    decisions = ("preserve", "correction-candidate", "review", "inconclusive")
    counts = {name: sum(page["estimate"]["decision"] == name for page in pages) for name in decisions}
    priority = sorted(pages, key=lambda page: -float(page["estimate"]["severity"]))[:20]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "assessment_type": ASSESSMENT_TYPE,
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
            "dark_polarity_frame_pages": sum(page["estimate"]["archetype"] == "dark-polarity-frame" for page in pages),
            "mixed_polarity_pages": sum(page["estimate"]["archetype"] == "mixed-polarity-page" for page in pages),
        },
        "priority_review_pages": [{
            "global_ordinal": page["global_ordinal"],
            "archetype": page["estimate"]["archetype"],
            "decision": page["estimate"]["decision"],
            "background_span": page["estimate"]["background_luminance_span"],
            "tonal_span": page["estimate"]["tonal_span"],
            "reasons": page["estimate"]["decision_reasons"],
        } for page in priority],
        "pages": pages,
    }
    identity_fields = ("assessment_type", "canonical_normalization_result_identity", "sample_identity", "config", "pages")
    payload["assessment_identity"] = canonical_hash({key: payload[key] for key in identity_fields})
    (output / "assessment.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (output / "assessment.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("global_ordinal", "archetype", "decision", "background_span", "tonal_span", "shadow_clipping", "highlight_clipping", "chroma_variation")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for page in pages:
            estimate = page["estimate"]
            writer.writerow({
                "global_ordinal": page["global_ordinal"], "archetype": estimate["archetype"], "decision": estimate["decision"],
                "background_span": estimate["background_luminance_span"], "tonal_span": estimate["tonal_span"],
                "shadow_clipping": estimate["shadow_clipping_fraction"], "highlight_clipping": estimate["highlight_clipping_fraction"],
                "chroma_variation": estimate["background_chroma_variation"],
            })
    summary = _summary(payload)
    (output / "summary.md").write_text(summary, encoding="utf-8")
    cards = "".join(f'<article><h2>Page {page["global_ordinal"]}</h2><img loading="lazy" src="contact-sheets/fs_{page["global_ordinal"]:04d}.jpg"></article>' for page in pages)
    (output / "index.html").write_text(
        f'<!doctype html><html><head><meta charset="utf-8"><title>HTH Photometric Assessment</title><style>body{{font-family:system-ui;background:#111820;color:#e6edf3;margin:2rem}}article{{margin:2rem 0}}img{{max-width:100%}}</style></head><body><h1>HTH Photometric Assessment</h1><p>Diagnostic evidence only; no pixels were changed.</p>{cards}</body></html>',
        encoding="utf-8",
    )
    return payload


def recommend(assessment_path: Path, output_path: Path) -> dict[str, Any]:
    assessment = _read_json(assessment_path)
    if assessment.get("schema_version") != SCHEMA_VERSION or assessment.get("assessment_type") != ASSESSMENT_TYPE:
        raise ValueError("Unsupported photometric assessment")
    if assessment.get("status") != "diagnostic-only":
        raise ValueError("Photometric assessment is not complete diagnostic evidence")
    identity_fields = ("assessment_type", "canonical_normalization_result_identity", "sample_identity", "config", "pages")
    if assessment.get("assessment_identity") != canonical_hash({key: assessment[key] for key in identity_fields}):
        raise ValueError("Photometric assessment identity does not match its contents")
    cfg = (assessment.get("config") or {}).get("recommendation") or {}
    aggregate = assessment.get("aggregate") or {}
    sample_count = int(assessment.get("sample_page_count") or 0)
    candidates = int(aggregate.get("correction_candidate_pages") or 0)
    inconclusive = int(aggregate.get("inconclusive_pages") or 0)
    gates = {
        "sample_size": sample_count >= int(cfg.get("minimum_sample_pages") or 50),
        "candidate_prevalence_below_skip_limit": candidates / max(1, sample_count) <= float(cfg.get("maximum_candidate_fraction_for_skip") or 0.025),
        "inconclusive_prevalence_below_limit": inconclusive / max(1, sample_count) <= float(cfg.get("maximum_inconclusive_fraction_for_skip") or 0.20),
    }
    skip = all(gates.values())
    payload = {
        "schema_version": SCHEMA_VERSION,
        "policy_type": "photometric-normalization",
        "policy_id": str(cfg.get("policy_id") or "photometric-preservation-assessment-v1"),
        "status": "skip-recommended" if skip else "manual-review-required",
        "action": "preserve" if skip else "withhold",
        "plain_language": (
            "Keep the current normalized pixels; the sampled pages do not show prevalent photometric defects that justify resampling."
            if skip else "Do not apply photometric correction automatically; the evidence requires focused review and policy development."
        ),
        "compatibility": {"canonical_normalization_result_identity": assessment.get("canonical_normalization_result_identity")},
        "evidence": {"assessment_identity": assessment.get("assessment_identity"), "gates": gates, "aggregate": aggregate},
    }
    payload["policy_identity"] = canonical_hash(payload)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = "\n".join([
        "## Photometric recommendation", "", f"**{payload['plain_language']}**", "",
        f"- Decision: `{payload['action']}`", f"- Status: `{payload['status']}`", "",
        *[f"- {'Passed' if value else 'Needs review'}: `{name}`" for name, value in gates.items()], "",
    ])
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
        rendered = _summary(payload)
    else:
        recommend(args.assessment, args.output)
        rendered = (args.output.parent / "recommendation.md").read_text(encoding="utf-8")
    if args.github_summary:
        with args.github_summary.open("a", encoding="utf-8") as handle:
            handle.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
