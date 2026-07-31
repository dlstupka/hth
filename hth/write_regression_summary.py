#!/usr/bin/env python3
"""Render a GitHub Actions job summary from a canonical HTH regression run."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any




DETECTOR_CHARACTERIZATION: dict[str, dict[str, Any]] = {
    "components": {
        "friendly_name": "Connected Components",
        "short_name": "Components",
        "role": "Generator",
        "evidence": [("Connected-component envelope", "Primary", "Generates a page-region hypothesis from grouped foreground components."), ("Morphological grouping", "Supporting", "Controls how fragmented marks are joined before envelope extraction.")],
    },
    "consensus_quad": {
        "friendly_name": "Consensus Quadrilateral",
        "short_name": "Consensus Quad",
        "role": "Hybrid (Contour Quad + Edge Contour)",
        "evidence": [("Contour Quad vote", "Primary", "Supplies one geometric quadrilateral hypothesis."), ("Edge Contour vote", "Primary", "Supplies an independently scored edge-supported hypothesis."), ("Polygon agreement", "Decision", "Requires sufficient IoU and corner agreement before fusion.")],
    },
    "contour": {
        "friendly_name": "Contour Envelope",
        "short_name": "Contour",
        "role": "Generator",
        "evidence": [("Contour geometry", "Primary", "Generates page-region hypotheses from thresholded contours."), ("Fragment merging", "Supporting", "Attempts to recover page boundaries split across multiple contours.")],
    },
    "contour_components": {
        "friendly_name": "Contour + Components",
        "short_name": "Contour Components",
        "role": "Hybrid (Contour Quad + Components)",
        "evidence": [("Contour quadrilateral", "Generator", "Produces candidate page quadrilaterals."), ("Component containment", "Validator", "Measures how well selected components fall within each candidate."), ("Component envelope overlap", "Validator", "Compares each contour candidate with the independent component envelope."), ("Component spread and density", "Validator", "Checks whether foreground evidence is distributed plausibly across the candidate.")],
    },
    "contour_projection": {
        "friendly_name": "Contour + Projection",
        "short_name": "Contour Projection",
        "role": "Hybrid (Contour Quad + Projection)",
        "evidence": [("Contour quadrilateral", "Generator", "Produces candidate page quadrilaterals."), ("Horizontal projection profile", "Validator", "Scores text-band structure after candidate normalization."), ("Vertical coverage", "Validator", "Checks whether foreground structure spans the candidate height."), ("Ink density", "Validator", "Rejects implausibly empty or saturated candidate interiors.")],
    },
    "contour_quad": {
        "friendly_name": "Contour Quadrilateral",
        "short_name": "Contour Quad",
        "role": "Generator",
        "evidence": [("Contour quadrilaterals", "Primary", "Generates multiple polygonal page hypotheses."), ("Area", "Scoring", "Rewards candidates occupying a plausible image fraction."), ("Rectangularity", "Scoring", "Rewards quadrilateral-like contour geometry."), ("Corner angles", "Scoring", "Rewards near-right-angle page geometry.")],
    },
    "edge_contour": {
        "friendly_name": "Edge-Supported Contour",
        "short_name": "Edge Contour",
        "role": "Hybrid (Contour Quad + LSD)",
        "evidence": [("Contour quadrilateral", "Generator", "Produces candidate page quadrilaterals."), ("LSD line segments", "Validator", "Independently detects line support near proposed borders."), ("Edge support", "Validator", "Measures border coverage after configurable dilation."), ("Geometry score", "Scoring", "Combines area, rectangularity, and angle quality.")],
    },
    "grabcut": {
        "friendly_name": "GrabCut Segmentation",
        "short_name": "GrabCut",
        "role": "Generator",
        "evidence": [("GrabCut foreground mask", "Primary", "Segments foreground pixels from a border-seeded background model."), ("Morphological cleanup", "Supporting", "Closes and erodes the segmentation before region extraction."), ("Foreground contour", "Geometry", "Converts the segmented region into a page polygon or bounding quadrilateral.")],
    },
    "hough": {
        "friendly_name": "Hough Line Borders",
        "short_name": "Hough",
        "role": "Generator",
        "evidence": [("Hough lines", "Primary", "Generates axis-aligned border hypotheses from detected lines."), ("Outer-line percentile", "Scoring", "Selects outer line groups used to form a page box."), ("Axis-angle tolerance", "Filtering", "Restricts candidate lines to near-horizontal or near-vertical orientations.")],
    },
    "lsd": {
        "friendly_name": "Line Segment Detector",
        "short_name": "LSD",
        "role": "Generator",
        "evidence": [("LSD segments", "Primary", "Generates border hypotheses directly from line segments."), ("Outer-line percentile", "Scoring", "Selects outer segment groups for page-boundary construction."), ("Axis-angle tolerance", "Filtering", "Limits segments to plausible page-border orientations.")],
    },
    "ransac": {
        "friendly_name": "RANSAC Border Fit",
        "short_name": "RANSAC",
        "role": "Generator",
        "evidence": [("Scan foreground samples", "Primary", "Samples likely border evidence along image scans."), ("RANSAC line fitting", "Primary", "Fits robust page-border models while rejecting outliers."), ("Inlier ratio", "Validation", "Requires sufficient support for accepted line models.")],
    },
}


def _detector_characterization(detector: str) -> dict[str, Any]:
    return DETECTOR_CHARACTERIZATION.get(detector, {
        "friendly_name": detector.replace("_", " ").title(),
        "short_name": detector,
        "role": "Unknown",
        "evidence": [("Detector output", "Primary", "Evidence characterization has not yet been registered for this detector.")],
    })


def _detector_friendly_name(detector: str) -> str:
    return str(_detector_characterization(detector).get("friendly_name") or detector)


def _detector_short_name(detector: str) -> str:
    return str(_detector_characterization(detector).get("short_name") or detector)


def _detector_heading(detector: str) -> str:
    return f"{_detector_friendly_name(detector)} (`{detector}`)"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _short(value: Any, length: int = 12) -> str:
    text = str(value or "").strip()
    return text[:length] if text else "unknown"


def _number(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "unknown"


def _duration(seconds: Any) -> str:
    """Format elapsed time using compact engineering units."""
    try:
        value = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "unknown"
    if value < 1:
        return f"{round(value * 1000):.0f} ms"
    if value < 60:
        rendered = f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{rendered}s"

    total_seconds = int(round(value))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def _compact_duration(seconds: Any) -> str:
    """Format stabilization time with the shared duration convention."""
    return _duration(seconds)


def _compact_percent(value: Any) -> str:
    """Format a fraction as a whole-number search percentage."""
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "unknown"


def _pluralized_parameter_sets(value: Any) -> str:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return "unknown parameter sets"
    noun = "parameter set" if count == 1 else "parameter sets"
    return f"{count} {noun}"


def _full_search_metrics(
    search: dict[str, Any],
    elapsed_seconds: Any,
    *,
    page_rate: Any = None,
    page_count: Any = None,
) -> tuple[str, str, str]:
    evaluated = search.get("parameter_sets")
    possible = search.get("possible_parameter_sets")
    try:
        evaluated_count = int(evaluated)
        possible_count = int(possible)
    except (TypeError, ValueError):
        return str(possible or "unknown"), "unknown", "unknown"
    if possible_count <= 0:
        return str(possible_count), "unknown", "unknown"

    evaluated_share = min(1.0, evaluated_count / possible_count)
    if evaluated_count >= possible_count:
        return str(possible_count), _percent(evaluated_share), "complete"

    remaining_sets = possible_count - evaluated_count
    try:
        measured_page_rate = float(page_rate)
        pages_per_set = int(page_count)
    except (TypeError, ValueError):
        measured_page_rate = 0.0
        pages_per_set = 0

    if measured_page_rate > 0 and pages_per_set > 0:
        eta = _duration(remaining_sets * pages_per_set / measured_page_rate)
    else:
        try:
            elapsed = float(elapsed_seconds)
        except (TypeError, ValueError):
            elapsed = 0.0
        if evaluated_count <= 0 or elapsed <= 0:
            eta = "unknown"
        else:
            eta = _duration(remaining_sets * (elapsed / evaluated_count))

    return str(possible_count), _percent(evaluated_share), eta


def _stabilization_interpretation(observation: dict[str, Any]) -> str:
    value = observation.get("search_space_fraction")
    if value is None:
        value = observation.get("search_fraction")
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        text = _search_space_percent(observation)
        try:
            fraction = float(str(text).rstrip("%")) / 100.0
        except (TypeError, ValueError):
            return "Unavailable because the completed-search fraction was not recorded."
    if fraction < 0.10:
        return "Early convergence — the final winner emerged within the first 10% of the evaluated search."
    if fraction <= 0.40:
        return "Moderate exploration — the final winner emerged after 10–40% of the evaluated search."
    if fraction <= 0.80:
        return "Late convergence — the final winner emerged after 40–80% of the evaluated search."
    return "No stable optimum — the final winner did not emerge until more than 80% of the evaluated search."


def _engineering_recommendation(detector: str, payload: dict[str, Any] | None, combined_rows: list[dict[str, Any]]) -> str:
    if not payload:
        return "Calibration evidence is unavailable; retain the current winner provisionally and rerun calibration before making additional detector-development decisions."
    search = payload.get("search", {}) if isinstance(payload.get("search"), dict) else {}
    landscape = payload.get("landscape", {}) if isinstance(payload.get("landscape"), dict) else {}
    near_best = float(landscape.get("near_best_share", 0.0) or 0.0)
    exhaustive = bool(search.get("exhaustive_complete"))
    evaluated = int(search.get("parameter_sets", 0) or 0)
    possible = int(search.get("possible_parameter_sets", 0) or 0)
    coverage = evaluated / possible if possible else 0.0
    rank = next((index for index, row in enumerate(combined_rows, start=1) if str(row.get("detector")) == detector), 1)
    if rank > 1:
        return "Further tuning is not currently justified for selection purposes; stronger detectors exist for this Golden Set. Preserve this detector when its evidence is complementary enough to support future fusion."
    if near_best >= 0.50 and (exhaustive or coverage >= 0.50):
        return "Detector appears well calibrated for this Golden Set; broad near-best coverage and mature search evidence make material gains from additional parameter tuning unlikely."
    if near_best <= 0.05 and not exhaustive and coverage < 0.50:
        return "Continue detector calibration; the observed optimum is narrow and parameter-space coverage is still limited, so additional exploration may produce a materially better configuration."
    return "Retain this detector as the current Golden Set recommendation. Additional tuning should be driven by unresolved page failures, late winner changes, or a plausible untested parameter region rather than by search expansion alone."


def _slugify_heading(text: str) -> str:
    value = re.sub(r"[`*_]", "", text).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "section"


def _navigation_heading(line: str) -> tuple[int, str] | None:
    """Return the visible report level and title for Markdown and details headings."""
    markdown = re.match(r"^(##|###) (.+)$", line)
    if markdown:
        return len(markdown.group(1)), markdown.group(2).strip()

    details_heading = re.fullmatch(
        r"<summary><h([23])>(.+)</h\1></summary>", line.strip()
    )
    if details_heading:
        return int(details_heading.group(1)), details_heading.group(2).strip()

    detector_heading = re.fullmatch(
        r"<summary><strong>(.+)</strong></summary>", line.strip()
    )
    if detector_heading:
        return 4, detector_heading.group(1).strip()

    return None


def _add_report_navigation(lines: list[str]) -> list[str]:
    """Add a complete TOC while leaving the report body independently collapsible.

    GitHub does not expose headings embedded in ``<summary>`` elements to the
    normal Markdown table of contents.  HTH deliberately uses those summaries
    for the calibration, regression, and per-detector nesting, so collect both
    ordinary Markdown headings and details-summary headings here.  Explicit
    anchors make every TOC entry stable even when its target is collapsed.
    """
    headings: list[tuple[int, str, str]] = []
    used: dict[str, int] = {}
    for line in lines:
        heading = _navigation_heading(line)
        if heading is None:
            continue
        level, title = heading
        base = _slugify_heading(title)
        used[base] = used.get(base, 0) + 1
        slug = base if used[base] == 1 else f"{base}-{used[base]}"
        headings.append((level, title, slug))

    if not headings:
        return lines

    navigation = [
        '<a id="table-of-contents"></a>',
        "",
        "<details open>",
        "<summary><strong>Navigation</strong></summary>",
        "",
    ]
    for level, title, slug in headings:
        indent = "  " * max(0, level - 2)
        navigation.append(f"{indent}- [{title}](#{slug})")
    navigation.extend(["", "</details>", ""])

    result: list[str] = []
    inserted_navigation = False
    heading_index = 0
    for line in lines:
        if not inserted_navigation and line.startswith("# "):
            result.append(line)
            result.extend(["", *navigation])
            inserted_navigation = True
            continue

        heading = _navigation_heading(line)
        if heading is not None:
            _, _, slug = headings[heading_index]

            # A <summary> must remain the first child of its <details> element.
            # Injecting navigation anchors between ``<details>`` and ``<summary>``
            # causes GitHub to render a stray generic "Details" disclosure and
            # the intended heading outside it.  Move the opening details tag
            # temporarily so the navigation link and anchor precede the entire
            # disclosure block instead.
            details_open = None
            trailing_blank = False
            if line.lstrip().startswith("<summary>"):
                if result and result[-1] == "":
                    result.pop()
                    trailing_blank = True
                if result and result[-1].strip().startswith("<details"):
                    details_open = result.pop()

            if result and result[-1] != "":
                result.append("")
            if heading_index > 0:
                result.extend(["[↑ Back to Navigation](#table-of-contents)", ""])
            result.append(f'<a id="{slug}"></a>')
            if details_open is not None:
                result.append(details_open)
            result.append(line)
            if trailing_blank:
                result.append("")
            heading_index += 1
        else:
            result.append(line)

    if heading_index:
        result.extend(["", "[↑ Back to Navigation](#table-of-contents)"])
    return result


def _parameter_short_name(result: dict[str, Any] | None) -> str:
    """Return the human-friendly parameter alias, falling back to its stable ID."""
    if not result:
        return "unknown"
    return str(
        result.get("parameter_short_name")
        or result.get("short_name")
        or result.get("profile")
        or _short(result.get("parameter_set_id"), 12)
    )


def _evaluation_seconds(result: dict[str, Any] | None) -> float | None:
    """Return parameter-set wall time for the detailed per-detector report."""
    if not result:
        return None
    summary = result.get("summary", {})
    if not isinstance(summary, dict):
        return None
    milliseconds = summary.get("wall_ms")
    if milliseconds is None:
        milliseconds = summary.get("elapsed_ms_total")
    try:
        return float(milliseconds) / 1000.0
    except (TypeError, ValueError):
        return None


def _detector_seconds(result: dict[str, Any] | None) -> float | None:
    """Return summed detector time across the evaluated Golden Set pages."""
    if not result:
        return None
    summary = result.get("summary", {})
    if not isinstance(summary, dict):
        return None
    milliseconds = summary.get("elapsed_ms_total")
    if milliseconds is None:
        milliseconds = summary.get("wall_ms")
    try:
        return float(milliseconds) / 1000.0
    except (TypeError, ValueError):
        return None


def _page_rate(result: dict[str, Any] | None, page_count: int) -> float | None:
    seconds = _detector_seconds(result)
    if seconds is None or seconds <= 0 or page_count <= 0:
        return None
    return page_count / seconds


def _format_page_rate(value: Any) -> str:
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if rate >= 10:
        return f"{rate:.2f} pg/s"
    if rate >= 1:
        return f"{rate:.3f} pg/s"
    return f"{rate:.4f} pg/s"


def _source_document_metadata(
    run_dir: Path,
    info: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Load source-document identity and image count from Golden Set metadata."""
    golden_set = info.get("golden_set") or parameters.get("golden_set")
    if not golden_set:
        return {}
    golden_path = Path(str(golden_set))
    candidates = [golden_path]
    if not golden_path.is_absolute():
        candidates.extend([run_dir / golden_path, Path.cwd() / golden_path])
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = _read_json(candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        source = payload.get("source_document", {})
        if isinstance(source, dict):
            return {
                "title": str(source.get("title") or "").strip(),
                "image_count": source.get("image_count"),
            }
    return {}


def _estimated_document_seconds(page_rate: Any, image_count: Any) -> float | None:
    try:
        rate = float(page_rate)
        count = int(image_count)
    except (TypeError, ValueError):
        return None
    if rate <= 0 or count <= 0:
        return None
    return count / rate


def _parameter_id(result: dict[str, Any] | None) -> str:
    if not result:
        return "unknown"
    return _short(result.get("parameter_set_id"), 12)


def _search_observation(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    observation = result.get("search_observation", {})
    return observation if isinstance(observation, dict) else {}


def _search_space_percent(observation: dict[str, Any]) -> str:
    fraction = observation.get("search_fraction")
    return _percent(fraction, 2) if fraction is not None else "unknown"


def build_summary(
    run_dir: Path,
    run_url: str = "",
    *,
    include_title: bool = True,
    include_metric_definitions: bool = True,
) -> str:
    manifest = _read_json(run_dir / "manifest.json")
    info = _read_json(run_dir / "RUN-INFO.json")
    parameters = _read_json(run_dir / "parameters.json")
    summary = _read_json(run_dir / "reports" / "summary.json")

    winner = summary.get("winner") if isinstance(summary.get("winner"), dict) else None
    baseline = summary.get("baseline") if isinstance(summary.get("baseline"), dict) else None
    winner_stats = winner.get("summary", {}) if winner else {}
    baseline = summary.get("baseline") if isinstance(summary.get("baseline"), dict) else None
    baseline_stats = baseline.get("summary", {}) if baseline else {}
    outputs = manifest.get("outputs", []) if isinstance(manifest.get("outputs"), list) else []
    progress = summary.get("progress", {}) if isinstance(summary.get("progress"), dict) else {}
    page_ordinals = summary.get("page_ordinals", []) if isinstance(summary.get("page_ordinals"), list) else []
    configuration = parameters.get("configuration", {}) if isinstance(parameters.get("configuration"), dict) else {}
    profiles = configuration.get("profiles", {}) if isinstance(configuration.get("profiles"), dict) else {}

    lines = []
    if include_title:
        lines.extend(["# Regression Manifest", ""])
    lines.extend([
        f"**Status:** {manifest.get('status', 'unknown')}",
        "",
        "## Run Information",
        "",
        "### Build Provenance",
        "",
        f"- Run ID: `{manifest.get('run_id', 'unknown')}`",
        f"- Detector: `{manifest.get('detector', 'unknown')}`",
        f"- Strategy: `{manifest.get('strategy', 'unknown')}`",
        f"- Pipeline commit: `{_short(info.get('pipeline_commit'))}`",
        f"- Python: `{info.get('python_version', 'unknown')}`",
        f"- OpenCV: `{info.get('opencv_version', 'unknown')}`",
        f"- Started: `{info.get('started_at_utc', manifest.get('started_at_utc', 'unknown'))}`",
        f"- Finished: `{info.get('finished_at_utc', manifest.get('finished_at_utc', 'unknown'))}`",
        f"- Elapsed: `{_duration(info.get('elapsed_seconds'))}`",
        "",
        "### Golden Set",
        "",
        f"- Configuration: `{info.get('golden_set', parameters.get('golden_set', 'unknown'))}`",
        f"- SHA-256: `{_short(info.get('golden_set_sha256', parameters.get('golden_set_sha256', summary.get('golden_set_sha256', 'unknown'))), 12)}`",
        f"- Pages: `{len(page_ordinals)}`",
        f"- Ordinals: `{', '.join(str(v) for v in page_ordinals) if page_ordinals else 'unknown'}`",
        "",
        "### Parameter Space",
        "",
        f"- All possible parameter sets: `{summary.get('parameter_space', {}).get('possible_parameter_sets', 'unknown')}`",
        f"- Parameter sets evaluated: `{summary.get('parameter_set_count', 'unknown')}`",
        f"- Evaluated sets (% of all possible parameter sets): `{_percent((float(summary.get('parameter_set_count', 0) or 0) / float(summary.get('parameter_space', {}).get('possible_parameter_sets', 0) or 1)) if summary.get('parameter_space', {}).get('possible_parameter_sets') else None, 2)}`",
        f"- Configured named profiles: `{', '.join(sorted(profiles)) if profiles else 'none'}`",
    ])

    if outputs:
        lines.extend(["", "### Outputs", ""])
        for output in outputs:
            path = run_dir / str(output)
            state = "present" if path.exists() else "missing"
            lines.append(f"- `{output}` — {state}")

    lines.extend([
        "",
        "## Results",
        "",
        "### Result",
        "",
        "| Result | Parameter Set ID | Parameter Short Name | Avg IoU | Min IoU | StdDev | Failures | Evaluation Time |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        f"| Winner | `{_parameter_id(winner)}` | `{_parameter_short_name(winner)}` | {_number(winner_stats.get('mean_iou'))} | {_number(winner_stats.get('minimum_iou'))} | {_number(winner_stats.get('stddev_iou'))} | {winner_stats.get('failure_count', 'unknown')} | {_duration(_evaluation_seconds(winner))} |",
    ])
    if baseline and _parameter_id(baseline) != _parameter_id(winner):
        lines.append(
            f"| Baseline | `{_parameter_id(baseline)}` | `{_parameter_short_name(baseline)}` | "
            f"{_number(baseline_stats.get('mean_iou'))} | "
            f"{_number(baseline_stats.get('minimum_iou'))} | "
            f"{_number(baseline_stats.get('stddev_iou'))} | "
            f"{baseline_stats.get('failure_count', 'unknown')} | "
            f"{_duration(_evaluation_seconds(baseline))} |"
        )

    detector_name = str(manifest.get("detector", "unknown"))
    characterization = _detector_characterization(detector_name)
    lines.extend([
        "",
        "### Detector Evidence",
        "",
        f"**Role:** {characterization.get('role', 'Unknown')}",
        "",
        "| Evidence source | Function | Interpretation |",
        "|---|---|---|",
    ])
    for evidence_name, function, interpretation in characterization.get("evidence", []):
        lines.append(f"| {evidence_name} | {function} | {interpretation} |")

    if include_metric_definitions:
        lines.extend([
            "",
            "### Metric Definitions",
            "",
            "- **Avg IoU:** Arithmetic mean of the winner's page IoUs across the Golden Set; the primary detector-ranking metric.",
            "- **Min IoU:** Lowest single-page IoU produced by the parameter set; exposes the worst Golden Set page rather than the worst parameter set.",
            "- **StdDev:** Population standard deviation of page IoUs; lower values indicate more even page-to-page performance, but must be interpreted with Avg IoU and Min IoU.",
            "- **Failures:** Number of Golden Set pages that could not be evaluated successfully.",
        ])

    lines.extend([
        "",
        "### Regression Statistics for Detector Calibration",
        "",
        "| Statistic | Count |",
        "|---|---:|",
        f"| Avg IoU improvements | {progress.get('mean_iou_improvements', 0)} |",
        f"| Minimum IoU improvements | {progress.get('minimum_iou_improvements', 0)} |",
        f"| StdDev improvements | {progress.get('stddev_improvements', 0)} |",
        f"| Total metric improvements | {progress.get('total_metric_improvements', 0)} |",
        f"| Parameter sets with improvements | {progress.get('parameter_sets_with_improvements', 0)} |",
        f"| Winner changes | {progress.get('winner_changes', 0)} |",
        f"| Baseline surpassed | {'yes' if progress.get('baseline_surpassed') else 'no'} |",
    ])

    top_parameter_sets = summary.get("top_parameter_sets", [])
    if isinstance(top_parameter_sets, list) and top_parameter_sets:
        winner_mean = float(winner_stats.get("mean_iou", 0.0) or 0.0)
        lines.extend([
            "",
            "### Top Parameter Sets",
            "",
            "| Rank | Parameter Set ID | Parameter Short Name | Avg IoU | Min IoU | StdDev | Δ Avg IoU | Failures | Discovery Time | Search Space % |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for result in top_parameter_sets[:5]:
            stats = result.get("summary", {}) if isinstance(result, dict) else {}
            mean_iou = float(stats.get("mean_iou", 0.0) or 0.0)
            delta_mean_iou = mean_iou - winner_mean
            parameter_set_name = _parameter_short_name(result)
            failure_count = stats.get("failure_count", "unknown")
            observation = _search_observation(result)

            lines.append(
                f"| {result.get('rank', 'unknown')} | `{_parameter_id(result)}` | `{parameter_set_name}` | "
                f"{_number(mean_iou)} | {_number(stats.get('minimum_iou'))} | "
                f"{_number(stats.get('stddev_iou'))} | {delta_mean_iou:+.4f} | "
                f"{failure_count} | {_duration(observation.get('elapsed_seconds'))} | "
                f"{_search_space_percent(observation)} |"
            )

    winner_page_report = summary.get("winner_page_report", {})
    winner_pages = (
        winner_page_report.get("pages", [])
        if isinstance(winner_page_report, dict) 
        else []
    )
    
    if isinstance(winner_pages, list) and winner_pages:
        winner_pages = sorted(
            winner_pages,
            key=lambda page: int(page.get("golden_set_page", 0) or 0),
        )
        lines.extend([
            "",
            "## Page Analysis",
            "",
            "### Golden Set Winner Summary",
            "",
            "| Golden Set Page | Parameter Set ID | Baseline | Winner | Δ IoU | Status |",
            "|---:|---|---:|---:|---:|---|",
        ])
        for page in winner_pages:
            lines.append(
                f"| {page.get('golden_set_page', 'unknown')} | `{_short(page.get('parameter_set'), 12)}` | "
                f"{_number(page.get('baseline_iou'))} | {_number(page.get('winner_iou'))} | "
                f"{float(page.get('delta_iou', 0.0) or 0.0):+.4f} | "
                f"{page.get('status', 'unknown')} |"
            )

        winner_history = progress.get("winner_history", []) if isinstance(progress.get("winner_history"), list) else []
        lines.extend([
            "",
            "#### Winner History",
            "",
            "| Discovery Order | Parameter Set ID | Search Time | % Search |",
            "|---:|---|---:|---:|",
        ])
        for event in winner_history[-5:]:
            marker = f"{event.get('change_number', 'unknown')}"
            if event is winner_history[-1]:
                marker += " (final)"
            lines.append(
                f"| {marker} | `{_short(event.get('parameter_set_id'), 12)}` | "
                f"{_duration(event.get('elapsed_seconds'))} | {_search_space_percent(event)} |"
            )
        if not winner_history:
            lines.append("| — | unavailable | unknown | unknown |")
        lines.extend([
            "",
            f"Total winner changes: **{progress.get('winner_changes', 0)}**.",
            f"Search completed in **{_duration(info.get('elapsed_seconds'))}**.",
            "",
            f"**Stabilization Interpretation:** {_stabilization_interpretation(winner_history[-1] if winner_history else {})}",
        ])

        thresholds = winner_page_report.get("thresholds", {})
        poor_match_threshold = float(thresholds.get("poor_match_iou_below", 0.50) or 0.50)
        regression_threshold = float(thresholds.get("regression_delta_below", -0.001) or -0.001)
        counts = winner_page_report.get("counts", {})
        problem_pages = [page for page in winner_pages if page.get("problem")]
        lines.extend([
            "",
            "### Status Definitions",
            "",
            "- **Recovered:** baseline IoU was zero and the winner found a matching polygon.",
            f"- **Improved:** Δ IoU is greater than `{abs(regression_threshold):.4f}`.",
            f"- **Unchanged:** Δ IoU is between `{regression_threshold:.4f}` and `{abs(regression_threshold):+.4f}`.",
            f"- **Regressed:** Δ IoU is less than `{regression_threshold:.4f}`.",
            f"- **Poor match:** Winner IoU is greater than zero but below `{poor_match_threshold:.4f}`.",
            "- **Zero overlap:** a polygon was returned, but its IoU is zero.",
            "- **No polygon found:** the detector completed without returning a polygon.",
            "- **Unprocessed:** evaluation raised an error.",
            "",
            "### Golden Set Page Issues",
            "",
            f"- Unprocessed pages: `{counts.get('unprocessed_pages', 0)}`",
            f"- No polygon found: `{counts.get('no_polygon_found', 0)}`",
            f"- Zero overlap: `{counts.get('zero_overlap', 0)}`",
            f"- Poor matches (Winner IoU < {poor_match_threshold:.4f}): `{counts.get('poor_matches', 0)}`",
            f"- Regressed pages (Δ IoU < {regression_threshold:.4f}): `{counts.get('regressions', 0)}`",
        ])
        if problem_pages:
            lines.extend([
                "",
                "#### Affected Pages",
                "",
                "| Golden Set Page | Parameter Set ID | Winner IoU | Problem |",
                "|---:|---|---:|---|",
            ])
            for page in problem_pages:
                reasons = "; ".join(str(reason) for reason in page.get("problem_reasons", [])) or str(page.get("status", "unknown"))
                lines.append(
                    f"| {page.get('golden_set_page', 'unknown')} | "
                    f"`{_short(page.get('parameter_set'), 12)}` | "
                    f"{_number(page.get('winner_iou'))} | {reasons} |"
                )
        else:
            lines.extend(["", "No problem pages were identified."])

    calibration_payload = _calibration_payload(run_dir)
    if calibration_payload is not None:
        requested_strategy = manifest.get("requested_strategy", info.get("requested_strategy", manifest.get("strategy", "unknown")))
        resolved_strategy = manifest.get("strategy", info.get("strategy", "unknown"))
        fallback_reason = manifest.get("strategy_fallback_reason", info.get("strategy_fallback_reason"))
        confidence = calibration_payload.get("calibration_confidence", {}) if isinstance(calibration_payload.get("calibration_confidence"), dict) else {}
        recommendations = calibration_payload.get("recommendations", {}) if isinstance(calibration_payload.get("recommendations"), dict) else {}
        domain_space = calibration_payload.get("domain_space", {}) if isinstance(calibration_payload.get("domain_space"), dict) else {}
        lines.extend([
            "",
            "## Calibration Intelligence",
            "",
            "This run generated the same machine-readable calibration intelligence used by the multi-detector smoke regression. The conclusions remain specific to this Golden Set and configured parameter grid.",
            "",
            "### Calibration Identity",
            "",
            f"- Calibration run ID: `{manifest.get('run_id', 'unknown')}`",
            f"- Calibration schema: `{calibration_payload.get('schema_version', 'unknown')}`",
            f"- Detector: `{manifest.get('detector', 'unknown')}`",
            f"- Detector configuration: `{info.get('detector_config', parameters.get('detector_config', 'unknown'))}`",
            f"- Golden Set configuration: `{info.get('golden_set', parameters.get('golden_set', 'unknown'))}`",
            f"- Golden Set SHA-256: `{info.get('golden_set_sha256', parameters.get('golden_set_sha256', summary.get('golden_set_sha256', 'unknown')))}`",
            f"- Pipeline commit: `{info.get('pipeline_commit', 'unknown')}`",
            f"- Source commit: `{info.get('source_commit', summary.get('source_commit', 'unknown'))}`",
            f"- Requested search strategy: `{requested_strategy}`",
            f"- Resolved search strategy: `{resolved_strategy}`",
            f"- Strategy fallback: `{fallback_reason or 'none'}`",
            f"- Configured threads: `{info.get('threads', summary.get('threads', parameters.get('threads', 'unknown')))}`",
            "",
            "### Detector-Selection Intelligence",
            "",
            f"- Recommended parameter set: `{_parameter_id(winner)}`",
            f"- Recommended parameter short name: `{_parameter_short_name(winner)}`",
            f"- Best observed Avg IoU: `{_number(winner_stats.get('mean_iou'))}`",
            f"- Worst Golden Set page (Min IoU): `{_number(winner_stats.get('minimum_iou'))}`",
            f"- Page-to-page StdDev: `{_number(winner_stats.get('stddev_iou'))}`",
            f"- Calibration evidence: `{confidence.get('rating', 'unknown')}`",
            f"- Dormant parameters: `{', '.join(str(v) for v in recommendations.get('dormant_parameters', [])) if isinstance(recommendations.get('dormant_parameters'), list) and recommendations.get('dormant_parameters') else 'none'}`",
            f"- Available domain spaces: `{', '.join(str(key) for key, value in domain_space.items() if isinstance(value, dict) and int(value.get('parameter_set_count', 0) or 0) > 0) or 'none'}`",
            "",
            "### Calibration Analysis",
            "",
        ])
        lines.extend(_render_detector_calibration(detector_name, calibration_payload, summary))

    if run_url:
        lines.extend(["", f"[Open workflow run]({run_url})"])
    lines.append("")
    return "\n".join(lines)




def _percent(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "unknown"


def _calibration_payload(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "reports" / "calibration-intelligence.json"
    if not path.is_file():
        return None
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if payload.get("available") else None


def _detector_summary_and_roi(detector: str, payload: dict[str, Any]) -> tuple[list[str], str]:
    search = payload.get("search", {}) if isinstance(payload.get("search"), dict) else {}
    landscape = payload.get("landscape", {}) if isinstance(payload.get("landscape"), dict) else {}
    parameters = payload.get("parameter_influence", []) if isinstance(payload.get("parameter_influence"), list) else []
    pages = payload.get("page_sensitivity", []) if isinstance(payload.get("page_sensitivity"), list) else []
    dormant_count = sum(1 for item in parameters if item.get("classification") == "Dormant")
    parameter_count = len(parameters)
    near_best = float(landscape.get("near_best_share", 0.0) or 0.0)
    spread = float(landscape.get("stddev_mean_iou", 0.0) or 0.0)
    success = float(search.get("fully_successful_rate", 0.0) or 0.0)
    page_success = min((float(page.get("success_rate", 0.0) or 0.0) for page in pages), default=0.0)

    findings: list[str] = []
    if near_best >= 0.90:
        findings.append("The evaluated calibration landscape is flat: nearly all tested parameter sets are equivalent or near-equivalent.")
    elif near_best <= 0.05:
        findings.append("The near-best coverage (basin) is narrow, so detector quality depends strongly on selecting a small part of the configured grid.")
    else:
        findings.append("The detector has a measurable but not singular near-best coverage (basin) within the evaluated grid.")
    if parameter_count and dormant_count == parameter_count:
        findings.append("Every measured parameter was dormant for this Golden Set and grid.")
    elif dormant_count:
        findings.append(f"{dormant_count} of {parameter_count} measured parameters were dormant and may be omitted from a source-specific follow-up search.")
    else:
        findings.append("No measured parameter was dormant in this calibration sample.")
    if success < 0.50 or page_success == 0.0:
        findings.append(f"Detector failed on at least one Golden Set page for {max(0, int(search.get('parameter_sets', 0) or 0) - int(search.get('fully_successful_parameter_sets', 0) or 0))} of {int(search.get('parameter_sets', 0) or 0)} parameter configurations.")
    elif success >= 0.90:
        findings.append("Most parameter sets evaluated every Golden Set page successfully.")
    if spread >= 0.10:
        findings.append("Avg IoU varies widely across the tested parameter sets.")
    elif spread <= 0.005:
        findings.append("Avg IoU varies very little across the tested parameter sets.")

    if near_best >= 0.90 and (not parameter_count or dormant_count / parameter_count >= 0.75):
        roi = "Further parameter tuning has low expected ROI for this source; improvement would more likely require an algorithmic change or a broader Golden Set."
    elif success < 0.50 or page_success == 0.0:
        roi = "Additional tuning may improve reliability, but detector-level ROI should be weighed against stronger alternatives before expanding the search."
    elif near_best <= 0.05 and not search.get("exhaustive_complete"):
        roi = "A broader or exhaustive search may still have calibration ROI because the observed optimum is narrow and coverage is incomplete."
    elif search.get("exhaustive_complete"):
        roi = "The configured grid is fully characterized for this Golden Set; continue detector work only if the resulting quality or failure pattern remains operationally inadequate."
    else:
        roi = "Some calibration ROI may remain, but it should be justified by page-level failures or a plausible untested parameter region."
    return findings, roi


def _render_detector_calibration(detector: str, payload: dict[str, Any], summary: dict[str, Any] | None = None) -> list[str]:
    search = payload.get("search", {}) if isinstance(payload.get("search"), dict) else {}
    landscape = payload.get("landscape", {}) if isinstance(payload.get("landscape"), dict) else {}
    confidence = payload.get("calibration_confidence", {}) if isinstance(payload.get("calibration_confidence"), dict) else {}
    parameters = payload.get("parameter_influence", []) if isinstance(payload.get("parameter_influence"), list) else []
    interactions = payload.get("interactions", []) if isinstance(payload.get("interactions"), list) else []
    pages = payload.get("page_sensitivity", []) if isinstance(payload.get("page_sensitivity"), list) else []
    recommendations = payload.get("recommendations", {}) if isinstance(payload.get("recommendations"), dict) else {}
    findings, roi = _detector_summary_and_roi(detector, payload)
    summary = summary or {}
    progress = summary.get("progress", {}) if isinstance(summary.get("progress"), dict) else {}
    winner = summary.get("winner") if isinstance(summary.get("winner"), dict) else None
    winner_observation = _search_observation(winner)
    page_ordinals = summary.get("page_ordinals", []) if isinstance(summary.get("page_ordinals"), list) else []
    page_count = len(page_ordinals)
    page_rate = _page_rate(winner, page_count)
    full_search_metrics = _full_search_metrics(
        search,
        summary.get("elapsed_seconds"),
        page_rate=page_rate,
        page_count=page_count,
    )

    lines = [
        str(payload.get("scope_note") or "Conclusions are specific to this run's Golden Set and parameter grid."),
        "",
        "#### Detector Summary",
        "",
    ]
    lines.extend(f"- {finding}" for finding in findings)
    lines.extend([
        "",
        "#### Evidence of ROI",
        "",
        roi,
        "",
        "#### Calibration Landscape",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Search coverage | {'complete exhaustive' if search.get('exhaustive_complete') else 'partial / adaptive'} |",
        f"| All possible parameter sets | {full_search_metrics[0]} |",
        f"| Parameter sets evaluated | {search.get('parameter_sets', 'unknown')} |",
        f"| Evaluated sets (% of all possible parameter sets) | {full_search_metrics[1]} |",
        f"| ETA for full parameter set evaluation* | {full_search_metrics[2]} |",
        f"| Fully successful parameter sets | {search.get('fully_successful_parameter_sets', 'unknown')} ({_percent(search.get('fully_successful_rate'))}) |",
        f"| Best Avg IoU | {_number(landscape.get('best_mean_iou'))} |",
        f"| Minimum Avg IoU | {_number(landscape.get('minimum_mean_iou'))} |",
        f"| Avg IoU StdDev | {_number(landscape.get('stddev_mean_iou'))} |",
        f"| Winner stabilized after | {_pluralized_parameter_sets(winner_observation.get('parameter_set_number'))} |",
        f"| Winner stabilized | {_compact_duration(winner_observation.get('elapsed_seconds'))} ({_compact_percent(winner_observation.get('search_fraction'))} of search) |",
        f"| Near-best coverage (basin; within {float(landscape.get('near_best_tolerance', 0.001) or 0.001):.4f}) | {landscape.get('near_best_count', 'unknown')} ({_percent(landscape.get('near_best_share'))}) |",
        f"| Equivalent-best configurations (within {float(landscape.get('equivalent_tolerance', 0.0001) or 0.0001):.4f}) | {landscape.get('equivalent_winner_count', 'unknown')} ({_percent(landscape.get('equivalent_winner_share'))}) |",
        f"| Calibration Evidence | {confidence.get('rating', 'unknown')} |",
        "",
        r"\* **ETA Note:** Long parameter-set regression ETAs assume a single-threaded serial run at the measured detector page rate. Actual wall time will vary with parallelization, worker count, scheduling overhead, and parameter-dependent runtime.",
    ])

    domain_space = payload.get("domain_space", {}) if isinstance(payload.get("domain_space"), dict) else {}
    if domain_space:
        exhaustive_count = int((domain_space.get("exhaustive") or {}).get("parameter_set_count", 0) or 0)
        exhaustive_time = full_search_metrics[2]
        estimated_full_seconds = None
        if exhaustive_count and page_rate and page_count:
            estimated_full_seconds = exhaustive_count * page_count / page_rate
        lines.extend([
            "", "#### Parameter Set Domain Space Reduction", "",
            "| Effect Size Group | Parameter Sets | % All Sets | New Time Est* | Set Reduction Factor |",
            "|---|---:|---:|---:|---:|",
        ])
        for key, label in (("exhaustive", "Exhaustive"), ("non_dormant", "Non-dormant"), ("low_plus", "Low+"), ("moderate_plus", "Moderate+"), ("important_plus", "Important+"), ("critical", "Critical")):
            entry = domain_space.get(key)
            if not isinstance(entry, dict):
                continue
            count_value = int(entry.get("parameter_set_count", 0) or 0)
            percent = count_value / exhaustive_count if exhaustive_count else 0.0
            seconds = estimated_full_seconds * percent if estimated_full_seconds is not None else None
            factor = exhaustive_count / count_value if count_value else None
            factor_text = f"{factor:.1f}×" if factor is not None else "unavailable"
            lines.append(f"| {label} | {count_value} | {_percent(percent)} | {_duration(seconds)} | {factor_text} |")
        lines.extend(["", r"\* Uses the same serial measured-page-rate assumptions as the Calibration Landscape ETA."])
    reasons = confidence.get("reasons", []) if isinstance(confidence.get("reasons"), list) else []
    if reasons:
        lines.extend(["", f"Calibration evidence basis: {', '.join(str(reason) for reason in reasons)}."])

    if parameters:
        lines.extend([
            "", "#### Parameter Influence", "",
            "Influence uses one-way η² over Avg IoU. It measures association within this configured grid; it does not establish causation.", "",
            "| Parameter | Classification | η² | Avg-IoU range | Near-best value coverage | Best observed values |",
            "|---|---|---:|---:|---:|---|",
        ])
        for item in parameters[:12]:
            best_values = item.get("best_values", []) if isinstance(item.get("best_values"), list) else []
            rendered_values = ", ".join(
                f"`{entry.get('value')}` ({_number(entry.get('mean_iou'))})"
                for entry in best_values[:3] if isinstance(entry, dict)
            ) or "unknown"
            lines.append(
                f"| `{item.get('parameter', 'unknown')}` | {item.get('classification', 'unknown')} | "
                f"{_number(item.get('eta_squared'))} | {_number(item.get('mean_iou_range'))} | "
                f"{_percent(item.get('near_best_value_coverage'))} | {rendered_values} |"
            )

    dormant = recommendations.get("dormant_parameters", []) if isinstance(recommendations.get("dormant_parameters"), list) else []
    if dormant:
        lines.extend([
            "", "#### Dormant Parameters", "",
            "These parameters had no material measured effect on Avg IoU for this Golden Set and grid:", "",
            ", ".join(f"`{name}`" for name in dormant) + ".", "",
            str(recommendations.get("note") or "Re-evaluate dormant parameters whenever the Golden Set changes."),
        ])

    meaningful_interactions = [
        item for item in interactions
        if isinstance(item, dict) and float(item.get("incremental_importance", 0.0) or 0.0) >= 0.001
    ]
    if meaningful_interactions:
        lines.extend([
            "", "#### Parameter Interactions", "",
            "Pairwise interaction importance is exploratory and estimated from a deterministic sample.", "",
            "| Parameters | Pair η² | Incremental importance | Sample size |",
            "|---|---:|---:|---:|",
        ])
        for item in meaningful_interactions[:5]:
            pair = item.get("parameters", [])
            pair_name = " × ".join(f"`{name}`" for name in pair) if isinstance(pair, list) else "unknown"
            lines.append(
                f"| {pair_name} | {_number(item.get('eta_squared'))} | "
                f"{_number(item.get('incremental_importance'))} | {item.get('sample_size', 'unknown')} |"
            )

    if pages:
        lines.extend([
            "", "#### Page Sensitivity", "",
            "| Golden Set Page | Avg IoU | Min IoU | Max IoU | StdDev | Success rate |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for page in pages:
            lines.append(
                f"| {page.get('global_ordinal', 'unknown')} | {_number(page.get('mean_iou'))} | "
                f"{_number(page.get('minimum_iou'))} | {_number(page.get('maximum_iou'))} | "
                f"{_number(page.get('stddev_iou'))} | {_percent(page.get('success_rate'))} |"
            )
    return lines


def _render_calibration_report(run_dirs: list[Path], combined_rows: list[dict[str, Any]]) -> list[str]:
    payload_by_detector: dict[str, dict[str, Any]] = {}
    summary_by_detector: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for run_dir in run_dirs:
        manifest = _read_json(run_dir / "manifest.json")
        detector = str(manifest.get("detector", run_dir.parent.name))
        payload = _calibration_payload(run_dir)
        summary_path = run_dir / "reports" / "summary.json"
        if summary_path.is_file():
            summary_by_detector[detector] = _read_json(summary_path)
        if payload is None:
            missing.append(detector)
        else:
            payload_by_detector[detector] = payload

    lines = [
        "<details open>",
        "<summary><h2>Detector Calibration Report</h2></summary>",
        "",
        "This section characterizes the evaluated calibration landscapes, parameter influence, interactions, near-best coverage width, page sensitivity, and opportunities to reduce future search cost. All findings are Golden Set- and grid-specific and must be revalidated when the Golden Set or parameter space changes.", "",
    ]
    if not payload_by_detector:
        lines.extend(["Calibration intelligence was not available for these runs.", "", "</details>"])
        return lines

    lines.extend([
        "### Calibration Overview", "",
        "| Rank | Detector | Short Name | Detector ID | Role | Coverage | Parameter Sets | Successful | Best Avg IoU | Min IoU | StdDev | Δ Baseline Avg IoU | Near-best Coverage (Basin) | Equivalent Best Configurations | Calibration Evidence |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for rank, row in enumerate(combined_rows, start=1):
        detector = str(row["detector"])
        payload = payload_by_detector.get(detector)
        if not payload:
            continue
        search = payload.get("search", {}) if isinstance(payload.get("search"), dict) else {}
        landscape = payload.get("landscape", {}) if isinstance(payload.get("landscape"), dict) else {}
        confidence = payload.get("calibration_confidence", {}) if isinstance(payload.get("calibration_confidence"), dict) else {}
        role = _detector_characterization(detector).get("role", "Unknown")
        delta = row.get("delta_baseline_mean_iou")
        delta_text = f"{float(delta):+.4f}" if delta is not None else "unknown"
        lines.append(
            f"| {rank} | {_detector_friendly_name(detector)} | {_detector_short_name(detector)} | `{detector}` | {role} | {'complete' if search.get('exhaustive_complete') else 'partial'} | "
            f"{search.get('parameter_sets', 'unknown')} | {_percent(search.get('fully_successful_rate'))} | "
            f"{_number(row.get('mean_iou'))} | {_number(row.get('minimum_iou'))} | {_number(row.get('stddev_iou'))} | "
            f"{delta_text} | {_percent(landscape.get('near_best_share'))} | "
            f"{_percent(landscape.get('equivalent_winner_share'))} | {confidence.get('rating', 'unknown')} |"
        )
    lines.extend([
        "", "### Calibration Report Legend", "",
        "- **Generator:** proposes an original page boundary from its primary visual evidence.",
        "- **Validator:** scores or confirms a hypothesis generated elsewhere without normally proposing a competing boundary.",
        "- **Hybrid (detectors):** combines the named generator and validator or fuses the named generators.",
        "- **Critical / Important / Moderate / Low / Dormant:** plain-English parameter-influence classes, from dominant measured association to no material measured effect in this grid.",
        "- **Near-best coverage (basin):** share of tested parameter sets within the displayed tolerance of the best Avg IoU; broader coverage indicates more forgiving calibration.",
        "- **Equivalent best configurations:** share of tested sets effectively tied with the best result at the stricter displayed tolerance.",
        "- **Calibration Evidence:** strength of evidence that this run adequately describes the tested landscape; it is not confidence that the detector generalizes beyond this Golden Set and grid.",
        "- **Evidence tables:** identify what each detector actually observes and whether that evidence generates, validates, filters, or scores a page hypothesis.",
    ])
    if missing:
        lines.extend(["", "Calibration intelligence unavailable for: " + ", ".join(f"`{name}`" for name in missing) + "."])
    lines.extend(["", "<details open>", "<summary><h3>Per-Detector Calibration Reports</h3></summary>", ""])
    for row in combined_rows:
        detector = str(row["detector"])
        payload = payload_by_detector.get(detector)
        if payload:
            lines.extend(["", "<details>", f"<summary><strong>{_detector_heading(detector)}</strong></summary>", ""])
            lines.extend(_render_detector_calibration(detector, payload, summary_by_detector.get(detector)))
            lines.extend(["", "</details>"])
    lines.extend(["", "</details>", "", "</details>"])
    return lines

def _combined_result_row(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    info = _read_json(run_dir / "RUN-INFO.json")
    parameters = _read_json(run_dir / "parameters.json")
    summary = _read_json(run_dir / "reports" / "summary.json")
    winner = summary.get("winner") if isinstance(summary.get("winner"), dict) else None
    baseline = summary.get("baseline") if isinstance(summary.get("baseline"), dict) else None
    winner_stats = winner.get("summary", {}) if winner else {}
    baseline_stats = baseline.get("summary", {}) if baseline else {}
    page_ordinals = summary.get("page_ordinals", []) if isinstance(summary.get("page_ordinals"), list) else []
    source_document = _source_document_metadata(run_dir, info, parameters)
    page_rate = _page_rate(winner, len(page_ordinals))
    return {
        "detector": str(manifest.get("detector", run_dir.parent.name)),
        "detector_name": _detector_friendly_name(str(manifest.get("detector", run_dir.parent.name))),
        "detector_short_name": _detector_short_name(str(manifest.get("detector", run_dir.parent.name))),
        "status": str(manifest.get("status", "unknown")),
        "parameter_short_name": _parameter_short_name(winner),
        "parameter_set_id": _parameter_id(winner),
        "mean_iou": winner_stats.get("mean_iou"),
        "minimum_iou": winner_stats.get("minimum_iou"),
        "stddev_iou": winner_stats.get("stddev_iou"),
        "baseline_mean_iou": baseline_stats.get("mean_iou"),
        "delta_baseline_mean_iou": (
            float(winner_stats.get("mean_iou", 0.0) or 0.0)
            - float(baseline_stats.get("mean_iou", 0.0) or 0.0)
        ) if baseline else None,
        "failures": winner_stats.get("failure_count", "unknown"),
        "parameter_sets": summary.get("parameter_set_count", "unknown"),
        "golden_set_pages": len(page_ordinals),
        "elapsed_seconds": info.get("elapsed_seconds"),
        "page_rate": page_rate,
        "document_seconds": _estimated_document_seconds(page_rate, source_document.get("image_count")),
        "source_document": source_document,
    }


def _combined_ranking_key(row: dict[str, Any]) -> tuple[float, float, int, float, float]:
    def number(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return (
        -number(row.get("mean_iou"), 0.0),
        -number(row.get("minimum_iou"), 0.0),
        int(number(row.get("failures"), 10**9)),
        number(row.get("stddev_iou"), float("inf")),
        -number(row.get("page_rate"), 0.0),
    )


def build_combined_summary(run_dirs: list[Path], run_url: str = "") -> str:
    if not run_dirs:
        raise ValueError("At least one regression run directory is required")
    if len(run_dirs) == 1:
        return build_summary(run_dirs[0], run_url)

    combined_rows = sorted(
        (_combined_result_row(run_dir) for run_dir in run_dirs),
        key=_combined_ranking_key,
    )
    source_documents = [
        row.get("source_document", {})
        for row in combined_rows
        if row.get("source_document")
    ]
    source_document = source_documents[0] if source_documents else {}
    lines = [
        "# Detector Regression Manifest",
        "",
        f"**Detectors evaluated:** {len(run_dirs)}",
        "",
        "## Source document",
        "",
    ]
    if source_document.get("title"):
        lines.append(f"- **Document:** {source_document['title']}")
    else:
        lines.append("- **Document:** unknown")
    if source_document.get("image_count"):
        lines.append(f"- **Images:** {source_document['image_count']}")
    best_row = combined_rows[0]
    best_detector = str(best_row.get("detector", "unknown"))
    best_payload = _calibration_payload(next(run_dir for run_dir in run_dirs if str(_read_json(run_dir / "manifest.json").get("detector", run_dir.parent.name)) == best_detector))
    recommendation_notes: list[str] = []
    if best_payload:
        findings, roi = _detector_summary_and_roi(best_detector, best_payload)
        recommendation_notes.extend(findings[:2])
        recommendation_notes.append(roi)
    lines.extend([
        "",
        "## Detector Recommendation for this Golden Set",
        "",
        f"- **Recommended detector:** {_detector_friendly_name(best_detector)}",
        f"- **Detector short name:** {_detector_short_name(best_detector)}",
        f"- **Detector ID:** `{best_detector}`",
        f"- **Best observed Avg IoU:** `{_number(best_row.get('mean_iou'))}`",
        f"- **Worst Golden Set page (Min IoU):** `{_number(best_row.get('minimum_iou'))}`",
        f"- **Page-to-page StdDev:** `{_number(best_row.get('stddev_iou'))}`",
        f"- **Role:** `{_detector_characterization(best_detector).get('role', 'Unknown')}`",
        f"- **Engineering Recommendation:** {_engineering_recommendation(best_detector, best_payload, combined_rows)}",
    ])
    if recommendation_notes:
        lines.extend(["", "**Recommendation basis:**", ""] + [f"- {note}" for note in recommendation_notes])
    lines.extend([
        "",
        "This recommendation is specific to the evaluated Golden Set and parameter grid and should be revisited when the Golden Set, parameter grid, or source document changes.",
        "",
        "## Ranked Detector Results",
        "",
        "| Rank | Detector | Short Name | Detector ID | Status | Parameter Set ID | Parameter Short Name | Avg IoU | Min IoU | StdDev | Failures | Parameter Sets | Eval Rate | Doc Time | Run Elapsed |",
        "|---:|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rank, row in enumerate(combined_rows, start=1):
        lines.append(
            f"| {rank} | {row['detector_name']} | {row['detector_short_name']} | `{row['detector']}` | {row['status']} | "
            f"`{row['parameter_set_id']}` | `{row['parameter_short_name']}` | {_number(row['mean_iou'])} | "
            f"{_number(row['minimum_iou'])} | {_number(row['stddev_iou'])} | "
            f"{row['failures']} | {row['parameter_sets']} | "
            f"{_format_page_rate(row['page_rate'])} | {_duration(row['document_seconds'])} | "
            f"{_duration(row['elapsed_seconds'])} |"
        )
    lines.extend([
        "",
        "### Metric Definitions",
        "",
        "- **Avg IoU:** Arithmetic mean of a detector winner's page IoUs across the Golden Set; this is the primary ranking metric.",
        "- **Min IoU:** Lowest single-page IoU produced by that winner across the Golden Set. It exposes the detector's weakest evaluated page; it is not the minimum Avg IoU across parameter sets.",
        "- **StdDev:** Population standard deviation of the winner's page IoUs. Lower values indicate more even page-to-page performance, but a uniformly poor detector can also have a low StdDev, so read it with Avg IoU and Min IoU.",
        "- **Failures:** Number of Golden Set pages the winning parameter set could not evaluate successfully.",
        "- **Ranking order:** Avg IoU descending, then Min IoU descending, failures ascending, StdDev ascending, and evaluation rate descending.",
        "- **Δ Baseline Avg IoU:** Winning Avg IoU minus the named baseline profile's Avg IoU for the same detector run.",
        "",
    ])
    lines.extend(_render_calibration_report(run_dirs, combined_rows))

    completed_runs = sum(1 for row in combined_rows if str(row.get("status", "")).lower() == "complete")
    total_parameter_sets = sum(int(row.get("parameter_sets", 0) or 0) for row in combined_rows)
    total_page_evaluations = sum(
        int(row.get("parameter_sets", 0) or 0) * int(row.get("golden_set_pages", 0) or 0)
        for row in combined_rows
    )
    aggregate_elapsed = sum(float(row.get("elapsed_seconds", 0.0) or 0.0) for row in combined_rows)
    lines.extend([
        "",
        "<details open>",
        "<summary><h2>Detector Regression Reports</h2></summary>",
        "",
        "### Regression Completion Summary",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Detector runs completed | {completed_runs} of {len(combined_rows)} |",
        f"| Parameter sets evaluated | {total_parameter_sets} |",
        f"| Golden Set page evaluations | {total_page_evaluations} |",
        f"| Aggregate detector-run elapsed time | {_duration(aggregate_elapsed)} |",
        f"| Source-document images | {source_document.get('image_count', 'unknown')} |",
        "",
        "The reports below preserve the complete manifest, winner, baseline, calibration statistics, page analysis, and output inventory for each detector run.",
        "",
        "<details open>",
        "<summary><h3>Per-Detector Regression Reports</h3></summary>",
        "",
    ])
    for index, run_dir in enumerate(run_dirs):
        manifest = _read_json(run_dir / "manifest.json")
        detector = str(manifest.get("detector", run_dir.parent.name))
        lines.extend(["", "<details>", f"<summary><strong>{_detector_heading(detector)}</strong></summary>", ""])
        lines.append(
            build_summary(
                run_dir,
                include_title=False,
                include_metric_definitions=False,
            ).rstrip()
        )
        lines.extend(["", "</details>"])
        if index != len(run_dirs) - 1:
            lines.extend([""])
    lines.extend(["", "</details>", "", "</details>"])
    if run_url:
        lines.extend(["", f"[Open workflow run]({run_url})"])
    lines.append("")
    return "\n".join(_add_report_navigation(lines))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--run-url", default=os.environ.get("HTH_RUN_URL", ""))
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    text = build_combined_summary(args.run_dir, args.run_url)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
