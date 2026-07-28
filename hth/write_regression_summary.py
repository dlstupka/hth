#!/usr/bin/env python3
"""Render a GitHub Actions job summary from a canonical HTH regression run."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


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
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if value < 1:
        return f"{value * 1000:.1f}ms"
    if value < 60:
        return f"{value:.1f}s"
    minutes, secs = divmod(int(round(value)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s" if hours else f"{minutes}m {secs}s"


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
        f"- Parameter sets evaluated: `{summary.get('parameter_set_count', 'unknown')}`",
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
        "| Result | Parameter Short Name | Parameter Set ID | Avg IoU | Min IoU | StdDev | Failures | Evaluation time |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        f"| Winner | `{_parameter_short_name(winner)}` | `{_parameter_id(winner)}` | {_number(winner_stats.get('mean_iou'))} | {_number(winner_stats.get('minimum_iou'))} | {_number(winner_stats.get('stddev_iou'))} | {winner_stats.get('failure_count', 'unknown')} | {_duration(_evaluation_seconds(winner))} |",
    ])
    if baseline and _parameter_id(baseline) != _parameter_id(winner):
        lines.append(
            f"| Baseline | `{_parameter_short_name(baseline)}` | `{_parameter_id(baseline)}` | "
            f"{_number(baseline_stats.get('mean_iou'))} | "
            f"{_number(baseline_stats.get('minimum_iou'))} | "
            f"{_number(baseline_stats.get('stddev_iou'))} | "
            f"{baseline_stats.get('failure_count', 'unknown')} | "
            f"{_duration(_evaluation_seconds(baseline))} |"
        )

    if include_metric_definitions:
        lines.extend([
            "",
            "### Metric Definitions",
            "",
            "- **Avg IoU:** Mean page IoU across the Golden Set.",
            "- **Min IoU:** Lowest page IoU across the Golden Set.",
            "- **StdDev:** Standard deviation of page IoUs across the Golden Set.",
            "- **Failures:** Number of pages that could not be evaluated.",
        ])

    lines.extend([
        "",
        "### Regression Statistics for Detector Calibration",
        "",
        "| Statistic | Count |",
        "|---|---:|",
        f"| Mean IoU improvements | {progress.get('mean_iou_improvements', 0)} |",
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
            "| Rank | Parameter Short Name | Avg IoU | Min IoU | StdDev | Δ Avg IoU | Failures |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ])
        for result in top_parameter_sets[:5]:
            stats = result.get("summary", {}) if isinstance(result, dict) else {}
            mean_iou = float(stats.get("mean_iou", 0.0) or 0.0)
            delta_mean_iou = mean_iou - winner_mean
            parameter_set_name = _parameter_short_name(result)
            failure_count = stats.get("failure_count", "unknown")

            lines.append(
                f"| {result.get('rank', 'unknown')} | `{parameter_set_name}` | "
                f"{_number(mean_iou)} | {_number(stats.get('minimum_iou'))} | "
                f"{_number(stats.get('stddev_iou'))} | {delta_mean_iou:+.4f} | "
                f"{failure_count} |"
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
            "| Golden Set Page | Baseline | Winner | Δ IoU | Status | Parameter Set |",
            "|---:|---:|---:|---:|---|---|",
        ])
        for page in winner_pages:
            lines.append(
                f"| {page.get('golden_set_page', 'unknown')} | "
                f"{_number(page.get('baseline_iou'))} | {_number(page.get('winner_iou'))} | "
                f"{float(page.get('delta_iou', 0.0) or 0.0):+.4f} | "
                f"{page.get('status', 'unknown')} | `{_short(page.get('parameter_set'), 12)}` |"
            )

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
            "### Problem Pages",
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
                "| Golden Set Page | Winner IoU | Problem | Parameter Set |",
                "|---:|---:|---|---|",
            ])
            for page in problem_pages:
                reasons = "; ".join(str(reason) for reason in page.get("problem_reasons", [])) or str(page.get("status", "unknown"))
                lines.append(
                    f"| {page.get('golden_set_page', 'unknown')} | "
                    f"{_number(page.get('winner_iou'))} | {reasons} | "
                    f"`{_short(page.get('parameter_set'), 12)}` |"
                )
        else:
            lines.extend(["", "No problem pages were identified."])

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


def _render_detector_calibration(detector: str, payload: dict[str, Any]) -> list[str]:
    search = payload.get("search", {}) if isinstance(payload.get("search"), dict) else {}
    landscape = payload.get("landscape", {}) if isinstance(payload.get("landscape"), dict) else {}
    confidence = payload.get("calibration_confidence", {}) if isinstance(payload.get("calibration_confidence"), dict) else {}
    parameters = payload.get("parameter_influence", []) if isinstance(payload.get("parameter_influence"), list) else []
    interactions = payload.get("interactions", []) if isinstance(payload.get("interactions"), list) else []
    pages = payload.get("page_sensitivity", []) if isinstance(payload.get("page_sensitivity"), list) else []
    recommendations = payload.get("recommendations", {}) if isinstance(payload.get("recommendations"), dict) else {}

    lines = [
        f"### {detector}",
        "",
        str(payload.get("scope_note") or "Conclusions are specific to this run's Golden Set and parameter grid."),
        "",
        "#### Calibration Landscape",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Search coverage | {'complete exhaustive' if search.get('exhaustive_complete') else 'partial / adaptive'} |",
        f"| Parameter sets evaluated | {search.get('parameter_sets', 'unknown')} |",
        f"| Fully successful parameter sets | {search.get('fully_successful_parameter_sets', 'unknown')} ({_percent(search.get('fully_successful_rate'))}) |",
        f"| Best Avg IoU | {_number(landscape.get('best_mean_iou'))} |",
        f"| Median Avg IoU | {_number(landscape.get('median_mean_iou'))} |",
        f"| 95th-percentile Avg IoU | {_number(landscape.get('p95_mean_iou'))} |",
        f"| Near-best basin (within {float(landscape.get('near_best_tolerance', 0.001) or 0.001):.4f}) | {landscape.get('near_best_count', 'unknown')} ({_percent(landscape.get('near_best_share'))}) |",
        f"| Equivalent-winner basin (within {float(landscape.get('equivalent_tolerance', 0.0001) or 0.0001):.4f}) | {landscape.get('equivalent_winner_count', 'unknown')} ({_percent(landscape.get('equivalent_winner_share'))}) |",
        f"| Calibration confidence | {confidence.get('rating', 'unknown')} |",
    ]
    reasons = confidence.get("reasons", []) if isinstance(confidence.get("reasons"), list) else []
    if reasons:
        lines.extend(["", f"Confidence basis: {', '.join(str(reason) for reason in reasons)}."])

    if parameters:
        lines.extend([
            "",
            "#### Parameter Influence",
            "",
            "Influence uses one-way η² over Avg IoU. It measures association within this configured grid; it does not establish causation.",
            "",
            "| Parameter | Classification | η² | Mean-IoU range | Near-best value coverage | Best observed values |",
            "|---|---|---:|---:|---:|---|",
        ])
        for item in parameters[:12]:
            best_values = item.get("best_values", []) if isinstance(item.get("best_values"), list) else []
            rendered_values = ", ".join(
                f"`{entry.get('value')}` ({_number(entry.get('mean_iou'))})"
                for entry in best_values[:3]
                if isinstance(entry, dict)
            ) or "unknown"
            lines.append(
                f"| `{item.get('parameter', 'unknown')}` | {item.get('classification', 'unknown')} | "
                f"{_number(item.get('eta_squared'))} | {_number(item.get('mean_iou_range'))} | "
                f"{_percent(item.get('near_best_value_coverage'))} | {rendered_values} |"
            )

    dormant = recommendations.get("dormant_parameters", []) if isinstance(recommendations.get("dormant_parameters"), list) else []
    if dormant:
        lines.extend([
            "",
            "#### Dormant Parameters",
            "",
            "These parameters had no material measured effect on Avg IoU for this Golden Set and grid:",
            "",
            ", ".join(f"`{name}`" for name in dormant) + ".",
            "",
            str(recommendations.get("note") or "Re-evaluate dormant parameters whenever the Golden Set changes."),
        ])

    meaningful_interactions = [
        item for item in interactions
        if isinstance(item, dict) and float(item.get("incremental_importance", 0.0) or 0.0) >= 0.001
    ]
    if meaningful_interactions:
        lines.extend([
            "",
            "#### Parameter Interactions",
            "",
            "Pairwise interaction importance is exploratory and estimated from a deterministic sample.",
            "",
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
            "",
            "#### Page Sensitivity",
            "",
            "| Golden Set Page | Mean IoU | Min IoU | Max IoU | StdDev | Success rate |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        for page in pages:
            lines.append(
                f"| {page.get('global_ordinal', 'unknown')} | {_number(page.get('mean_iou'))} | "
                f"{_number(page.get('minimum_iou'))} | {_number(page.get('maximum_iou'))} | "
                f"{_number(page.get('stddev_iou'))} | {_percent(page.get('success_rate'))} |"
            )

    return lines


def _render_calibration_report(run_dirs: list[Path]) -> list[str]:
    available: list[tuple[str, dict[str, Any]]] = []
    missing: list[str] = []
    for run_dir in run_dirs:
        manifest = _read_json(run_dir / "manifest.json")
        detector = str(manifest.get("detector", run_dir.parent.name))
        payload = _calibration_payload(run_dir)
        if payload is None:
            missing.append(detector)
        else:
            available.append((detector, payload))

    lines = [
        "## Detector Calibration Report",
        "",
        "This section characterizes the evaluated calibration landscapes, parameter influence, interactions, optimum-basin width, page sensitivity, and opportunities to reduce future search cost. All findings are corpus- and grid-specific and must be revalidated when the Golden Set or parameter space changes.",
        "",
    ]
    if not available:
        lines.append("Calibration intelligence was not available for these runs.")
        return lines

    lines.extend([
        "### Calibration Overview",
        "",
        "| Detector | Coverage | Parameter sets | Successful | Best Avg IoU | Median Avg IoU | Near-best basin | Equivalent winners | Confidence |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for detector, payload in available:
        search = payload.get("search", {}) if isinstance(payload.get("search"), dict) else {}
        landscape = payload.get("landscape", {}) if isinstance(payload.get("landscape"), dict) else {}
        confidence = payload.get("calibration_confidence", {}) if isinstance(payload.get("calibration_confidence"), dict) else {}
        lines.append(
            f"| `{detector}` | {'complete' if search.get('exhaustive_complete') else 'partial'} | "
            f"{search.get('parameter_sets', 'unknown')} | {_percent(search.get('fully_successful_rate'))} | "
            f"{_number(landscape.get('best_mean_iou'))} | {_number(landscape.get('median_mean_iou'))} | "
            f"{_percent(landscape.get('near_best_share'))} | {_percent(landscape.get('equivalent_winner_share'))} | "
            f"{confidence.get('rating', 'unknown')} |"
        )
    if missing:
        lines.extend(["", "Calibration intelligence unavailable for: " + ", ".join(f"`{name}`" for name in missing) + "."])
    for detector, payload in available:
        lines.extend(["", *_render_detector_calibration(detector, payload)])
    return lines

def _combined_result_row(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    info = _read_json(run_dir / "RUN-INFO.json")
    parameters = _read_json(run_dir / "parameters.json")
    summary = _read_json(run_dir / "reports" / "summary.json")
    winner = summary.get("winner") if isinstance(summary.get("winner"), dict) else None
    winner_stats = winner.get("summary", {}) if winner else {}
    page_ordinals = summary.get("page_ordinals", []) if isinstance(summary.get("page_ordinals"), list) else []
    source_document = _source_document_metadata(run_dir, info, parameters)
    page_rate = _page_rate(winner, len(page_ordinals))
    return {
        "detector": str(manifest.get("detector", run_dir.parent.name)),
        "status": str(manifest.get("status", "unknown")),
        "parameter_short_name": _parameter_short_name(winner),
        "parameter_set_id": _parameter_id(winner),
        "mean_iou": winner_stats.get("mean_iou"),
        "minimum_iou": winner_stats.get("minimum_iou"),
        "stddev_iou": winner_stats.get("stddev_iou"),
        "failures": winner_stats.get("failure_count", "unknown"),
        "parameter_sets": summary.get("parameter_set_count", "unknown"),
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
    lines.extend([
        "",
        "## Ranked detector results",
        "",
        "| Rank | Detector | Status | Parameter Short Name | Parameter Set ID | Avg IoU | Min IoU | StdDev | Failures | Parameter sets | Eval rate | Doc time | Run elapsed |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rank, row in enumerate(combined_rows, start=1):
        lines.append(
            f"| {rank} | `{row['detector']}` | {row['status']} | `{row['parameter_short_name']}` | "
            f"`{row['parameter_set_id']}` | {_number(row['mean_iou'])} | "
            f"{_number(row['minimum_iou'])} | {_number(row['stddev_iou'])} | "
            f"{row['failures']} | {row['parameter_sets']} | "
            f"{_format_page_rate(row['page_rate'])} | {_duration(row['document_seconds'])} | "
            f"{_duration(row['elapsed_seconds'])} |"
        )
    lines.extend([
        "",
        "### Metric Definitions",
        "",
        "- **Avg IoU:** Mean page IoU across the Golden Set.",
        "- **Min IoU:** Lowest page IoU across the Golden Set.",
        "- **StdDev:** Standard deviation of page IoUs across the Golden Set.",
        "- **Failures:** Number of pages that could not be evaluated.",
        "",
    ])
    lines.extend(_render_calibration_report(run_dirs))
    lines.append("")
    for index, run_dir in enumerate(run_dirs):
        manifest = _read_json(run_dir / "manifest.json")
        detector = str(manifest.get("detector", run_dir.parent.name))
        lines.extend([f"## {detector}", ""])
        lines.append(
            build_summary(
                run_dir,
                include_title=False,
                include_metric_definitions=False,
            ).rstrip()
        )
        if index != len(run_dirs) - 1:
            lines.extend(["", "---", ""])
    if run_url:
        lines.extend(["", f"[Open workflow run]({run_url})"])
    lines.append("")
    return "\n".join(lines)


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
