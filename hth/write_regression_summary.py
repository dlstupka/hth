#!/usr/bin/env python3
"""Render a GitHub Actions job summary from a canonical HTH regression run."""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from hth.regression.result_metrics import normalize_summary_metrics
from hth.regression.authoritative_record import authoritative_record
from hth.regression.calibration_intelligence import detector_characterization
from hth.domain.result_metrics import baseline_surpassed, calibration_metric_view, result_metric_view
from hth.runtime_store import coherent_execution_profile, select_runtime_observation
from hth.regression.parameter_provenance import parameter_identity_sha256, resolve_parameter_set




def _detector_characterization(detector: str) -> dict[str, Any]:
    return detector_characterization(detector)


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




def _acceleration(value: Any) -> str:
    try:
        acceleration = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if acceleration <= 0:
        return "unknown"
    return f"{acceleration:.2f}×"

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


def _build_parameter_build_index(
    calibration_index: Path | None,
) -> dict[str, list[dict[str, Any]]]:
    """Read historical parameter provenance once and prepare cheap reverse lookups.

    The previous reporter reopened and reparsed every provenance file for every
    Top Parameter Sets row, then called resolve_parameter_set(), which can scan
    the entire Cartesian grid.  A combined manifest multiplied that work across
    every detector.  This index loads each historical provenance file once and
    keeps only the compact identity/grid metadata needed to answer subsequent
    exact-parameter build queries.
    """
    by_detector: dict[str, list[dict[str, Any]]] = {}
    if calibration_index is None or not calibration_index.is_file():
        return by_detector
    try:
        index = _read_json(calibration_index)
    except (OSError, ValueError, json.JSONDecodeError):
        return by_detector

    for entry in index.get("entries", []):
        if not isinstance(entry, dict):
            continue

        # Provisional calibration records are smoke evidence in HTH's persisted
        # calibration contract.  They are intentionally excluded from human
        # "Known Builds" / "Last Build" history to keep those lists useful.
        if str(entry.get("calibration_status") or "").lower() == "provisional":
            continue

        detector = str(entry.get("detector_id") or "")
        prov_rel = entry.get("parameter_provenance_path")
        if not detector or not prov_rel:
            continue
        prov_path = calibration_index.parent / str(prov_rel)
        if not prov_path.is_file():
            continue
        try:
            provenance = _read_json(prov_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue

        identity = provenance.get("identity", {}) if isinstance(provenance.get("identity"), dict) else {}
        schema = str(identity.get("parameter_schema_version") or "1")
        grid = provenance.get("grid", {}) if isinstance(provenance.get("grid"), dict) else {}
        grid_order = tuple(str(name) for name in grid.get("parameter_order", []))
        grid_values = grid.get("values", {}) if isinstance(grid.get("values"), dict) else {}

        explicit = provenance.get("explicit_parameter_sets", {})
        explicit_shas = frozenset(
            str(full)
            for full in explicit
        ) if isinstance(explicit, dict) else frozenset()

        profile_shas: set[str] = set()
        profiles = provenance.get("profiles", {})
        if isinstance(profiles, dict):
            for parameters in profiles.values():
                if not isinstance(parameters, dict):
                    continue
                profile_shas.add(
                    parameter_identity_sha256(detector, parameters, schema_version=schema)
                )

        build = entry.get("build") if isinstance(entry.get("build"), dict) else {}
        by_detector.setdefault(detector, []).append({
            "build_number": str(build.get("github_run_number") or "unknown"),
            "build_url": str(build.get("run_url") or ""),
            "date": str(entry.get("created_at_utc") or "")[:10],
            "evidence": str(entry.get("calibration_status") or "known"),
            "schema": schema,
            "grid_order": grid_order,
            "grid_values": grid_values,
            "explicit_shas": explicit_shas,
            "profile_shas": frozenset(profile_shas),
        })

    return by_detector


def _parameter_in_index_record(
    record: dict[str, Any],
    *,
    detector: str,
    full_sha: str,
    parameters: dict[str, Any] | None,
) -> bool:
    if full_sha in record.get("explicit_shas", ()):
        return True
    if full_sha in record.get("profile_shas", ()):
        return True
    if not parameters:
        return False

    order = tuple(record.get("grid_order") or ())
    values = record.get("grid_values") if isinstance(record.get("grid_values"), dict) else {}
    if not order or set(parameters) != set(order):
        return False

    # Test grid membership directly instead of scanning itertools.product().
    # JSON canonicalization handles scalar/list values consistently with
    # parameter provenance identity construction.
    for name in order:
        target = json.dumps(parameters[name], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        domain = values.get(name, [])
        if not isinstance(domain, list):
            return False
        if not any(
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == target
            for value in domain
        ):
            return False

    schema = str(record.get("schema") or "1")
    return parameter_identity_sha256(
        detector,
        parameters,
        schema_version=schema,
    ) == full_sha


def _build_sort_key(row: tuple[str, str, str, str]) -> tuple[int, str]:
    number = row[0]
    try:
        numeric = int(number)
    except (TypeError, ValueError):
        numeric = -1
    # GitHub run numbers are monotonic for this workflow/repository and are the
    # clearest definition of newest build; date is only a fallback/tiebreaker.
    return numeric, row[2]


def _known_builds_for_parameter(
    *,
    detector: str,
    full_sha: str,
    parameters: dict[str, Any] | None,
    info: dict[str, Any],
    run_url: str,
    parameter_build_index: dict[str, list[dict[str, Any]]],
) -> list[tuple[str, str, str, str]]:
    builds: list[tuple[str, str, str, str]] = []
    current_number = info.get("github_run_number")
    current_url = info.get("github_run_url") or info.get("run_url") or run_url
    if current_number or current_url:
        builds.append((
            str(current_number or "current"),
            str(current_url or ""),
            str(info.get("started_at_utc") or "")[:10],
            "current run",
        ))

    if full_sha:
        for record in parameter_build_index.get(detector, []):
            if not _parameter_in_index_record(
                record,
                detector=detector,
                full_sha=full_sha,
                parameters=parameters,
            ):
                continue
            row = (
                str(record.get("build_number") or "unknown"),
                str(record.get("build_url") or ""),
                str(record.get("date") or ""),
                str(record.get("evidence") or "known"),
            )
            if row not in builds:
                builds.append(row)

    # The report being generated is the newest observation by definition.
    # Pin it first, then retain newest-to-oldest ordering for filtered prior
    # authoritative history.
    current_rows = [row for row in builds if row[3] == "current run"]
    current_numbers = {row[0] for row in current_rows if row[0] not in {"", "current", "unknown"}}
    current_urls = {row[1] for row in current_rows if row[1]}
    prior_rows = [
        row for row in builds
        if row[3] != "current run"
        and row[0] not in current_numbers
        and row[1] not in current_urls
    ]
    prior_rows.sort(key=_build_sort_key, reverse=True)
    return current_rows + prior_rows


def _last_build_for_parameter(
    result: dict[str, Any] | None,
    *,
    detector: str,
    info: dict[str, Any],
    run_url: str,
    parameter_build_index: dict[str, list[dict[str, Any]]],
) -> str:
    if not result:
        return "unknown"
    full_sha = str(result.get("parameter_identity_sha256") or "")
    if not full_sha:
        return "unknown"
    parameters = result.get("parameters") if isinstance(result.get("parameters"), dict) else None
    builds = _known_builds_for_parameter(
        detector=detector,
        full_sha=full_sha,
        parameters=parameters,
        info={},  # Last Build means the most recent known build prior to this report's run.
        run_url="",
        parameter_build_index=parameter_build_index,
    )
    current_number = str(info.get("github_run_number") or "")
    current_url = str(info.get("github_run_url") or info.get("run_url") or run_url or "")
    builds = [
        row for row in builds
        if not (current_number and row[0] == current_number)
        and not (current_url and row[1] == current_url)
    ]
    if not builds:
        return "—"

    number, url, _, _ = builds[0]
    label = f"#{number}" if number not in {"current", "unknown"} else ("current" if number == "current" else "unknown")
    return f"[{label}]({url})" if url else label


def _parameter_set_details(
    run_dir: Path,
    winner: dict[str, Any],
    *,
    detector: str,
    info: dict[str, Any],
    run_url: str,
    parameter_build_index: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Render exact winner parameters and builds known to use that identity."""
    provenance_path = run_dir / "parameter-provenance.json"
    provenance = _read_json(provenance_path) if provenance_path.is_file() else {}
    identifier = str(winner.get("parameter_identity_sha256") or winner.get("parameter_set_id") or "")
    resolved = resolve_parameter_set(provenance, identifier) if provenance and identifier else None
    parameters = dict((resolved or {}).get("parameters") or winner.get("parameters") or {})
    full_sha = str((resolved or {}).get("sha256") or winner.get("parameter_identity_sha256") or "")
    legacy_id = str((resolved or {}).get("legacy_parameter_set_id") or winner.get("legacy_parameter_set_id") or winner.get("parameter_set_id") or "unknown")
    ordinal = (resolved or {}).get("grid_ordinal", winner.get("parameter_grid_ordinal"))
    grid_sha = winner.get("parameter_grid_sha256") or (provenance.get("grid") or {}).get("sha256")
    identity = provenance.get("identity") if isinstance(provenance.get("identity"), dict) else {}

    lines = [
        "", "### Parameter Set Details", "",
        "This is the exact winning parameter configuration. The short Parameter Set ID is a legacy convenience alias; the full SHA-256 identity is authoritative.", "",
        "| Identity field | Value |", "|---|---|",
        f"| Parameter Set ID (legacy alias) | `{legacy_id}` |",
        f"| Absolute parameter SHA-256 | `{full_sha or 'unavailable (legacy record)'}` |",
        f"| Identity schema | `{identity.get('identity_schema_version', winner.get('parameter_identity_schema_version', 'unknown'))}` |",
        f"| Parameter schema | `{identity.get('parameter_schema_version', winner.get('parameter_schema_version', 'unknown'))}` |",
        f"| Grid SHA-256 | `{grid_sha or 'not a grid member'}` |",
        f"| Grid ordinal | `{ordinal if ordinal is not None else 'not a grid member'}` |",
        f"| Reproducibility | {'**Fully reproducible**' if parameters and full_sha else '**Legacy identity only / exact configuration unavailable**'} |",
        "", "#### Exact Parameter Settings", "",
        "| Parameter | Value |", "|---|---|",
    ]
    if parameters:
        for name in sorted(parameters):
            value = json.dumps(parameters[name], sort_keys=True, ensure_ascii=False)
            lines.append(f"| `{name}` | `{value.replace('`', '&#96;')}` |")
    else:
        lines.append("| _unavailable_ | Legacy record does not contain reconstructable parameter provenance. |")

    builds = _known_builds_for_parameter(
        detector=detector,
        full_sha=full_sha,
        parameters=parameters,
        info=info,
        run_url=run_url,
        parameter_build_index=parameter_build_index,
    )
    lines.extend(["", "#### Known Builds Using This Exact Parameter Set", "", "Builds known at the time this report was generated; matching is by the authoritative full parameter identity.", "", "| Build | Date | Evidence |", "|---|---|---|"])
    if builds:
        for number, url, date, evidence in builds:
            label = f"#{number}" if number != "current" else "current run"
            build_text = f"[{label}]({url})" if url else label
            lines.append(f"| {build_text} | {date or 'unknown'} | {evidence} |")
    else:
        lines.append("| current/unknown | unknown | No build provenance was available. |")
    return lines


def _individual_heading(title: str, detector: str) -> str:
    """Render an individual-manifest section heading with obvious detector context."""
    return f"{title} — {detector}"


def _preferred_execution_shape_lines(info: dict[str, Any], summary: dict[str, Any]) -> list[str]:
    """Render the execution shape actually selected for this detector run.

    New runs persist the resolver source explicitly.  Older records still expose
    their pipeline/thread geometry, so keep them useful while labeling the source
    honestly as legacy/unknown.
    """
    pipeline = info.get("detector_pipeline") if isinstance(info.get("detector_pipeline"), dict) else {}
    if not pipeline and isinstance(summary.get("detector_pipeline"), dict):
        pipeline = summary.get("detector_pipeline", {})
    pipelines = pipeline.get("pipeline_count")
    threads = info.get("threads", summary.get("threads"))
    try:
        allocated = int(pipelines) * int(threads)
    except (TypeError, ValueError):
        allocated = None
    source = str(pipeline.get("execution_shape_source") or "unknown (legacy record)")
    runner = str(info.get("runner_name") or (summary.get("runner") or {}).get("runner_name") or "unknown")
    budget = pipeline.get("execution_thread_budget")
    return [
        "",
        "### Preferred Execution Shape",
        "",
        "The regression execution shape selected for this detector run is recorded here so the calibration result can be interpreted without returning to build provenance.",
        "",
        "| Source | Pipelines | Threads / pipeline | Allocated | Runner | Runner budget |",
        "|---|---:|---:|---:|---|---:|",
        f"| `{source}` | {pipelines if pipelines is not None else 'unknown'} | {threads if threads is not None else 'unknown'} | {allocated if allocated is not None else 'unknown'} | `{runner}` | {budget if budget not in (None, '') else 'unknown'} |",
    ]


def build_summary(
    run_dir: Path,
    run_url: str = "",
    *,
    include_title: bool = True,
    include_metric_definitions: bool = True,
    pipeline_repository: str = "",
    results_repository: str = "",
    results_commit: str = "",
    calibration_index: Path | None = None,
    parameter_build_index: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    manifest = _read_json(run_dir / "manifest.json")
    info = _read_json(run_dir / "RUN-INFO.json")
    parameters = _read_json(run_dir / "parameters.json")
    summary = normalize_summary_metrics(_read_json(run_dir / "reports" / "summary.json"))

    if parameter_build_index is None:
        parameter_build_index = _build_parameter_build_index(calibration_index)

    winner = summary.get("winner") if isinstance(summary.get("winner"), dict) else None
    baseline = summary.get("baseline") if isinstance(summary.get("baseline"), dict) else None
    winner_stats = winner.get("summary", {}) if winner else {}
    baseline = summary.get("baseline") if isinstance(summary.get("baseline"), dict) else None
    baseline_stats = baseline.get("summary", {}) if baseline else {}
    historic_best = summary.get("historic_best") if isinstance(summary.get("historic_best"), dict) else None
    outputs = manifest.get("outputs", []) if isinstance(manifest.get("outputs"), list) else []
    progress = summary.get("progress", {}) if isinstance(summary.get("progress"), dict) else {}
    page_ordinals = summary.get("page_ordinals", []) if isinstance(summary.get("page_ordinals"), list) else []
    configuration = parameters.get("configuration", {}) if isinstance(parameters.get("configuration"), dict) else {}
    profiles = configuration.get("profiles", {}) if isinstance(configuration.get("profiles"), dict) else {}

    detector_name = str(manifest.get("detector", "unknown"))
    lines = []
    if include_title:
        lines.extend(["# Regression Manifest", ""])
    lines.extend([
        f"**Status:** {manifest.get('status', 'unknown')}",
        "",
        f"## {_individual_heading('Run Information', detector_name)}",
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
        f"- Wall-clock elapsed: `{_duration(info.get('wall_elapsed_seconds', info.get('elapsed_seconds')))}`",
        f"- Est. serial runtime: `{_duration(info.get('estimated_serial_runtime_seconds', summary.get('estimated_serial_runtime_seconds')))}`",
        f"- Effective acceleration: `{_acceleration(info.get('effective_acceleration', summary.get('effective_acceleration')))}`",
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

    golden_set_id, _ = _golden_set_identity(run_dir, info, parameters, summary)
    detector_config_sha = str(
        info.get("detector_config_sha256")
        or parameters.get("detector_config_sha256")
        or "unknown"
    )
    detector_config_id = _short(detector_config_sha, 12)
    implementation_id = _short(
        info.get("pipeline_commit")
        or info.get("source_commit")
        or summary.get("source_commit")
        or "unknown",
        12,
    )

    lines.extend([
        "",
        f"## {_individual_heading('Results', detector_name)}",
        "",
        "### Result",
        "",
        "| Result | Golden Set ID | Detector Config ID* | Parameter Set ID | Parameter Short Name | Avg IoU | Min IoU | StdDev | Avg IoU Success | Failures | Evaluation Time |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        f"| Winner | `{golden_set_id}` | `{detector_config_id}` | `{_parameter_id(winner)}` | `{_parameter_short_name(winner)}` | {_number(winner_stats.get('mean_iou'))} | {_number(winner_stats.get('minimum_iou'))} | {_number(winner_stats.get('stddev_iou'))} | {_number(winner_stats.get('mean_iou_success', winner_stats.get('mean_iou')))} | {winner_stats.get('failure_count', 'unknown')} | {_duration(_evaluation_seconds(winner))} |",
    ])
    if baseline and _parameter_id(baseline) != _parameter_id(winner):
        lines.append(
            f"| Baseline | `{golden_set_id}` | `{detector_config_id}` | `{_parameter_id(baseline)}` | `{_parameter_short_name(baseline)}` | "
            f"{_number(baseline_stats.get('mean_iou'))} | "
            f"{_number(baseline_stats.get('minimum_iou'))} | "
            f"{_number(baseline_stats.get('stddev_iou'))} | "
            f"{_number(baseline_stats.get('mean_iou_success', baseline_stats.get('mean_iou')))} | "
            f"{baseline_stats.get('failure_count', 'unknown')} | "
            f"{_duration(_evaluation_seconds(baseline))} |"
        )

    lines.extend([
        "",
        f"\\* **Detector Config ID** is the short SHA-256 of the detector configuration used by this run. "
        f"It identifies the declared detector settings, while detector implementation identity is pinned by "
        f"the run's pipeline/source revision (`{implementation_id}`). Exact regression-result reproducibility "
        f"requires the tuple **detector implementation + parameter set + Golden Set**; matching a parameter "
        f"SHA alone does not imply identical results across detector-code revisions.",
    ])

    lines.extend(_parameter_set_details(
        run_dir,
        winner or {},
        detector=detector_name,
        info=info,
        run_url=run_url,
        parameter_build_index=parameter_build_index,
    ))

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
            "- **Avg IoU:** Arithmetic mean across every Golden Set page; failed/no-candidate pages contribute `0.0000` and remain in the denominator. This is the primary detector-ranking metric.",
            "- **Avg IoU Success:** Arithmetic mean across successful Golden Set page evaluations only; failed/no-candidate pages are excluded.",
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
        f"| Baseline surpassed | {'yes' if baseline_surpassed(winner, baseline) else 'no'} |",
    ])
    lines.extend(_preferred_execution_shape_lines(info, summary))

    search_top_parameter_sets = summary.get("search_top_parameter_sets", [])
    if not isinstance(search_top_parameter_sets, list):
        search_top_parameter_sets = []
    if not search_top_parameter_sets:
        legacy_top = summary.get("top_parameter_sets", [])
        if isinstance(legacy_top, list):
            reference_ids = {
                _parameter_id(result)
                for result in (baseline, historic_best)
                if isinstance(result, dict)
            }
            search_top_parameter_sets = [
                result for result in legacy_top
                if isinstance(result, dict) and _parameter_id(result) not in reference_ids
            ]

    reference_rows: list[tuple[str, dict[str, Any]]] = []
    if baseline:
        reference_rows.append(("Baseline*", baseline))
    if historic_best and _parameter_id(historic_best) != _parameter_id(baseline):
        reference_rows.append(("Best**", historic_best))

    if reference_rows or search_top_parameter_sets:
        winner_mean = float(winner_stats.get("mean_iou", 0.0) or 0.0)
        lines.extend([
            "",
            "### Top Parameter Sets",
            "",
            "| Rank | Last Build | Parameter Set ID | Parameter Short Name | Avg IoU | Min IoU | StdDev | Δ Avg IoU | Avg IoU Success | Failures | Discovery Time | Search Space % |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])

        def append_parameter_row(rank_label: str, result: dict[str, Any], *, reference: bool) -> None:
            stats = result.get("summary", {}) if isinstance(result, dict) else {}
            mean_iou = float(stats.get("mean_iou", 0.0) or 0.0)
            delta_mean_iou = mean_iou - winner_mean
            parameter_set_name = _parameter_short_name(result)
            failure_count = stats.get("failure_count", "unknown")
            observation = _search_observation(result)
            last_build = _last_build_for_parameter(
                result,
                detector=detector_name,
                info=info,
                run_url=run_url,
                parameter_build_index=parameter_build_index,
            )
            lines.append(
                f"| {rank_label} | {last_build} | `{_parameter_id(result)}` | `{parameter_set_name}` | "
                f"{_number(mean_iou)} | {_number(stats.get('minimum_iou'))} | "
                f"{_number(stats.get('stddev_iou'))} | {delta_mean_iou:+.4f} | "
                f"{_number(stats.get('mean_iou_success', stats.get('mean_iou')))} | {failure_count} | "
                f"{'reference' if reference else _duration(observation.get('elapsed_seconds'))} | "
                f"{'reference' if reference else _search_space_percent(observation)} |"
            )

        for label, result in reference_rows:
            append_parameter_row(label, result, reference=True)

        shown_reference_ids = {_parameter_id(result) for _, result in reference_rows}
        displayed = 0
        for result in search_top_parameter_sets:
            if _parameter_id(result) in shown_reference_ids:
                continue
            displayed += 1
            append_parameter_row(str(result.get("search_rank", displayed)), result, reference=False)
            if displayed >= 5:
                break

        lines.extend([
            "",
            r"\* **Baseline** is the detector's default parameter-set configuration.",
            r"\*\* **Best** is the historic best-known compatible parameter set prior to this regression run.",
            "",
            "**Last Build** is the most recent known prior build that evaluated the exact absolute parameter identity; the current manifest build is intentionally excluded.",
            "",
            "Baseline and Best are mandatory evaluated references in every regression and are not assigned numeric search ranks. If either exact reference parameter set is also present in the requested search, it is evaluated once and shown only as the reference row.",
        ])

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
            f"## {_individual_heading('Page Analysis', detector_name)}",
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
            lines.append("| — | no history | no history | no history |")
        lines.extend([
            "",
            f"Total winner changes: **{progress.get('winner_changes', 0)}**.",
            f"Search completed in **{_duration(info.get('wall_elapsed_seconds', info.get('elapsed_seconds')))}** wall-clock time.",
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

    if include_title:
        best_known = _best_known_calibrations(calibration_index, current_runs=[run_dir])
        best_known_lines = _render_best_known_calibrations(
            best_known,
            heading_level=2,
            results_repository=results_repository,
            results_ref=results_commit or "main",
        )
        if best_known_lines and best_known_lines[0].startswith("## "):
            best_known_lines[0] = f"## {_individual_heading('Best Known Detector Calibrations', detector_name)}"
        lines.extend(["", *best_known_lines])

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
            f"## {_individual_heading('Calibration Intelligence', detector_name)}",
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
            "**Search strategy legend:** `exhaustive` covers the current live declared Cartesian space and keeps configured zombie parameters pinned; `exhaustive-with-zombies` deliberately restores retained zombie value domains for revalidation. Effect-size strategies operate on the current live space.",
            "",
        ])
        lines.extend([
            "### Detector-Selection Intelligence",
            "",
            f"- Recommended parameter set: `{_parameter_id(winner)}`",
            f"- Recommended parameter short name: `{_parameter_short_name(winner)}`",
            f"- Best observed Avg IoU: `{_number(winner_stats.get('mean_iou'))}`",
            f"- Avg IoU Success: `{_number(winner_stats.get('mean_iou_success', winner_stats.get('mean_iou')))}`",
            f"- Worst Golden Set page (Min IoU): `{_number(winner_stats.get('minimum_iou'))}`",
            f"- Page-to-page StdDev: `{_number(winner_stats.get('stddev_iou'))}`",
            f"- Calibration evidence: `{confidence.get('rating', 'unknown')}`",
            f"- Dormant parameters: `{', '.join(str(v) for v in recommendations.get('dormant_parameters', [])) if isinstance(recommendations.get('dormant_parameters'), list) and recommendations.get('dormant_parameters') else 'none'}`",
            f"- Configured zombie parameters: `{', '.join(str(v) for v in recommendations.get('configured_zombie_parameters', [])) if isinstance(recommendations.get('configured_zombie_parameters'), list) and recommendations.get('configured_zombie_parameters') else 'none'}`",
            f"- Available domain spaces: `{', '.join(str(key) for key, value in domain_space.items() if isinstance(value, dict) and int(value.get('parameter_set_count', 0) or 0) > 0) or 'none'}`",
            "",
            "### Calibration Analysis",
            "",
        ])
        lines.extend(_render_detector_calibration(detector_name, calibration_payload, summary))

    if calibration_payload is not None and include_title:
        engineering_lines = _engineering_continuous_improvement_lines(
            run_url=run_url,
            pipeline_repository=pipeline_repository,
            results_repository=results_repository,
            results_commit=results_commit,
        )
        if engineering_lines and engineering_lines[0] == "## Engineering Continuous Improvement":
            engineering_lines[0] = f"## {_individual_heading('Engineering Continuous Improvement', detector_name)}"
        lines.extend(["", *engineering_lines])

    lines.append("")
    rendered_lines = _add_report_navigation(lines) if include_title else lines
    return "\n".join(rendered_lines)




def _percent(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "unknown"


def _display_golden_set_id(value: Any) -> str:
    """Return a canonical display form for Golden Set identifiers."""
    text = str(value or "unknown").strip()
    return text if not text or text.lower() == "unknown" else text.upper()


def _markdown_table_row(cells: list[str], *, bold: bool = False) -> str:
    """Render a Markdown table row, optionally emphasizing every cell."""
    rendered = [f"**{cell}**" if bold else cell for cell in cells]
    return "| " + " | ".join(rendered) + " |"


def _calibration_payload(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "reports" / "calibration-intelligence.json"
    if not path.is_file():
        return None
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if payload.get("available") else None


def _golden_set_identity(run_dir: Path, info: dict[str, Any], parameters: dict[str, Any], summary: dict[str, Any]) -> tuple[str, str]:
    golden_sha = str(info.get("golden_set_sha256") or parameters.get("golden_set_sha256") or summary.get("golden_set_sha256") or "unknown")
    golden_set = info.get("golden_set") or parameters.get("golden_set")
    if golden_set:
        configured = Path(str(golden_set))
        candidates = [configured]
        if not configured.is_absolute():
            candidates.extend([run_dir / configured, Path.cwd() / configured])
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                payload = _read_json(candidate)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            for key in ("collection_id", "id", "name"):
                value = payload.get(key)
                if value:
                    return str(value), golden_sha
    return "unknown", golden_sha


def _calibration_search_type(entry: dict[str, Any], payload: dict[str, Any]) -> str:
    if str(entry.get("calibration_status") or "").lower() == "provisional":
        return "smoke"
    search = payload.get("search", {}) if isinstance(payload.get("search"), dict) else {}
    strategy = search.get("strategy") or (entry.get("search") or {}).get("strategy")
    if search.get("exhaustive_complete"):
        return "exhaustive"
    return str(strategy or "unknown").replace("_", "-")


def _calibration_record_from_payload(
    detector: str,
    payload: dict[str, Any],
    *,
    entry: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
    golden_set_id: str = "unknown",
    golden_set_sha256: str = "unknown",
) -> dict[str, Any]:
    entry = entry or {}
    summary = summary or {}
    search = payload.get("search", {}) if isinstance(payload.get("search"), dict) else {}
    landscape = payload.get("landscape", {}) if isinstance(payload.get("landscape"), dict) else {}
    confidence = payload.get("calibration_confidence", {}) if isinstance(payload.get("calibration_confidence"), dict) else {}
    selection = payload.get("detector_selection_intelligence", {}) if isinstance(payload.get("detector_selection_intelligence"), dict) else {}
    identity = payload.get("calibration_identity", {}) if isinstance(payload.get("calibration_identity"), dict) else {}
    winner = summary.get("winner") if isinstance(summary.get("winner"), dict) else None
    baseline = summary.get("baseline") if isinstance(summary.get("baseline"), dict) else None
    winner_stats = winner.get("summary", {}) if winner else {}
    baseline_stats = baseline.get("summary", {}) if baseline else {}
    metric_view = calibration_metric_view(payload, summary)
    mean_iou = metric_view.get("mean_iou")
    mean_iou_success = metric_view.get("mean_iou_success")
    minimum_iou = metric_view.get("minimum_iou")
    stddev_iou = metric_view.get("stddev_iou")
    failures = metric_view.get("failure_count", "unknown")
    parameter_id = selection.get("recommended_parameter_set_id") or (entry.get("selection") or {}).get("recommended_parameter_set_id") or _parameter_id(winner)
    baseline_mean = baseline_stats.get("mean_iou")
    delta = None
    try:
        if baseline_mean is not None and mean_iou is not None:
            delta = float(mean_iou) - float(baseline_mean)
    except (TypeError, ValueError):
        delta = None
    evidence = selection.get("calibration_evidence")
    if isinstance(evidence, dict):
        evidence = evidence.get("rating")
    evidence = evidence or confidence.get("rating") or (entry.get("selection") or {}).get("calibration_evidence")
    date_value = entry.get("created_at_utc") or identity.get("created_at_utc") or entry.get("published_at_utc")
    date_text = str(date_value or "unknown")[:10]
    actual_golden_id = str(entry.get("golden_set_id") or golden_set_id or "unknown")
    actual_golden_sha = str(entry.get("golden_set_sha256") or golden_set_sha256 or "unknown")
    status = str(entry.get("calibration_status") or ("authoritative" if search.get("exhaustive_complete") else "partial"))
    build = entry.get("build") if isinstance(entry.get("build"), dict) else identity.get("build")
    build = build if isinstance(build, dict) else {}
    return {
        "detector": detector,
        "golden_set_id": actual_golden_id,
        "golden_set_sha256": actual_golden_sha,
        "date": date_text,
        "created_at_utc": str(date_value or ""),
        "search_type": _calibration_search_type({**entry, "calibration_status": status}, payload),
        "status": status,
        "parameter_set_id": _short(parameter_id, 12),
        "role": _detector_characterization(detector).get("role", "Unknown"),
        "coverage": "complete" if search.get("exhaustive_complete") else "partial",
        "parameter_sets": search.get("parameter_sets", (entry.get("search") or {}).get("parameter_sets", "unknown")),
        "successful_rate": search.get("fully_successful_rate"),
        "mean_iou": mean_iou,
        "mean_iou_success": mean_iou_success,
        "minimum_iou": minimum_iou,
        "stddev_iou": stddev_iou,
        "failures": failures,
        "delta_baseline_mean_iou": delta,
        "near_best_share": landscape.get("near_best_share", selection.get("near_best_coverage")),
        "equivalent_winner_share": landscape.get("equivalent_winner_share", selection.get("equivalent_best_coverage")),
        "calibration_evidence": evidence or "unknown",
        "build_name": build.get("workflow") or "unknown",
        "build_number": build.get("github_run_number") or "unknown",
        "build_url": build.get("run_url") or "",
        "run_time_seconds": build.get("run_time_seconds", summary.get("estimated_serial_runtime_seconds", summary.get("elapsed_seconds"))),
        "intelligence_path": str(entry.get("intelligence_path") or ""),
    }


def _best_known_calibrations(
    calibration_index: Path | None,
    *,
    current_runs: list[Path],
) -> list[dict[str, Any]]:
    current_records: list[dict[str, Any]] = []
    current_sha = "unknown"
    for run_dir in current_runs:
        manifest = _read_json(run_dir / "manifest.json")
        info = _read_json(run_dir / "RUN-INFO.json")
        parameters = _read_json(run_dir / "parameters.json")
        summary = normalize_summary_metrics(_read_json(run_dir / "reports" / "summary.json"))
        detector = str(manifest.get("detector", run_dir.parent.name))
        golden_id, golden_sha = _golden_set_identity(run_dir, info, parameters, summary)
        if current_sha == "unknown":
            current_sha = golden_sha
        payload = _calibration_payload(run_dir)
        if payload:
            current_records.append(_calibration_record_from_payload(
                detector, payload, summary=summary, golden_set_id=golden_id, golden_set_sha256=golden_sha,
                entry={
                    "calibration_status": "provisional" if int(summary.get("parameter_set_count", 0) or 0) <= 20 else ("authoritative" if (payload.get("search") or {}).get("exhaustive_complete") else "partial"),
                    "created_at_utc": info.get("started_at_utc"),
                    "build": {
                        "github_run_number": info.get("github_run_number"),
                        "run_url": info.get("github_run_url") or info.get("run_url"),
                        "run_time_seconds": info.get("estimated_serial_runtime_seconds", info.get("elapsed_seconds")),
                    },
                },
            ))

    indexed_records: list[dict[str, Any]] = []
    if calibration_index and calibration_index.is_file():
        try:
            index = _read_json(calibration_index)
        except (OSError, ValueError, json.JSONDecodeError):
            index = {}
        for entry in index.get("entries", []):
            if not isinstance(entry, dict):
                continue
            if current_sha != "unknown" and str(entry.get("golden_set_sha256") or "unknown") != current_sha:
                continue
            intelligence_path = calibration_index.parent / str(entry.get("intelligence_path") or "")
            if not intelligence_path.is_file():
                continue
            try:
                payload = _read_json(intelligence_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            record_dir = calibration_index.parent / str(entry.get("record_path") or "")
            summary_path = record_dir / "summary.json"
            info_path = record_dir / "RUN-INFO.json"
            summary = normalize_summary_metrics(_read_json(summary_path)) if summary_path.is_file() else {}
            info = _read_json(info_path) if info_path.is_file() else {}
            indexed_entry = dict(entry)
            indexed_build = dict(entry.get("build")) if isinstance(entry.get("build"), dict) else {}
            if indexed_build.get("run_time_seconds") is None:
                runtime_value = info.get("estimated_serial_runtime_seconds", info.get("elapsed_seconds"))
                if runtime_value is not None:
                    indexed_build["run_time_seconds"] = runtime_value
            indexed_entry["build"] = indexed_build
            detector = str(entry.get("detector_id") or payload.get("detector") or "unknown")
            indexed_records.append(_calibration_record_from_payload(detector, payload, entry=indexed_entry, summary=summary))

    # Current runs and persisted calibration-index records are one provenance
    # population.  Never let the existence of the index hide a current run.
    # Authoritative/full provenance gates smoke and partial evidence; within the
    # authoritative population, keep the strongest measured calibration as the
    # best-known incumbent instead of letting recency alone downgrade it.
    candidates = indexed_records + current_records
    records_by_detector: dict[str, list[dict[str, Any]]] = {}
    for record in candidates:
        detector = str(record.get("detector"))
        records_by_detector.setdefault(detector, []).append(record)

    selected: list[dict[str, Any]] = []
    for records in records_by_detector.values():
        record = authoritative_record(records)
        if record is not None:
            selected.append(record)

    return sorted(selected, key=_combined_ranking_key)


def _persistent_intelligence_url(
    records: list[dict[str, Any]],
    *,
    results_repository: str = "",
    results_ref: str = "main",
) -> str:
    if not results_repository:
        return ""
    for row in records:
        intelligence_path = str(row.get("intelligence_path") or "")
        if intelligence_path:
            return _github_url(results_repository) + "/blob/" + (results_ref or "main") + "/" + intelligence_path
    return ""


def _build_link_footnote(
    records: list[dict[str, Any]],
    *,
    results_repository: str = "",
    results_ref: str = "main",
) -> str:
    intelligence_url = _persistent_intelligence_url(
        records,
        results_repository=results_repository,
        results_ref=results_ref,
    )
    return (
        "- **Build*:** `#run` links open GitHub Actions logs and artifacts and expire according to repository retention; "
        "the calibration data persists in "
        + _markdown_link("calibration-intelligence.json", intelligence_url)
        + "."
    )


def _calibration_approval_level(search_type: Any, evidence: Any) -> str:
    """Return the deterministic Golden Set-scoped engineering approval level."""
    search = str(search_type or "unknown").strip().lower().replace("_", "-")
    rating = str(evidence or "unknown").strip().lower()
    if search == "smoke" or search == "unknown" or rating == "unknown":
        return "Provisional"
    if search != "exhaustive":
        return "Candidate"
    if rating == "high":
        return "Approved"
    if rating == "medium":
        return "Recommended"
    return "Candidate"


def _render_best_known_calibrations(
    records: list[dict[str, Any]],
    *,
    heading_level: int = 3,
    results_repository: str = "",
    results_ref: str = "main",
    include_build_footnote: bool = True,
) -> list[str]:
    heading = "#" * heading_level
    lines = [
        f"{heading} Best Known Detector Calibrations", "",
        "**Engineering Decision**", "",
        "This table is the authoritative detector ranking for this Golden Set. The Rank #1 detector is the current engineering recommendation based on the best approved calibration available for this Golden Set.", "",
        "This table prefers compatible full calibrations when available and falls back to the latest smoke evidence for detectors without a full calibration on this Golden Set.", "",
        "**Parameter-space note:** `Parameter Sets` is the declared discrete calibration grid for that run. `exhaustive` means every valid set in that declared grid was evaluated; it does not imply every value in an underlying continuous mathematical domain was tested. Invalid combinations should be rejected before evaluation, while behaviorally redundant/no-op combinations should be canonicalized so they do not inflate search or basin statistics.", "",
        "| Rank | Detector | Detector ID | Role | Golden Set ID | Date | Build* | Est. Serial Runtime** | Parameter Set ID | Parameter Sets | Search Type | Successful Parameter Sets | Best Avg IoU | Min IoU | StdDev | Avg IoU Success | Failures | Δ Baseline Avg IoU | Near-best Coverage (Basin) | Equivalent Best Configurations | Calibration Evidence | Approval Level |",
        "|---:|---|---|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, row in enumerate(records, start=1):
        delta = row.get("delta_baseline_mean_iou")
        delta_text = f"{float(delta):+.4f}" if delta is not None else "unknown"
        build_number = str(row.get("build_number") or "unknown")
        build_label = f"#{build_number}" if build_number != "unknown" else "unknown"
        build_text = _markdown_link(build_label, str(row.get("build_url") or ""))
        search_type = row.get("search_type", "unknown")
        calibration_evidence = row.get("calibration_evidence", "unknown")
        approval_level = _calibration_approval_level(search_type, calibration_evidence)
        cells = [
            str(rank),
            _detector_friendly_name(str(row["detector"])),
            f"`{row['detector']}`",
            str(row.get("role", "Unknown")),
            f"`{_display_golden_set_id(row.get('golden_set_id', 'unknown'))}`",
            str(row.get("date", "unknown")),
            build_text,
            _duration(row.get("run_time_seconds")),
            f"`{row.get('parameter_set_id', 'unknown')}`",
            str(row.get("parameter_sets", "unknown")),
            str(search_type),
            _percent(row.get("successful_rate")),
            _number(row.get("mean_iou")),
            _number(row.get("minimum_iou")),
            _number(row.get("stddev_iou")),
            _number(row.get("mean_iou_success", row.get("mean_iou"))),
            str(row.get("failures", "unknown")),
            delta_text,
            _percent(row.get("near_best_share")),
            _percent(row.get("equivalent_winner_share")),
            str(calibration_evidence),
            approval_level,
        ]
        lines.append(_markdown_table_row(cells, bold=(rank == 1)))
    if not records:
        lines.append("| — | No compatible calibration evidence available | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |")
    if include_build_footnote:
        lines.extend([
            "",
            _build_link_footnote(
                records,
                results_repository=results_repository,
                results_ref=results_ref,
            ),
            r"- **Est. Serial Runtime\*\*:** Estimated single-detector serial runtime derived from recorded regression evidence; actual wall time varies with parallelism and scheduling.",
        ])
    return lines


def _detector_summary_and_roi(detector: str, payload: dict[str, Any]) -> tuple[list[str], str]:
    search = payload.get("search", {}) if isinstance(payload.get("search"), dict) else {}
    landscape = payload.get("landscape", {}) if isinstance(payload.get("landscape"), dict) else {}
    parameters = payload.get("parameter_influence", []) if isinstance(payload.get("parameter_influence"), list) else []
    pages = payload.get("page_sensitivity", []) if isinstance(payload.get("page_sensitivity"), list) else []
    measurement = payload.get("measurement_state", {}) if isinstance(payload.get("measurement_state"), dict) else {}
    if measurement and not bool(measurement.get("informative", True)):
        status = str(measurement.get("status") or "unavailable")
        if status == "no_valid_measurements":
            findings = [
                "Calibration did not produce a valid measurement: no evaluated page returned a usable detector candidate.",
                "The zero Avg IoU values are failure placeholders, not evidence of a flat calibration landscape or dormant parameters.",
            ]
            reason_counts = measurement.get("failure_reason_counts") if isinstance(measurement.get("failure_reason_counts"), dict) else {}
            if reason_counts:
                rendered = ", ".join(f"{reason} ({count})" for reason, count in reason_counts.items())
                findings.append(f"Observed detector failure reasons: {rendered}.")
        else:
            findings = [
                "Calibration did not produce a usable quality signal: detector candidates were returned, but none had positive overlap with an approved Golden Set bounding box.",
                "The all-zero Avg IoU field must not be interpreted as parameter equivalence or parameter dormancy.",
            ]
        return findings, "Do not expand or reduce the parameter search yet. Inspect detector inference/debug evidence and restore a valid overlap signal before drawing tuning-ROI conclusions."
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
        f"| Est. serial runtime for full parameter set evaluation* | {full_search_metrics[2]} |",
        f"| Fully successful parameter sets | {search.get('fully_successful_parameter_sets', 'unknown')} ({_percent(search.get('fully_successful_rate'))}) |",
        *([f"| Calibration signal | {measurement.get('status', 'unavailable')} |"] if (measurement := payload.get("measurement_state", {})) and not bool(measurement.get("informative", True)) else []),
        f"| Best Avg IoU | {_number(landscape.get('best_mean_iou'))} |",
        f"| Minimum Avg IoU | {_number(landscape.get('minimum_mean_iou'))} |",
        f"| Avg IoU StdDev | {_number(landscape.get('stddev_mean_iou'))} |",
        f"| Winner stabilized after | {_pluralized_parameter_sets(winner_observation.get('parameter_set_number'))} |",
        f"| Winner stabilized | {_compact_duration(winner_observation.get('elapsed_seconds'))} ({_compact_percent(winner_observation.get('search_fraction'))} of search) |",
        f"| Near-best coverage (basin; within {float(landscape.get('near_best_tolerance', 0.001) or 0.001):.4f}) | {landscape.get('near_best_count', 'unknown')} ({_percent(landscape.get('near_best_share'))}) |",
        f"| Equivalent-best configurations (within {float(landscape.get('equivalent_tolerance', 0.0001) or 0.0001):.4f}) | {landscape.get('equivalent_winner_count', 'unknown')} ({_percent(landscape.get('equivalent_winner_share'))}) |",
        f"| Calibration Evidence | {confidence.get('rating', 'unknown')} |",
        "",
        r"\* **Serial-runtime note:** Long parameter-set estimates assume a single-threaded serial run at the measured detector page rate. Actual wall time varies with parallelization, worker count, scheduling overhead, and parameter-dependent runtime.",
    ])

    domain_space = payload.get("domain_space", {}) if isinstance(payload.get("domain_space"), dict) else {}
    measurement = payload.get("measurement_state", {}) if isinstance(payload.get("measurement_state"), dict) else {}
    if measurement and not bool(measurement.get("informative", True)):
        lines.extend([
            "", "#### Parameter Set Domain Space Reduction", "",
            "Withheld: effect-size reduction is not meaningful until calibration produces valid positive-overlap measurements.",
        ])
    elif domain_space:
        exhaustive_count = int((domain_space.get("exhaustive") or {}).get("parameter_set_count", 0) or 0)
        zombie_exhaustive_count = int((domain_space.get("exhaustive_with_zombies") or {}).get("parameter_set_count", exhaustive_count) or exhaustive_count)
        exhaustive_time = full_search_metrics[2]
        estimated_full_seconds = None
        if zombie_exhaustive_count and page_rate and page_count:
            estimated_full_seconds = zombie_exhaustive_count * page_count / page_rate
        lines.extend([
            "", "#### Parameter Set Domain Space Reduction", "",
            "| Effect Size Group | Parameter Sets | % All Sets | New Time Est* | Set Reduction Factor |",
            "|---|---:|---:|---:|---:|",
        ])
        for key, label in (("exhaustive_with_zombies", "Exhaustive-with-zombies"), ("exhaustive", "Exhaustive"), ("non_dormant", "Non-dormant"), ("low_plus", "Low+"), ("moderate_plus", "Moderate+"), ("important_plus", "Important+"), ("critical", "Critical")):
            entry = domain_space.get(key)
            if not isinstance(entry, dict):
                continue
            count_value = int(entry.get("parameter_set_count", 0) or 0)
            percent = count_value / zombie_exhaustive_count if zombie_exhaustive_count else 0.0
            seconds = estimated_full_seconds * percent if estimated_full_seconds is not None else None
            factor = zombie_exhaustive_count / count_value if count_value else None
            factor_text = f"{factor:.1f}×" if factor is not None else "unavailable"
            lines.append(f"| {label} | {count_value} | {_percent(percent)} | {_duration(seconds)} | {factor_text} |")
        lines.extend([
            "",
            r"\* Uses the same serial measured-page-rate assumptions as the Calibration Landscape serial-runtime estimate.",
            "",
            "*Contracted-search note: every excluded parameter is pinned to its detector baseline value, so reduced-grid parameter identities are stable across builds. Under the current HTH classification thresholds, `Non-dormant` and `Low+` intentionally resolve to the same parameter domain; both labels are retained for compatibility and possible future policy differentiation.*",
        ])
    reasons = confidence.get("reasons", []) if isinstance(confidence.get("reasons"), list) else []
    if reasons:
        lines.extend(["", f"Calibration evidence basis: {', '.join(str(reason) for reason in reasons)}."])

    if parameters:
        lines.extend([
            "", "#### Parameter Influence", "",
            "Influence uses one-way η² over Avg IoU. It measures association within this configured grid; it does not establish causation.", "",
            "| Parameter | Classification | η² | Avg-IoU range | Near-best value coverage | Best observed values | Evidence |",
            "|---|---|---:|---:|---:|---|---|",
        ])
        for item in parameters:
            best_values = item.get("best_values", []) if isinstance(item.get("best_values"), list) else []
            rendered_values = ", ".join(
                f"`{entry.get('value')}` ({_number(entry.get('mean_iou'))})"
                for entry in best_values[:3] if isinstance(entry, dict)
            ) or "unknown"
            coverage = _percent(item.get("near_best_value_coverage")) if item.get("near_best_value_coverage") is not None else "unknown"
            lines.append(
                f"| `{item.get('parameter', 'unknown')}` | {item.get('classification', 'unknown')} | "
                f"{_number(item.get('eta_squared'))} | {_number(item.get('mean_iou_range'))} | "
                f"{coverage} | {rendered_values} | {item.get('evidence_source', 'current run')} |"
            )
        if any(bool(item.get("retained")) for item in parameters):
            lines.extend([
                "",
                "*Dormant and Zombie are canonical measured effect-size classifications, not synonyms. Retained rows were not varied in this run: their last compatible audited measurements are shown for visibility only and do not contribute to this run's search-space counts, influence calculations, interactions, or winner selection.*",
            ])
        lines.extend([
            "", "#### Parameter Classification", "",
            "| Classification | Canonical HTH criterion | Engineering interpretation |",
            "|---|---|---|",
            "| Zombie | η² < 0.0005 **and** Avg-IoU range < 0.0005 | Practically indistinguishable from zero in this characterized grid |",
            "| Dormant | η² < 0.005, excluding Zombie | Measurable or potentially measurable, but operationally negligible |",
            "| Low | 0.005 ≤ η² < 0.02 | Small effect |",
            "| Moderate | 0.02 ≤ η² < 0.06 | Meaningful secondary influence |",
            "| Important | 0.06 ≤ η² < 0.14 | Strong influence |",
            "| Critical | η² ≥ 0.14 | Dominant influence |",
        ])

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


def _render_calibration_report(
    run_dirs: list[Path],
    combined_rows: list[dict[str, Any]],
    calibration_index: Path | None = None,
    *,
    results_repository: str = "",
    results_ref: str = "main",
) -> list[str]:
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

    best_known = _best_known_calibrations(calibration_index, current_runs=run_dirs)
    lines.extend(_render_best_known_calibrations(
        best_known,
        heading_level=3,
        results_repository=results_repository,
        results_ref=results_ref,
        include_build_footnote=False,
    ))
    lines.extend([
        "", "### Calibration Report Legend", "",
        "- **Generator:** proposes an original page boundary from its primary visual evidence.",
        "- **Validator:** scores or confirms a hypothesis generated elsewhere without normally proposing a competing boundary.",
        "- **Hybrid (detectors):** combines the named generator and validator or fuses the named generators.",
        "- **Critical / Important / Moderate / Low / Dormant:** plain-English parameter-influence classes, from dominant measured association to no material measured effect in this grid.",
        "- **Near-best coverage (basin):** share of tested parameter sets within the displayed tolerance of the best Avg IoU; broader coverage indicates more forgiving calibration.",
        "- **Equivalent best configurations:** share of tested sets effectively tied with the best result at the stricter displayed tolerance.",
        "- **Calibration Evidence:** deterministic evidence score for how completely this run characterizes the evaluated Golden Set and parameter grid. Score 2 points for complete exhaustive coverage, 1 point when at least 90% of parameter sets succeed on every page, and 1 point when at least 1% of tested sets are within 0.001 Avg IoU of the winner. **Low** = 0–1 points, **Medium** = 2–3 points, and **High** = 4 points. This is not confidence that the detector generalizes beyond this Golden Set and grid.",
        "- **Approval Level:** automatic Golden Set-scoped engineering status derived from Search Type and Calibration Evidence. **Provisional** = smoke or unavailable evidence; **Candidate** = any reduced search or exhaustive search with Low evidence; **Recommended** = exhaustive search with Medium evidence; **Approved** = exhaustive search with High evidence. A different Golden Set requires its own calibration and approval.",
        "- **Evidence tables:** identify what each detector actually observes and whether that evidence generates, validates, filters, or scores a page hypothesis.",
        _build_link_footnote(
            best_known,
            results_repository=results_repository,
            results_ref=results_ref,
        ),
        r"- **Est. Serial Runtime\*\*:** Estimated single-detector serial runtime derived from recorded regression evidence; actual wall time varies with parallelism and scheduling.",
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


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _pipeline_context(run_dir: Path) -> dict[str, Any]:
    info = _read_json(run_dir / "RUN-INFO.json")
    parameters = _read_json(run_dir / "parameters.json")
    summary = normalize_summary_metrics(_read_json(run_dir / "reports" / "summary.json"))
    for payload in (info, parameters, summary):
        context = payload.get("detector_pipeline")
        if isinstance(context, dict):
            return context
    return {}


def _common_value(values: list[Any], default: Any = "unknown") -> Any:
    filtered = [value for value in values if value not in (None, "", "unknown")]
    if not filtered:
        return default
    first = filtered[0]
    return first if all(value == first for value in filtered) else "mixed"


def _lpt_makespan(durations: list[float], pipelines: int) -> float | None:
    if not durations or pipelines <= 0:
        return None
    loads = [0.0] * min(pipelines, len(durations))
    for duration in sorted((max(0.0, value) for value in durations), reverse=True):
        index = min(range(len(loads)), key=loads.__getitem__)
        loads[index] += duration
    return max(loads)


_EFFECT_SCOPE_FALLBACKS: dict[str, tuple[str, ...]] = {
    "critical": ("critical", "important_plus", "moderate_plus", "low_plus", "non_dormant", "exhaustive"),
    "non_dormant": ("non_dormant", "exhaustive"),
    "exhaustive": ("exhaustive",),
}


def _scope_parameter_count(payload: dict[str, Any] | None, scope: str) -> int | None:
    if not payload:
        return None
    domain_space = payload.get("domain_space")
    if not isinstance(domain_space, dict):
        return None
    for key in _EFFECT_SCOPE_FALLBACKS[scope]:
        entry = domain_space.get(key)
        if not isinstance(entry, dict):
            continue
        count = int(entry.get("parameter_set_count", 0) or 0)
        if count > 0:
            return count
    return None


def _load_runtime_index(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"observations": []}
    return _read_json(path)


def _combined_detector_ids(run_dirs: list[Path]) -> list[str]:
    values = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "manifest.json"
        if manifest_path.is_file():
            values.append(str(_read_json(manifest_path).get("detector", run_dir.parent.name)))
        else:
            values.append(run_dir.parent.name)
    return values


def _combined_golden_sha(run_dirs: list[Path]) -> str:
    values = set()
    for run_dir in run_dirs:
        info_path = run_dir / "RUN-INFO.json"
        if info_path.is_file():
            value = str(_read_json(info_path).get("golden_set_sha256") or "")
            if value:
                values.add(value)
    return next(iter(values)) if len(values) == 1 else ""


def _coherent_report_execution_profile(
    run_dirs: list[Path], runtime_index_path: Path | None,
) -> dict[str, Any] | None:
    if runtime_index_path is None:
        return None
    return coherent_execution_profile(
        _load_runtime_index(runtime_index_path),
        _combined_detector_ids(run_dirs),
        golden_set_sha256=_combined_golden_sha(run_dirs),
    )


def _estimate_scope_makespan(
    run_dirs: list[Path],
    scope: str,
    pipelines: int,
    *,
    runtime_index_path: Path | None = None,
    execution_profile: dict[str, Any] | None = None,
) -> float | None:
    """Estimate full-scope wall time using the same shard/LPT execution model.

    The measured detector elapsed time is scaled to the requested parameter
    domain at the current thread setting.  Full exhaustive work is then split
    into the same bounded ~30-minute shards used by the regression launcher and
    those shard durations are placed across the active detector pipelines with
    LPT.  Treating each detector as one indivisible task badly overstates the
    makespan once long detectors are sharded.
    """
    from math import ceil
    from hth.regression.sharding import SAFETY_FACTOR, TARGET_SHARD_SECONDS

    runtime_index = _load_runtime_index(runtime_index_path)
    shard_estimates: list[float] = []
    for run_dir in run_dirs:
        info = _read_json(run_dir / "RUN-INFO.json")
        summary = normalize_summary_metrics(_read_json(run_dir / "reports" / "summary.json"))
        evaluated = int(summary.get("parameter_set_count", 0) or 0)
        elapsed = float(info.get("elapsed_seconds", 0.0) or 0.0)
        target_count = _scope_parameter_count(_calibration_payload(run_dir), scope)

        if execution_profile and runtime_index_path is not None:
            manifest_path = run_dir / "manifest.json"
            detector = (
                str(_read_json(manifest_path).get("detector", run_dir.parent.name))
                if manifest_path.is_file() else run_dir.parent.name
            )
            observation, _ = select_runtime_observation(
                runtime_index,
                detector,
                mode=str(execution_profile.get("mode") or ""),
                search_strategy=str(execution_profile.get("strategy") or ""),
                threads=int(execution_profile["threads"]),
                max_dimension=int(execution_profile["max_dimension"]),
                golden_set_sha256=_combined_golden_sha(run_dirs),
                runner_label=str(execution_profile.get("runner_label") or ""),
            )
            if observation:
                observation_sets = int(observation.get("actual_parameter_sets") or 0)
                observation_elapsed = float(observation.get("wall_clock_seconds") or 0.0)
                if observation_sets > 0 and observation_elapsed > 0:
                    evaluated = observation_sets
                    elapsed = observation_elapsed

        if evaluated <= 0 or elapsed <= 0 or target_count is None:
            return None

        scaled_work = elapsed * target_count / evaluated
        # Mirror the launcher's automatic shard count at the measured thread
        # setting.  Shards remain bounded by one parameter set each and by the
        # framework's normal 96-shard planning ceiling.
        shard_count = max(1, min(96, target_count, ceil(
            scaled_work * SAFETY_FACTOR / TARGET_SHARD_SECONDS
        )))
        shard_estimates.extend([scaled_work / shard_count] * shard_count)
    return _lpt_makespan(shard_estimates, pipelines)


def _regression_execution_metadata(
    run_dirs: list[Path],
    *,
    runtime_index_path: Path | None = None,
) -> dict[str, Any]:
    contexts = [_pipeline_context(run_dir) for run_dir in run_dirs]
    infos = [_read_json(run_dir / "RUN-INFO.json") for run_dir in run_dirs]
    starts = [
        timestamp for timestamp in
        (_parse_utc_timestamp(info.get("started_at_utc")) for info in infos)
        if timestamp is not None
    ]
    finishes = [
        timestamp for timestamp in
        (_parse_utc_timestamp(info.get("finished_at_utc")) for info in infos)
        if timestamp is not None
    ]
    span_seconds = None
    if starts and finishes:
        span_seconds = max(0.0, (max(finishes) - min(starts)).total_seconds())
    profile = _coherent_report_execution_profile(run_dirs, runtime_index_path)
    if profile:
        return {
            "pipeline_count": profile["pipeline_count"],
            "loading_strategy": profile["loading_strategy"],
            "stagger_minutes": 0,
            "threads": profile["threads"],
            "span_seconds": span_seconds,
            "profile": profile,
            "source": f"runtime-index coherent build {profile['build_id']} ({profile['coverage']}/{len(run_dirs)} detectors)",
        }
    return {
        "pipeline_count": _common_value([context.get("pipeline_count") for context in contexts], 1),
        "loading_strategy": _common_value([context.get("loading_strategy") for context in contexts], "fifo"),
        "stagger_minutes": _common_value([context.get("stagger_minutes") for context in contexts], 0),
        "threads": _common_value([info.get("threads") for info in infos], "unknown"),
        "span_seconds": span_seconds,
        "profile": None,
        "source": "persisted calibration records",
    }


def _queue_rows(
    run_dirs: list[Path],
    *,
    runtime_index_path: Path | None = None,
    execution_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    runtime_index = _load_runtime_index(runtime_index_path)
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        manifest = _read_json(run_dir / "manifest.json")
        info = _read_json(run_dir / "RUN-INFO.json")
        context = _pipeline_context(run_dir)
        detector = str(manifest.get("detector", run_dir.parent.name))
        estimate = context.get("runtime_estimate_seconds")
        source = context.get("runtime_estimate_source") or "unknown"
        if execution_profile and runtime_index_path is not None:
            observation, source = select_runtime_observation(
                runtime_index,
                detector,
                mode=str(execution_profile.get("mode") or ""),
                search_strategy=str(execution_profile.get("strategy") or ""),
                threads=int(execution_profile["threads"]),
                max_dimension=int(execution_profile["max_dimension"]),
                golden_set_sha256=_combined_golden_sha(run_dirs),
                runner_label=str(execution_profile.get("runner_label") or ""),
            )
            estimate = observation.get("wall_clock_seconds") if observation else None
        rows.append({
            "detector": detector,
            "queue_position": context.get("queue_position"),
            "pipeline_number": context.get("pipeline_number"),
            "estimate_seconds": estimate,
            "estimate_source": source,
            "ranked_quality": context.get("ranked_quality"),
            "started": _parse_utc_timestamp(info.get("started_at_utc")),
        })

    if execution_profile:
        rows.sort(key=lambda row: (
            -(float(row["estimate_seconds"]) if row.get("estimate_seconds") is not None else -1.0),
            row["detector"],
        ))
        pipeline_available = [0.0] * int(execution_profile["pipeline_count"])
        for position, row in enumerate(rows, start=1):
            pipeline_index = min(range(len(pipeline_available)), key=lambda idx: pipeline_available[idx])
            row["queue_position"] = position
            row["pipeline_number"] = pipeline_index + 1
            if row.get("estimate_seconds") is not None:
                pipeline_available[pipeline_index] += float(row["estimate_seconds"])
        return rows

    rows.sort(key=lambda row: (
        int(row["queue_position"]) if str(row.get("queue_position", "")).isdigit() else 10**9,
        row["started"].timestamp() if row.get("started") is not None else float("inf"),
        row["detector"],
    ))
    return rows


def _next_claim_optimization(queue_rows: list[dict[str, Any]], execution: dict[str, Any]) -> dict[str, Any]:
    pipelines=int(execution.get("pipeline_count",1) or 1)
    loading=str(execution.get("loading_strategy","")).strip().lower()
    seed_count=min(pipelines,len(queue_rows)) if queue_rows else 0
    if pipelines>1 and loading=="lpt" and queue_rows:
        return {"strategy":"10s LPT claim batches","seed_count":seed_count,
                "target_seconds":10.0,"estimate_floor_seconds":0.1,
                "initial_lock":"parent pre-batched",
                "refill":"one serialized claim per >=10s LPT batch",
                "claim_wait":"amortized per batch",
                "basis":"current LPT runtime intelligence + persisted short-run occupation"}
    return {"strategy":"dynamic queue claims","seed_count":0,
            "target_seconds":None,"estimate_floor_seconds":0.1,
            "initial_lock":"serialized","refill":"serialized dynamic claims",
            "claim_wait":"minimal","basis":"current execution profile"}


def _initial_claim_batches(queue_rows: list[dict[str, Any]], pipeline_count: int,
                           *, target_seconds: float=10.0,
                           estimate_floor_seconds: float=0.1) -> list[dict[str, Any]]:
    batches=[]; cursor=0
    for pipeline in range(1,max(1,pipeline_count)+1):
        tasks=[]; total=0.0
        while cursor<len(queue_rows):
            row=queue_rows[cursor]; cursor+=1
            try: seconds=float(row.get("estimate_seconds"))
            except (TypeError,ValueError): seconds=estimate_floor_seconds
            seconds=max(estimate_floor_seconds,seconds)
            tasks.append(row); total+=seconds
            if total>=target_seconds: break
        if not tasks: break
        batches.append({"pipeline":pipeline,"tasks":tasks,"estimated_seconds":total})
    return batches


def _github_url(repository: str) -> str:
    return f"https://github.com/{repository}" if repository else ""


def _markdown_link(label: str, url: str) -> str:
    return f"[{label}]({url})" if url else label


def _engineering_continuous_improvement_lines(
    *,
    run_url: str = "",
    pipeline_repository: str = "",
    results_repository: str = "",
    results_commit: str = "",
) -> list[str]:
    return [
        "## Engineering Continuous Improvement",
        "",
        "Every completed regression contributes reusable quality and runtime evidence so future document analysis and regression execution can begin from measured history rather than rediscovering prior results.",
        "",
        "### Calibration Intelligence Persistence",
        "",
        "- `calibration-index.json` retains detector quality, winner, parameter influence, domain-space, page-sensitivity, and calibration-evidence metadata.",
        "- Compatible authoritative calibrations remain preferred over provisional smoke observations.",
        f"- Results commit: {_markdown_link(results_commit, _github_url(results_repository) + '/commit/' + results_commit) if results_repository and results_commit else '`unknown`'}.",
        f"- Workflow run: {_markdown_link('Open workflow run', run_url) if run_url else 'unknown'}.",
        f"- Pipeline repository: {_markdown_link(pipeline_repository, _github_url(pipeline_repository)) if pipeline_repository else 'unknown'}.",
        f"- Results repository: {_markdown_link(results_repository, _github_url(results_repository)) if results_repository else 'unknown'}.",
        f"- Calibration index: {_markdown_link('calibration-index.json', _github_url(results_repository) + '/blob/' + (results_commit or 'main') + '/calibration-index.json') if results_repository else '`calibration-index.json`'}.",
        f"- Runtime index: {_markdown_link('runtime-index.json', _github_url(results_repository) + '/blob/' + (results_commit or 'main') + '/runtime-index.json') if results_repository else '`runtime-index.json`'}.",
        "- Smoke records are provisional; complete exhaustive full regressions are authoritative.",
        "",
        "### Runtime Intelligence Persistence",
        "",
        "- `runtime-index.json` retains detector wall-clock time, workload size, threads, pipeline placement, loading strategy, runner characteristics, and scheduler estimates.",
        "- `parallelism-index.json` retains measured shard, pipeline, and thread execution shapes so equivalent workloads can be compared by wall-clock time and effective acceleration.",
        "- Runtime history supports LPT queueing, regression-duration estimates, and future evidence-based thread recommendations.",
        "",
        "### Engineering Notes",
        "",
        "- Runtime estimates are derived from historical detector measurements and improve as additional compatible regressions are collected.",
        "- Multi-detector execution defaults to Longest Processing Time (LPT) scheduling to reduce the all-detector makespan.",
        "- Detector and parameter-thread recommendations must remain grounded in measured runtime history rather than runner CPU count alone.",
        "- Calibration recommendations evolve from accumulated quality evidence; runtime recommendations evolve independently from accumulated execution evidence.",
        "- Estimates and recommendations are specific to the Golden Set, detector configuration, parameter grid, strategy, thread count, and runner characteristics represented by the stored observations.",
    ]


def _combined_result_row(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    info = _read_json(run_dir / "RUN-INFO.json")
    parameters = _read_json(run_dir / "parameters.json")
    summary = normalize_summary_metrics(_read_json(run_dir / "reports" / "summary.json"))
    winner = summary.get("winner") if isinstance(summary.get("winner"), dict) else None
    baseline = summary.get("baseline") if isinstance(summary.get("baseline"), dict) else None
    winner_stats = result_metric_view(winner.get("summary", {}) if winner else {})
    baseline_stats = result_metric_view(baseline.get("summary", {}) if baseline else {})
    page_ordinals = summary.get("page_ordinals", []) if isinstance(summary.get("page_ordinals"), list) else []
    source_document = _source_document_metadata(run_dir, info, parameters)
    golden_set_id, golden_set_sha256 = _golden_set_identity(run_dir, info, parameters, summary)
    page_rate = _page_rate(winner, len(page_ordinals))
    return {
        "golden_set_id": golden_set_id,
        "golden_set_sha256": golden_set_sha256,
        "detector": str(manifest.get("detector", run_dir.parent.name)),
        "detector_name": _detector_friendly_name(str(manifest.get("detector", run_dir.parent.name))),
        "detector_short_name": _detector_short_name(str(manifest.get("detector", run_dir.parent.name))),
        "status": str(manifest.get("status", "unknown")),
        "parameter_short_name": _parameter_short_name(winner),
        "parameter_set_id": _parameter_id(winner),
        "mean_iou": winner_stats.get("mean_iou"),
        "mean_iou_success": winner_stats.get("mean_iou_success", winner_stats.get("mean_iou")),
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
        "started_at_utc": info.get("started_at_utc"),
        "finished_at_utc": info.get("finished_at_utc"),
        "detector_pipeline": _pipeline_context(run_dir),
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


def build_combined_summary(
    run_dirs: list[Path],
    run_url: str = "",
    *,
    pipeline_repository: str = "",
    results_repository: str = "",
    results_commit: str = "",
    calibration_index: Path | None = None,
    runtime_index: Path | None = None,
) -> str:
    if not run_dirs:
        raise ValueError("At least one regression run directory is required")
    parameter_build_index = _build_parameter_build_index(calibration_index)
    if len(run_dirs) == 1:
        return build_summary(
            run_dirs[0],
            run_url,
            pipeline_repository=pipeline_repository,
            results_repository=results_repository,
            results_commit=results_commit,
            calibration_index=calibration_index,
            parameter_build_index=parameter_build_index,
        )

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
        "## Ranked Detector Smoke Test Results",
        "",
        "| Rank | Detector | Detector ID | Role | Golden Set ID | Status | Parameter Set ID | Parameter Short Name | Avg IoU | Min IoU | StdDev | Avg IoU Success | Failures | Parameter Sets | Eval Rate | Doc Time | Run Elapsed |",
        "|---:|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rank, row in enumerate(combined_rows, start=1):
        lines.append(
            f"| {rank} | {row['detector_name']} | `{row['detector']}` | {_detector_characterization(str(row['detector'])).get('role', 'Unknown')} | `{_display_golden_set_id(row.get('golden_set_id', 'unknown'))}` | {row['status']} | "
            f"`{row['parameter_set_id']}` | `{row['parameter_short_name']}` | {_number(row['mean_iou'])} | "
            f"{_number(row['minimum_iou'])} | {_number(row['stddev_iou'])} | "
            f"{_number(row.get('mean_iou_success', row['mean_iou']))} | {row['failures']} | {row['parameter_sets']} | "
            f"{_format_page_rate(row['page_rate'])} | {_duration(row['document_seconds'])} | "
            f"{_duration(row['elapsed_seconds'])} |"
        )
    lines.extend([
        "",
        "### Metric Definitions",
        "",
        "- **Avg IoU:** Arithmetic mean across every Golden Set page; failed/no-candidate pages contribute `0.0000` and remain in the denominator. This is the primary ranking metric.",
        "- **Avg IoU Success:** Arithmetic mean across successful Golden Set page evaluations only; failed/no-candidate pages are excluded.",
        "- **Min IoU:** Lowest single-page IoU produced by that winner across the Golden Set. It exposes the detector's weakest evaluated page; it is not the minimum Avg IoU across parameter sets.",
        "- **StdDev:** Population standard deviation of the winner's page IoUs. Lower values indicate more even page-to-page performance, but a uniformly poor detector can also have a low StdDev, so read it with Avg IoU and Min IoU.",
        "- **Failures:** Number of Golden Set pages the winning parameter set could not evaluate successfully.",
        "- **Ranking order:** Avg IoU descending, then Min IoU descending, failures ascending, StdDev ascending, and evaluation rate descending.",
        "- **Δ Baseline Avg IoU:** Winning Avg IoU minus the named baseline profile's Avg IoU for the same detector run.",
        "",
    ])
    lines.extend(_render_calibration_report(
        run_dirs,
        combined_rows,
        calibration_index,
        results_repository=results_repository,
        results_ref=results_commit or "main",
    ))

    completed_runs = sum(1 for row in combined_rows if str(row.get("status", "")).lower() == "complete")
    total_parameter_sets = sum(int(row.get("parameter_sets", 0) or 0) for row in combined_rows)
    total_page_evaluations = sum(
        int(row.get("parameter_sets", 0) or 0) * int(row.get("golden_set_pages", 0) or 0)
        for row in combined_rows
    )
    aggregate_elapsed = sum(float(row.get("elapsed_seconds", 0.0) or 0.0) for row in combined_rows)
    execution = _regression_execution_metadata(run_dirs, runtime_index_path=runtime_index)
    pipeline_count = int(execution.get("pipeline_count", 1) or 1) if str(execution.get("pipeline_count", 1)).isdigit() else 1
    regression_span = execution.get("span_seconds")
    concurrency = (
        aggregate_elapsed / float(regression_span)
        if regression_span is not None and float(regression_span) > 0
        else None
    )
    queue_rows = _queue_rows(run_dirs, runtime_index_path=runtime_index, execution_profile=execution.get('profile'))
    exhaustive_estimate = _estimate_scope_makespan(
        run_dirs, "exhaustive", pipeline_count,
        runtime_index_path=runtime_index, execution_profile=execution.get("profile"),
    )
    non_dormant_estimate = _estimate_scope_makespan(
        run_dirs, "non_dormant", pipeline_count,
        runtime_index_path=runtime_index, execution_profile=execution.get("profile"),
    )
    critical_estimate = _estimate_scope_makespan(
        run_dirs, "critical", pipeline_count,
        runtime_index_path=runtime_index, execution_profile=execution.get("profile"),
    )

    lines.extend([
        "",
        "<details open>",
        "<summary><h2>Detector Regression Reports</h2></summary>",
        "",
        "### Regression Completion Summary",
        "",
        "| Measure | Value | Notes |",
        "|---|---:|---|",
        f"| Detector runs completed | {completed_runs} of {len(combined_rows)} | Successful detector regressions completed out of those scheduled. |",
        f"| Parameter sets evaluated | {total_parameter_sets} | Total detector parameter configurations evaluated across all runs. |",
        f"| Golden Set page evaluations | {total_page_evaluations} | Parameter sets multiplied by evaluated Golden Set pages. |",
        f"| Aggregate detector runtime | {_duration(aggregate_elapsed)} | Sum of detector wall-clock runtimes; this is not the elapsed time experienced by the user. |",
        f"| Regression wall-clock span | {_duration(regression_span)} | Earliest detector start through latest detector finish. |",
        f"| Effective detector concurrency | {f'{concurrency:.2f}×' if concurrency is not None else 'unknown'} | Aggregate detector runtime divided by regression wall-clock span. |",
        f"| Detector pipelines | {execution.get('pipeline_count', 'unknown')} | Maximum concurrent detector regressions used by this build. |",
        f"| Loading strategy | {('LPT (Longest Processing Time first)' if str(execution.get('loading_strategy', '')).lower() == 'lpt' else str(execution.get('loading_strategy', 'unknown')).upper())} | Strategy used to order the shared detector queue. |",
        f"| Pipeline stagger | {execution.get('stagger_minutes', 'unknown')}m | Delay between initial pipeline starts; replacement loads begin immediately. |",
        f"| Source-document images | {source_document.get('image_count', 'unknown')} | Total images recorded for the source document. |",
        "",
        "### Regression Execution and Detector Queueing",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Detector pipelines | {execution.get('pipeline_count', 'unknown')} |",
        f"| Detector loading strategy | {('LPT (Longest Processing Time first)' if str(execution.get('loading_strategy', '')).lower() == 'lpt' else str(execution.get('loading_strategy', 'unknown')).upper())} |",
        f"| Threads per detector regression | {execution.get('threads', 'unknown')} |",
        f"| Execution recommendation basis | {execution.get('source', 'unknown')} |",
        f"| Pipeline start stagger | {execution.get('stagger_minutes', 'unknown')}m |",
        f"| Runtime intelligence | `runtime-index.json` |",
        f"| Parallelism intelligence | `parallelism-index.json` |",
        f"| Calibration intelligence | `calibration-index.json` |",
        "",
        "Detector pipelines pull continuously from one shared queue. Once a detector finishes, that pipeline immediately loads the next queued detector until the queue is empty.",
        "",
        "| Queue | Detector | Pipeline | Estimated Runtime | Scheduling Basis |",
        "|---:|---|---:|---:|---|",
    ])
    for queue_index, row in enumerate(queue_rows, start=1):
        position = row.get("queue_position")
        queue_number = int(position) if str(position or "").isdigit() else queue_index
        estimate_seconds = row.get("estimate_seconds")
        estimate = "no history" if estimate_seconds is None else _duration(estimate_seconds)
        source = str(row.get("estimate_source") or "no-history")
        lines.append(
            f"| {queue_number} | {_detector_friendly_name(str(row['detector']))} (`{row['detector']}`) | "
            f"{row.get('pipeline_number', 'unknown')} | {estimate} | {source} |"
        )

    claim_optimization = _next_claim_optimization(queue_rows, execution)
    lines.extend([
        "",
        "#### Execution Optimization — Next Run Claim Strategy",
        "",
        "| Setting | Preferred next run |",
        "|---|---|",
        f"| Claim strategy | {claim_optimization['strategy']} |",
        f"| Batch target | {claim_optimization['target_seconds'] or 'n/a'}s estimated work |",
        f"| Scheduling estimate floor | {claim_optimization['estimate_floor_seconds']}s |",
        f"| Initial-wave claims | {claim_optimization['initial_lock']} |",
        f"| Refill strategy | {claim_optimization['refill']} |",
        f"| Claim-wait objective | {claim_optimization['claim_wait']} |",
        f"| Optimization basis | {claim_optimization['basis']} |",
    ])
    if claim_optimization["target_seconds"] is not None:
        initial_batches=_initial_claim_batches(
            queue_rows,int(execution.get("pipeline_count",1) or 1),
            target_seconds=float(claim_optimization["target_seconds"]),
            estimate_floor_seconds=float(claim_optimization["estimate_floor_seconds"]),
        )
        lines.extend(["","| Pipeline | Initial LPT claim batch | Estimated Work | Threads |",
                      "|---:|---|---:|---:|"])
        for batch in initial_batches:
            names="<br>".join(
                f"{_detector_friendly_name(str(row['detector']))} (`{row['detector']}`)"
                for row in batch["tasks"]
            )
            lines.append(
                f"| {batch['pipeline']} | {names} | {_duration(batch['estimated_seconds'])} | "
                f"{execution.get('threads','unknown')} |"
            )
    lines.extend([
        "",
        "Each short-run claim atomically removes consecutive work from the LPT queue until the batch contains at least 10 seconds of estimated work, using a 0.1-second scheduling floor. The parent constructs the initial batches before workers start; refill batches use one serialized queue transaction each. The final claimant drains whatever work remains—there is no special tail-mode reversion.",
        "",
        "Queue order reflects the selected loading strategy. LPT (Longest Processing Time first) schedules the longest estimated detector work first, FIFO preserves configured detector order, and Ranked uses historical detector quality.",
        "",
        "### Regression Recommendations Summary",
        "",
        "#### Execution Configuration",
        "",
        "| Setting | Recommended | Basis |",
        "|---|---|---|",
        "| Detector pipelines | 4 | Current HTH default for multi-detector regressions. |",
        "| Detector loading | LPT (Longest Processing Time first) | Reduces the slow-detector tail by loading historically longest regressions first. |",
        f"| Threads per detector regression | {execution.get('threads', 'Auto')} | Preserve the current measured setting until runtime history supports a different thread recommendation. |",
        "| Startup stagger | 0m | Avoids idle startup time unless runner contention requires a stagger. |",
        "",
        "#### Estimated Runtime",
        "",
        "| All-Detector Regression Scope | Estimated Wall Time* |",
        "|---|---:|",
        f"| Exhaustive | {_duration(exhaustive_estimate)} |",
        f"| Non-dormant | {_duration(non_dormant_estimate)} |",
        f"| Critical only | {_duration(critical_estimate)} |",
        "",
        r"\* Estimates scale each detector's measured runtime to the selected effect-size domain, apply the normal bounded shard plan, and simulate shard-level LPT placement across the recommended detector pipelines. Effect-group fallback remains active when a detector has no parameter sets in the requested group.",
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
                calibration_index=calibration_index,
                parameter_build_index=parameter_build_index,
            ).rstrip()
        )
        lines.extend(["", "</details>"])
        if index != len(run_dirs) - 1:
            lines.extend([""])
    lines.extend([
        "", "</details>", "", "</details>", "",
        *_engineering_continuous_improvement_lines(
            run_url=run_url,
            pipeline_repository=pipeline_repository,
            results_repository=results_repository,
            results_commit=results_commit,
        ),
    ])
    lines.append("")
    return "\n".join(_add_report_navigation(lines))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--run-url", default=os.environ.get("HTH_RUN_URL", ""))
    p.add_argument("--pipeline-repository", default=os.environ.get("HTH_PIPELINE_REPOSITORY", ""))
    p.add_argument("--results-repository", default=os.environ.get("HTH_RESULTS_REPOSITORY", ""))
    p.add_argument("--results-commit", default=os.environ.get("HTH_RESULTS_COMMIT", ""))
    p.add_argument("--calibration-index", type=Path)
    p.add_argument("--runtime-index", type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    text = build_combined_summary(
        args.run_dir,
        args.run_url,
        pipeline_repository=args.pipeline_repository,
        results_repository=args.results_repository,
        results_commit=args.results_commit,
        calibration_index=args.calibration_index,
        runtime_index=args.runtime_index,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
