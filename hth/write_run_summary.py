#!/usr/bin/env python3
"""Write a durable HTH GitHub Actions pipeline-health summary.

The script owns presentation while the workflow supplies paths and provenance.
It can derive processing counts from HTH's generated JSON, keeping workflow YAML
small and making the summary testable outside GitHub Actions.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _short(value: str, length: int = 12) -> str:
    value = value.strip()
    return value[:length] if value else "unknown"


def _display_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    if seconds < 10:
        return f"{seconds:.1f}s"
    rounded = int(round(seconds))
    minutes, secs = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _status_icon(status: str) -> str:
    normalized = status.lower().strip()
    return {
        "success": "✅",
        "partial": "⚠️",
        "failure": "❌",
        "cancelled": "⏹️",
        "skipped": "⏭️",
    }.get(normalized, "ℹ️")



def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_stage_timings(path: str) -> list[dict[str, Any]]:
    if not path:
        return []
    source = Path(path)
    if not source.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object at {source}:{line_number}")
        records.append(payload)
    return records


def _read_json(path: str) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    if not source.is_file():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {source}")
    return payload


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _as_int(value)
        if parsed is not None:
            return parsed
    return None


def _hydrate_from_generated_json(args: argparse.Namespace) -> None:
    preprocess = _read_json(args.summary_json)
    analysis = _read_json(args.analysis_summary_json)

    if not args.collection_id:
        args.collection_id = str(preprocess.get("collection_id", "")).strip()

    args.docx_count = _first_int(
        args.docx_count,
        preprocess.get("source_docx_count"),
    )
    args.page_count = _first_int(
        args.page_count,
        preprocess.get("image_count"),
        analysis.get("page_count"),
    )
    args.processed_count = _first_int(
        args.processed_count,
        analysis.get("page_count"),
        preprocess.get("image_count"),
    )

    quality_counts = analysis.get("quality_status_counts", {})
    analysis_errors = quality_counts.get("error") if isinstance(quality_counts, dict) else None
    args.error_count = _first_int(
        args.error_count,
        analysis_errors,
        preprocess.get("errors"),
        0 if preprocess or analysis else None,
    )


def _read_detector_performance(path: str) -> list[dict[str, Any]]:
    payload = _read_json(path)
    summary = payload.get("geometry_candidate_summary", {})
    if not isinstance(summary, dict):
        return []
    catalog = summary.get("detectors", [])
    performance = summary.get("detector_performance", {})
    statuses = summary.get("method_status_counts", {})
    if not isinstance(catalog, list) or not isinstance(performance, dict):
        return []

    rows: list[dict[str, Any]] = []
    for detector in catalog:
        if not isinstance(detector, dict):
            continue
        method = str(detector.get("method", "")).strip()
        if not method:
            continue
        perf = performance.get(method, {})
        counts = statuses.get(method, {})
        if not isinstance(perf, dict):
            perf = {}
        if not isinstance(counts, dict):
            counts = {}
        rows.append({
            "display_name": detector.get("display_name") or detector.get("name") or method,
            "version": detector.get("version", ""),
            "runs": perf.get("runs"),
            "average_ms": perf.get("elapsed_ms_average"),
            "total_ms": perf.get("elapsed_ms_total"),
            "average_confidence": perf.get("confidence_average"),
            "ok": counts.get("ok", 0),
            "no_candidate": counts.get("no_candidate", 0),
            "error": counts.get("error", 0),
        })
    return rows


def _existing_outputs(paths: Iterable[str]) -> list[str]:
    result: list[str] = []
    for raw in paths:
        raw = raw.strip()
        if not raw:
            continue
        path = Path(raw)
        suffix = "/" if path.is_dir() else ""
        state = "present" if path.exists() else "not created"
        result.append(f"- `{raw}{suffix}` — {state}")
    return result


def build_summary(args: argparse.Namespace) -> str:
    icon = _status_icon(args.status)
    lines = [
        f"# {icon} HTH {args.pipeline_name}",
        "",
        "## Run",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Status | **{args.status}** |",
        f"| Collection | `{args.collection_id or 'unknown'}` |",
        f"| Source repository | `{args.source_repository or 'unknown'}` |",
        f"| Source commit | `{_short(args.source_commit)}` |",
        f"| Pipeline commit | `{_short(args.pipeline_commit)}` |",
        f"| Workflow | `{args.workflow_name or 'unknown'}` |",
        f"| Run | `{args.run_number or 'unknown'}` |",
        f"| Pipeline started | `{args.pipeline_started_at or 'unknown'}` |",
        f"| Summary generated | `{args.summary_generated_at}` |",
        f"| Duration | `{_display_duration(args.elapsed_seconds)}` |",
        "",
        "## Processing",
        "",
        "| Metric | Count |",
        "|---|---:|",
        f"| DOCX masters | {args.docx_count if args.docx_count is not None else 'unknown'} |",
        f"| Pages discovered | {args.page_count if args.page_count is not None else 'unknown'} |",
        f"| Pages processed | {args.processed_count if args.processed_count is not None else 'unknown'} |",
        f"| Page errors | {args.error_count if args.error_count is not None else 'unknown'} |",
    ]

    stage_timings = _read_stage_timings(args.stage_timings_jsonl)
    if stage_timings:
        lines.extend([
            "",
            "## Stage performance",
            "",
            "| Stage | Status | Started UTC | Completed UTC | Elapsed |",
            "|---|---|---|---|---:|",
        ])
        for record in stage_timings:
            elapsed = _display_duration(_as_float(record.get("elapsed_seconds")))
            lines.append(
                f"| `{record.get('stage', 'unknown')}` "
                f"| {record.get('status', 'unknown')} "
                f"| `{record.get('started_at_utc', 'unknown')}` "
                f"| `{record.get('completed_at_utc', 'unknown')}` "
                f"| {elapsed} |"
            )

    detector_rows = _read_detector_performance(args.page_analysis_json)
    if detector_rows:
        lines.extend([
            "",
            "## Detector performance",
            "",
            "| Detector | Runs | Candidate | No candidate | Errors | Avg elapsed | Avg confidence |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in detector_rows:
            average_ms = _as_float(row.get("average_ms"))
            elapsed = f"{average_ms:.1f} ms" if average_ms is not None else "unknown"
            confidence = _as_float(row.get("average_confidence"))
            confidence_text = f"{confidence:.3f}" if confidence is not None else "—"
            version = str(row.get("version", "")).strip()
            detector = str(row.get("display_name", "unknown"))
            if version:
                detector += f" `v{version}`"
            lines.append(
                f"| {detector} | {row.get('runs', 'unknown')} "
                f"| {row.get('ok', 0)} | {row.get('no_candidate', 0)} "
                f"| {row.get('error', 0)} | {elapsed} | {confidence_text} |"
            )

    if args.notes:
        lines.extend(["", "## Notes", "", args.notes.strip()])

    output_lines = _existing_outputs(args.output)
    if output_lines:
        lines.extend(["", "## Publication outputs", "", *output_lines])

    if args.run_url:
        lines.extend(["", f"[Open workflow run]({args.run_url})"])

    lines.append("")
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline-name", default="Preprocess Pipeline")
    p.add_argument("--status", default="success")
    p.add_argument("--collection-id", default=_env("HTH_COLLECTION_ID"))
    p.add_argument("--source-repository", default=_env("HTH_SOURCE_REPOSITORY"))
    p.add_argument("--source-commit", default=_env("HTH_SOURCE_COMMIT"))
    p.add_argument("--pipeline-commit", default=_env("GITHUB_SHA"))
    p.add_argument("--workflow-name", default=_env("GITHUB_WORKFLOW"))
    p.add_argument("--run-number", default=_env("GITHUB_RUN_NUMBER"))
    p.add_argument("--run-url", default="")
    p.add_argument("--elapsed-seconds", type=float)
    p.add_argument("--pipeline-started-at", default="")
    p.add_argument("--summary-generated-at", default="")
    p.add_argument("--stage-timings-jsonl", default="")
    p.add_argument("--docx-count", type=int)
    p.add_argument("--page-count", type=int)
    p.add_argument("--processed-count", type=int)
    p.add_argument("--error-count", type=int)
    p.add_argument("--summary-json", default="")
    p.add_argument("--analysis-summary-json", default="")
    p.add_argument(
        "--page-analysis-json",
        "--geometry-json",
        dest="page_analysis_json",
        default="",
        help=(
            "Path to page-analysis.json. --geometry-json remains as a "
            "compatibility alias."
        ),
    )
    p.add_argument("--notes", default="")
    p.add_argument("--output", action="append", default=[])
    p.add_argument("--destination", default=_env("GITHUB_STEP_SUMMARY"))
    return p


def main() -> int:
    args = parser().parse_args()
    if not args.summary_generated_at:
        args.summary_generated_at = _utc_now()
    _hydrate_from_generated_json(args)

    if not args.run_url and _env("GITHUB_SERVER_URL") and _env("GITHUB_REPOSITORY") and _env("GITHUB_RUN_ID"):
        args.run_url = (
            f"{_env('GITHUB_SERVER_URL')}/{_env('GITHUB_REPOSITORY')}"
            f"/actions/runs/{_env('GITHUB_RUN_ID')}"
        )

    summary = build_summary(args)
    if args.destination:
        destination = Path(args.destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(summary)
    else:
        print(summary, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
