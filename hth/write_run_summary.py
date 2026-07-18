#!/usr/bin/env python3
"""Write a durable HTH GitHub Actions job summary.

The script deliberately owns presentation while the workflow only supplies facts.
This keeps YAML small and makes the summary testable outside GitHub Actions.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _short(value: str, length: int = 12) -> str:
    value = value.strip()
    return value[:length] if value else "unknown"


def _display_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
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

    if args.notes:
        lines.extend(["", "## Notes", "", args.notes.strip()])

    output_lines = _existing_outputs(args.output)
    if output_lines:
        lines.extend(["", "## Outputs", "", *output_lines])

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
    p.add_argument("--docx-count", type=int)
    p.add_argument("--page-count", type=int)
    p.add_argument("--processed-count", type=int)
    p.add_argument("--error-count", type=int)
    p.add_argument("--notes", default="")
    p.add_argument("--output", action="append", default=[])
    p.add_argument("--destination", default=_env("GITHUB_STEP_SUMMARY"))
    return p


def main() -> int:
    args = parser().parse_args()
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
