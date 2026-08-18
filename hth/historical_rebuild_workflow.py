#!/usr/bin/env python3
"""Workflow-facing orchestration for historical canonical reranking.

GitHub Actions supplies inputs; this module owns selection, artifact streaming,
compatibility checks, cleanup, and summary accounting.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from hth.historical_rerank import HistoricalRerankSkip, rerank_run


def _run(command: list[str], *, cwd: Path | None = None, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=capture)


def resolve_runs(
    *, repository: str, pipeline_root: Path, scope: str, run_number: str,
    floor_sha: str, output: Path,
) -> dict[str, int]:
    result = _run([
        "gh", "api", "--paginate", f"/repos/{repository}/actions/runs?per_page=100",
        "--jq", '.workflow_runs[] | select(.path == ".github/workflows/regress-detector.yml") '
                '| select(.status == "completed" and .conclusion == "success") '
                '| [.id, .run_number, .head_sha] | @tsv',
    ])
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "Unable to enumerate historical workflow runs")

    eligible: list[tuple[str, str, str]] = []
    all_rows: list[tuple[str, str, str]] = []
    filtered = 0
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        run_id, number, head_sha = parts
        all_rows.append((run_id, number, head_sha))
        ancestry = _run(
            ["git", "merge-base", "--is-ancestor", floor_sha, head_sha],
            cwd=pipeline_root,
        )
        if ancestry.returncode == 0:
            eligible.append((run_id, number, head_sha))
        else:
            filtered += 1
            print(f"Skipping pre-calibration-intelligence regression build #{number} ({head_sha})")

    if scope == "single-build":
        selected = [row for row in eligible if row[1] == str(run_number)]
        if not selected:
            if any(row[1] == str(run_number) for row in all_rows):
                raise RuntimeError(
                    f"Regression build #{run_number} predates calibration intelligence "
                    f"({floor_sha}) and cannot be historically reranked."
                )
            raise RuntimeError(f"No successful regress-detector workflow run found for build #{run_number}")
    else:
        selected = eligible

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(row[0] for row in selected) + ("\n" if selected else ""), encoding="utf-8")
    print(f"Historical rerank evidence floor: {floor_sha}")
    print(f"Eligible workflow runs selected: {len(selected)}")
    print(f"Filtered workflow runs: {filtered}")
    return {"selected": len(selected), "filtered": filtered}


def _eligible_run_dirs(target: Path) -> list[Path]:
    dirs: list[Path] = []
    for raw in sorted(target.rglob("raw/results.csv")):
        run_dir = raw.parent.parent
        manifest = run_dir / "manifest.json"
        summary = run_dir / "reports" / "summary.json"
        if not manifest.is_file() or not summary.is_file():
            print(f"WARNING: Skipping incomplete historical artifact: {run_dir}")
            continue
        payload = json.loads(summary.read_text(encoding="utf-8"))
        strategy = payload.get("strategy") or payload.get("requested_strategy") or ""
        if strategy not in {"exhaustive", "exhaustive-with-zombies", "cartesian"}:
            print(f"Skipping non-exhaustive historical run: {run_dir} ({strategy})")
            continue
        dirs.append(run_dir)
    return dirs


def stream_rebuild(
    *, repository: str, run_ids_file: Path, results_root: Path, top: int,
    summary_path: Path,
) -> dict[str, int]:
    totals = {
        "downloaded": 0, "processed": 0, "reranked": 0,
        "skipped_incompatible": 0, "skipped_downloads": 0,
    }
    lines = ["## Historical regression rerank — streamed rebuild", ""]
    base = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())

    for run_id in [line.strip() for line in run_ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]:
        target = base / f"hth-historical-artifact-{run_id}"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        print(f"::group::Historical workflow run {run_id}")
        try:
            download = _run(["gh", "run", "download", run_id, "--repo", repository, "--dir", str(target)])
            if download.stdout:
                print(download.stdout, end="")
            if download.returncode:
                error = (download.stderr or "").strip()
                if "no space left on device" in error.lower():
                    raise RuntimeError(
                        f"Runner disk exhausted while downloading workflow run {run_id}; "
                        "aborting rather than misclassifying it as unavailable."
                    )
                print(
                    f"WARNING: Could not download artifacts for workflow run {run_id} "
                    f"from {repository}; artifact may be expired or unavailable. Skipping."
                )
                totals["skipped_downloads"] += 1
                continue

            totals["downloaded"] += 1
            run_rows: list[dict[str, Any]] = []
            skipped: list[str] = []
            for run_dir in _eligible_run_dirs(target):
                try:
                    row = rerank_run(run_dir, results_root, top=top)
                except HistoricalRerankSkip as exc:
                    print(f"WARNING: Skipping incompatible historical artifact: {run_dir}: {exc}")
                    totals["skipped_incompatible"] += 1
                    skipped.append(f"`{run_dir}` — {exc}")
                    continue
                totals["reranked"] += 1
                run_rows.append(row)
                print(json.dumps(row, sort_keys=True))

            totals["processed"] += 1
            if run_rows or skipped:
                lines += [f"### Workflow run `{run_id}`", ""]
                for row in run_rows:
                    changed = "changed" if row["winner_changed"] else "unchanged"
                    lines.append(
                        f"- `{row['detector']}` / `{row['run_id']}`: winner {changed}; "
                        f"`{row['winner']}` — Avg IoU `{float(row['avg_iou'] or 0):.4f}`, "
                        f"Avg IoU Success `{float(row['avg_iou_success'] or 0):.4f}`, "
                        f"failures `{row['failures']}`."
                    )
                for item in skipped:
                    lines.append(f"- skipped incompatible: {item}")
                lines.append("")
        finally:
            shutil.rmtree(target, ignore_errors=True)
            print(f"Disk after workflow run {run_id}:")
            subprocess.run(["df", "-h", "."], check=False)
            print("::endgroup::")

    lines += [
        "## Streamed rebuild totals", "",
        f"- Downloaded workflow artifacts: **{totals['downloaded']}**",
        f"- Processed downloaded workflows: **{totals['processed']}**",
        f"- Historical detector runs reranked: **{totals['reranked']}**",
        f"- Incompatible historical records skipped: **{totals['skipped_incompatible']}**",
        f"- Unavailable/expired downloads skipped: **{totals['skipped_downloads']}**",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with Path(step_summary).open("a", encoding="utf-8") as handle:
            handle.write(summary_path.read_text(encoding="utf-8"))
    if totals["downloaded"] == 0:
        raise RuntimeError("No retained regression artifacts were available for reranking.")
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    resolve = sub.add_parser("resolve")
    resolve.add_argument("--repository", required=True)
    resolve.add_argument("--pipeline-root", type=Path, required=True)
    resolve.add_argument("--scope", choices=("single-build", "all-available-artifacts"), required=True)
    resolve.add_argument("--run-number", default="")
    resolve.add_argument("--floor-sha", required=True)
    resolve.add_argument("--output", type=Path, required=True)

    stream = sub.add_parser("stream")
    stream.add_argument("--repository", required=True)
    stream.add_argument("--run-ids-file", type=Path, required=True)
    stream.add_argument("--results-root", type=Path, required=True)
    stream.add_argument("--top", type=int, default=20)
    stream.add_argument("--summary", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "resolve":
        resolve_runs(
            repository=args.repository, pipeline_root=args.pipeline_root,
            scope=args.scope, run_number=args.run_number,
            floor_sha=args.floor_sha, output=args.output,
        )
    else:
        stream_rebuild(
            repository=args.repository, run_ids_file=args.run_ids_file,
            results_root=args.results_root, top=args.top, summary_path=args.summary,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
