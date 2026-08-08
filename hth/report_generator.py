#!/usr/bin/env python3
"""Regenerate HTH human-facing reports from persisted results-repository data."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from hth.optimizer_store import build_optimizer_index, render_heatmap_svg, render_markdown
from hth.write_regression_summary import build_combined_summary

_STATUS_PRIORITY = {"authoritative": 3, "partial": 2, "provisional": 1}


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _golden_sha(path: Path | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibration_run_dirs(results_root: Path, golden_set: Path | None = None) -> list[Path]:
    """Resolve one best persisted calibration record per detector."""
    index_path = results_root / "calibration-index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}")
    index = _read_json(index_path)
    expected_sha = _golden_sha(golden_set)
    candidates: dict[str, dict[str, Any]] = {}
    for entry in index.get("entries", []):
        if not isinstance(entry, dict):
            continue
        detector = str(entry.get("detector_id") or "").strip()
        record_path = str(entry.get("record_path") or "").strip()
        if not detector or not record_path:
            continue
        if expected_sha and str(entry.get("golden_set_sha256") or "") != expected_sha:
            continue
        record_dir = results_root / record_path
        if not record_dir.is_dir():
            continue
        rank = (
            _STATUS_PRIORITY.get(str(entry.get("calibration_status") or ""), 0),
            str(entry.get("created_at_utc") or entry.get("published_at_utc") or ""),
        )
        current = candidates.get(detector)
        if current is None:
            candidates[detector] = entry
            continue
        current_rank = (
            _STATUS_PRIORITY.get(str(current.get("calibration_status") or ""), 0),
            str(current.get("created_at_utc") or current.get("published_at_utc") or ""),
        )
        if rank > current_rank:
            candidates[detector] = entry
    if not candidates:
        suffix = f" matching {golden_set}" if expected_sha else ""
        raise ValueError(f"No persisted calibration records found{suffix}")
    return [results_root / str(candidates[key]["record_path"]) for key in sorted(candidates)]


def generate_calibration_manifest(
    results_root: Path,
    output: Path,
    *,
    golden_set: Path | None,
    pipeline_repository: str,
    results_repository: str,
    results_commit: str,
    run_url: str,
) -> Path:
    persisted_dirs = calibration_run_dirs(results_root, golden_set)

    # calibration_store intentionally persists a compact, flattened record:
    # reports/summary.json becomes <record>/summary.json, etc.  The normal
    # summary renderer consumes the live regression-run layout, so reconstruct
    # only that tiny layout in a temporary directory for report generation.
    # This also keeps report-only runs read-only with respect to persisted data.
    with tempfile.TemporaryDirectory(prefix="hth-report-") as temp:
        temp_root = Path(temp)
        run_dirs: list[Path] = []
        for ordinal, persisted in enumerate(persisted_dirs, start=1):
            normalized = temp_root / f"run-{ordinal:03d}"
            reports = normalized / "reports"
            reports.mkdir(parents=True, exist_ok=True)
            for name in ("manifest.json", "parameters.json", "RUN-INFO.json"):
                source = persisted / name
                if source.is_file():
                    shutil.copy2(source, normalized / name)
            for name in ("summary.json", "winner-pages.json", "calibration-intelligence.json"):
                source = persisted / name
                if source.is_file():
                    shutil.copy2(source, reports / name)
            required = (normalized / "manifest.json", normalized / "parameters.json", normalized / "RUN-INFO.json", reports / "summary.json")
            missing = [path.name for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Persisted calibration record {persisted} is incomplete; missing: {', '.join(missing)}"
                )
            run_dirs.append(normalized)

        text = build_combined_summary(
            run_dirs,
            run_url,
            pipeline_repository=pipeline_repository,
            results_repository=results_repository,
            results_commit=results_commit,
            calibration_index=results_root / "calibration-index.json",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def _completed_optimizer_run_id(results_root: Path, detector: str) -> str | None:
    """Return the run id of the last optimizer execution that was fully published.

    The persisted human-facing optimizer summary is written only by the successful
    end-of-run publish step.  Treat it as the completion marker rather than
    inferring completion from shard/shape observations, which may belong to an
    interrupted execution.
    """
    summary_path = results_root / "execution-optimizer" / detector / "summary.md"
    if not summary_path.is_file():
        return None
    text = summary_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Optimizer run:\s*\*\*([^*]+)\*\*", text)
    return match.group(1).strip() if match else None


def _latest_completed_run_from_index(index: dict[str, Any], detector: str) -> str | None:
    """Find a run explicitly marked complete by finalized run metadata."""
    runs = index.get("runs") if isinstance(index.get("runs"), dict) else {}
    matches: list[tuple[str, str]] = []
    for run_id, payload in runs.items():
        if not isinstance(payload, dict) or str(payload.get("detector_id")) != detector:
            continue
        metadata = payload.get("run_metadata") if isinstance(payload.get("run_metadata"), dict) else {}
        # stop_reason is written only after the shape loop exits normally (range
        # complete or throughput plateau), immediately before final publication.
        if not str(metadata.get("stop_reason") or "").strip():
            continue
        matches.append((str(payload.get("updated_at_utc") or ""), str(run_id)))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def _completed_run_payload(index: dict[str, Any], detector: str, run_id: str) -> dict[str, Any]:
    runs = index.get("runs") if isinstance(index.get("runs"), dict) else {}
    payload = runs.get(str(run_id))
    if isinstance(payload, dict) and str(payload.get("detector_id")) == detector:
        return payload
    # Legacy completed optimizer runs may predate the per-run map.  The published
    # summary still proves completion; report generation can rebuild the table
    # from the run-tagged shape observations in parallelism-index.json.
    return {
        "optimizer_run_id": str(run_id),
        "detector_id": detector,
        "run_metadata": {},
    }


def generate_optimizer_report(results_root: Path, detector: str, output_dir: Path) -> dict[str, Path]:
    optimizer_path = results_root / "optimizer-index.json"
    parallelism_path = results_root / "parallelism-index.json"
    if not optimizer_path.is_file():
        raise FileNotFoundError(f"Missing {optimizer_path}")
    if not parallelism_path.is_file():
        raise FileNotFoundError(f"Missing {parallelism_path}")

    # Report ONLY a fully published optimizer execution.  Do not infer a run from
    # shard checkpoints or partially completed shape observations.
    optimizer = _read_json(optimizer_path)
    parallelism = _read_json(parallelism_path)
    persisted_report_dir = results_root / "execution-optimizer" / detector
    persisted_summary = persisted_report_dir / "summary.md"
    persisted_profile = persisted_report_dir / "heatmap.svg"
    run_id = _completed_optimizer_run_id(results_root, detector)
    if run_id is None:
        run_id = _latest_completed_run_from_index(optimizer, detector)
    if run_id is None:
        # Optimizer reports published before run IDs were embedded in the summary
        # are still completed artifacts: the optimizer workflow publishes this
        # directory only after the optimizer step succeeds.  Preserve that legacy
        # completed report verbatim rather than guessing a run from observations.
        if persisted_summary.is_file() and persisted_profile.is_file():
            output_dir.mkdir(parents=True, exist_ok=True)
            summary = output_dir / "summary.md"
            profile = output_dir / "heatmap.svg"
            shutil.copy2(persisted_summary, summary)
            shutil.copy2(persisted_profile, profile)
            return {"summary": summary, "profile": profile}
        raise ValueError(f"No completed persisted optimizer run found for detector {detector}")

    run_payload = _completed_run_payload(optimizer, detector, run_id)
    current = build_optimizer_index(parallelism, detector, run_id)
    if not current.get("observation_count"):
        raise ValueError(
            f"Completed optimizer run {run_id} has no persisted completed shape observations for {detector}"
        )
    run_metadata = run_payload.get("run_metadata") if isinstance(run_payload.get("run_metadata"), dict) else {}
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "summary.md"
    profile = output_dir / "heatmap.svg"
    summary.write_text(render_markdown(current, run_metadata), encoding="utf-8")
    profile.write_text(render_heatmap_svg(current), encoding="utf-8")
    return {"summary": summary, "profile": profile}

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="report", required=True)
    calibration = sub.add_parser("detector-calibration-manifest")
    calibration.add_argument("--results-root", type=Path, required=True)
    calibration.add_argument("--golden-set", type=Path)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--pipeline-repository", default="")
    calibration.add_argument("--results-repository", default="")
    calibration.add_argument("--results-commit", default="")
    calibration.add_argument("--run-url", default="")

    optimizer = sub.add_parser("execution-optimizer")
    optimizer.add_argument("--results-root", type=Path, required=True)
    optimizer.add_argument("--detector", required=True)
    optimizer.add_argument("--output-dir", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.report == "detector-calibration-manifest":
        path = generate_calibration_manifest(
            args.results_root,
            args.output,
            golden_set=args.golden_set,
            pipeline_repository=args.pipeline_repository,
            results_repository=args.results_repository,
            results_commit=args.results_commit,
            run_url=args.run_url,
        )
        print(path)
        return 0
    paths = generate_optimizer_report(args.results_root, args.detector, args.output_dir)
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
