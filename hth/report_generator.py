#!/usr/bin/env python3
"""Regenerate HTH human-facing reports from persisted results-repository data."""
from __future__ import annotations

import argparse
import hashlib
import json
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
    run_dirs = calibration_run_dirs(results_root, golden_set)
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


def _latest_optimizer_run(index: dict[str, Any], detector: str) -> tuple[str, dict[str, Any]]:
    matches: list[tuple[str, dict[str, Any]]] = []
    runs = index.get("runs") if isinstance(index.get("runs"), dict) else {}
    for run_id, payload in runs.items():
        if not isinstance(payload, dict) or str(payload.get("detector_id")) != detector:
            continue
        matches.append((str(run_id), payload))
    if not matches:
        raise ValueError(f"No persisted optimizer run found for detector {detector}")
    matches.sort(key=lambda item: (str(item[1].get("updated_at_utc") or ""), item[0]), reverse=True)
    return matches[0]


def generate_optimizer_report(results_root: Path, detector: str, output_dir: Path) -> dict[str, Path]:
    optimizer_path = results_root / "optimizer-index.json"
    parallelism_path = results_root / "parallelism-index.json"
    if not optimizer_path.is_file():
        raise FileNotFoundError(f"Missing {optimizer_path}")
    if not parallelism_path.is_file():
        raise FileNotFoundError(f"Missing {parallelism_path}")
    optimizer = _read_json(optimizer_path)
    run_id, run_payload = _latest_optimizer_run(optimizer, detector)
    parallelism = _read_json(parallelism_path)
    current = build_optimizer_index(parallelism, detector, run_id)
    if not current.get("observation_count"):
        raise ValueError(f"Optimizer run {run_id} has no compatible observations for {detector}")
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
