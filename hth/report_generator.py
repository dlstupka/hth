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

from hth.results_layout import canonical_index_path, readable_index_path
from typing import Any

from hth.optimizer_store import build_optimizer_index, render_all_markdown, render_heatmap_svg, render_markdown, select_preferred_shape
from hth.write_regression_summary import build_combined_summary
from hth.domain.calibration import authoritative_record
from hth.calibration_store import load_index_with_persisted_backfill



def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _golden_sha(path: Path | None) -> str | None:
    if path is None:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matching_index_entries(results_root: Path, golden_set: Path | None = None) -> list[dict[str, Any]]:
    index_path = readable_index_path(results_root, "calibration-index.json")
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}")
    index = load_index_with_persisted_backfill(index_path)
    expected_sha = _golden_sha(golden_set)
    entries: list[dict[str, Any]] = []
    for entry in index.get("entries", []):
        if not isinstance(entry, dict):
            continue
        detector = str(entry.get("detector_id") or "").strip()
        record_path = str(entry.get("record_path") or "").strip()
        if not detector or not record_path:
            continue
        if expected_sha and str(entry.get("golden_set_sha256") or "") != expected_sha:
            continue
        if not (results_root / record_path).is_dir():
            continue
        entries.append(entry)
    return entries


def calibration_run_dirs(results_root: Path, golden_set: Path | None = None) -> list[Path]:
    """Resolve the best persisted calibration record per detector.

    This is the authoritative/best-known view.  Smoke observations are a
    separate provenance stream and must never replace a full calibration here.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in _matching_index_entries(results_root, golden_set):
        grouped.setdefault(str(entry["detector_id"]), []).append(entry)
    candidates: dict[str, dict[str, Any]] = {}
    for detector, records in grouped.items():
        selected = authoritative_record(records)
        if selected is not None:
            candidates[detector] = selected
    if not candidates:
        suffix = f" matching {golden_set}" if golden_set else ""
        raise ValueError(f"No persisted calibration records found{suffix}")
    return [results_root / str(candidates[key]["record_path"]) for key in sorted(candidates)]


def smoke_run_dirs(results_root: Path, golden_set: Path | None = None) -> list[Path]:
    """Resolve the latest persisted smoke observation per detector.

    Ranked Detector Smoke Test Results is an observation table, not a
    best-calibration table.  Only provisional records (the persistence status
    assigned to smoke runs) are eligible, regardless of stronger full evidence.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in _matching_index_entries(results_root, golden_set):
        if str(entry.get("calibration_status") or "").lower() != "provisional":
            continue
        grouped.setdefault(str(entry["detector_id"]), []).append(entry)
    selected: dict[str, dict[str, Any]] = {}
    for detector, records in grouped.items():
        selected[detector] = max(
            records,
            key=lambda item: (
                str(item.get("created_at_utc") or item.get("published_at_utc") or ""),
                str((item.get("build") or {}).get("github_run_number") if isinstance(item.get("build"), dict) else ""),
            ),
        )
    if not selected:
        suffix = f" matching {golden_set}" if golden_set else ""
        raise ValueError(f"No persisted smoke records found{suffix}")
    return [results_root / str(selected[key]["record_path"]) for key in sorted(selected)]


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
    persisted_dirs = smoke_run_dirs(results_root, golden_set)

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
            calibration_index=readable_index_path(results_root, "calibration-index.json"),
            runtime_index=readable_index_path(results_root, "runtime-index.json"),
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


def _latest_legacy_published_run_from_parallelism(parallelism: dict[str, Any], detector: str) -> str | None:
    """Recover the run id behind a legacy published optimizer report.

    Older optimizer summaries did not embed their optimizer run id.  A persisted
    summary/profile pair still proves that an optimizer execution completed and
    was published.  Under the legacy publication contract, aggregate optimizer
    observations reached the results repository only at successful end-of-run
    publication, so the newest run-tagged aggregate observation for this detector
    identifies that published execution.  Shard checkpoints are intentionally
    ignored: they are not completion evidence.
    """
    candidates: dict[str, str] = {}
    for row in parallelism.get("observations", []):
        if not isinstance(row, dict):
            continue
        if str(row.get("detector_id")) != detector or row.get("source") != "execution-optimizer":
            continue
        run_id = str(row.get("optimizer_run_id") or "").strip()
        if not run_id:
            continue
        stamp = str(row.get("captured_at_utc") or row.get("observed_at_utc") or row.get("created_at_utc") or "")
        candidates[run_id] = max(candidates.get(run_id, ""), stamp)
    if not candidates:
        return None
    return max(candidates.items(), key=lambda item: (item[1], item[0]))[0]




def _completed_optimizer_run_ids(index: dict[str, Any], detector: str) -> set[str]:
    """Return only optimizer runs explicitly finalized for this detector."""
    runs = index.get("runs") if isinstance(index.get("runs"), dict) else {}
    completed: set[str] = set()
    for run_id, payload in runs.items():
        if not isinstance(payload, dict) or str(payload.get("detector_id")) != detector:
            continue
        metadata = payload.get("run_metadata") if isinstance(payload.get("run_metadata"), dict) else {}
        if str(metadata.get("stop_reason") or "").strip():
            completed.add(str(run_id))
    return completed

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



def _legacy_completed_index_from_summary(path: Path, detector: str) -> dict[str, Any] | None:
    """Recover one completed execution profile from a pre-run-id published summary.

    Legacy optimizer reports are completion artifacts but may predate run tagging.
    Their table can contain historical compatibility rows. Recover only the
    concrete runner profile with the most measured shapes; never import rows
    whose runner identity is unknown. The legacy table schema changed over time,
    so resolve fields by column heading rather than fixed position.
    """
    if not path.is_file():
        return None

    def key(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def seconds(text: str) -> float:
        total = 0.0
        for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", text):
            total += float(value) * {"h": 3600.0, "m": 60.0, "s": 1.0}[unit]
        return total

    header: dict[str, int] | None = None
    groups: dict[str, list[dict[str, Any]]] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.startswith("|") or raw.startswith("|---"):
            continue
        cells = [cell.strip() for cell in raw.strip().strip("|").split("|")]
        normalized = [key(cell.replace("**", "")) for cell in cells]
        if "runner" in normalized and "pipelines" in normalized:
            header = {name: idx for idx, name in enumerate(normalized)}
            continue
        if header is None:
            continue

        def field(*names: str) -> str:
            for name in names:
                idx = header.get(key(name))
                if idx is not None and idx < len(cells):
                    return cells[idx].replace("**", "").strip()
            return ""

        runner = field("runner")
        if not runner or runner.lower().startswith("unknown"):
            continue
        try:
            pipelines = int(field("pipelines"))
            shards = int(field("shards"))
            threads = int(field("threads / pipeline", "threads per pipeline"))
            allocated = int(field("allocated threads", "allocated"))
            rate = float(field("sets/s", "parameter sets / second"))
        except (TypeError, ValueError):
            continue
        wall = seconds(field("fastest wall", "wall"))
        if wall <= 0.0 or rate <= 0.0:
            continue
        speedup = None
        try:
            speedup = float(field("speedup vs 1 pipeline", "speedup").rstrip("×x"))
        except ValueError:
            pass
        groups.setdefault(runner, []).append({
            "pipelines": pipelines, "shards": shards,
            "threads_per_pipeline": threads, "allocated_threads": allocated,
            "fastest_wall_clock_seconds": wall,
            "parameter_sets_per_second": rate,
            "observed_speedup_vs_one_pipeline": speedup,
            "execution_shape": f"{pipelines}p/{shards}s/{threads}t",
            "optimizer_shape_sequence": pipelines,
        })
    if not groups:
        return None
    runner_title, shapes = max(groups.items(), key=lambda item: (len(item[1]), max((x["pipelines"] for x in item[1]), default=0)))
    shapes.sort(key=lambda shape: shape["pipelines"])
    best = select_preferred_shape(shapes)
    return {
        "schema_version": 1, "detector_id": detector,
        "optimizer_run_id": "legacy-published", "runner_count": 1,
        "observation_count": len(shapes), "best_across_runners": best,
        "runners": [{"runner_title": runner_title, "shapes": shapes, "best_shape": best}],
    }

def _attach_optimizer_run_metadata(index: dict[str, Any], optimizer: dict[str, Any], detector: str) -> dict[str, Any]:
    runs = optimizer.get("runs") if isinstance(optimizer.get("runs"), dict) else {}
    metadata_by_id: dict[str, dict[str, Any]] = {}
    for run_id, payload in runs.items():
        if not isinstance(payload, dict) or str(payload.get("detector_id") or "") != detector:
            continue
        metadata = payload.get("run_metadata")
        if isinstance(metadata, dict):
            metadata_by_id[str(run_id)] = metadata
    index["run_metadata_by_id"] = metadata_by_id
    return index



def _attach_prediction_history(results_root: Path, index: dict[str, Any], detector: str) -> dict[str, Any]:
    path = results_root / "optimizer-predictions.json"
    if not path.is_file():
        return index
    payload = _read_json(path)
    index["prediction_history"] = [
        row for row in payload.get("predictions", [])
        if isinstance(row, dict) and str(row.get("detector_id") or "") == detector
    ]
    return index


def _optimizer_report_components(results_root: Path, detector: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    optimizer_path = readable_index_path(results_root, "optimizer-index.json")
    parallelism_path = readable_index_path(results_root, "parallelism-index.json")
    if not parallelism_path.is_file():
        raise FileNotFoundError(f"Missing {parallelism_path}")

    # optimizer-index.json is derived planning state. During the indexes/
    # migration it may be absent even though durable completed optimizer
    # summaries and parallelism observations still exist. Treat a missing
    # optimizer index as empty derived state and recover only from persisted
    # completion evidence; never infer completion from checkpoints.
    optimizer = _read_json(optimizer_path) if optimizer_path.is_file() else {}
    parallelism = _read_json(parallelism_path)
    persisted_report_dir = results_root / "execution-optimizer" / detector
    persisted_summary = persisted_report_dir / "summary.md"
    persisted_profile = persisted_report_dir / "heatmap.svg"
    run_id = _completed_optimizer_run_id(results_root, detector)
    if run_id is None:
        run_id = _latest_completed_run_from_index(optimizer, detector)
    if run_id is None and persisted_summary.is_file() and persisted_profile.is_file():
        run_id = _latest_legacy_published_run_from_parallelism(parallelism, detector)
    legacy_current = None
    if run_id is None and persisted_summary.is_file() and persisted_profile.is_file():
        legacy_current = _legacy_completed_index_from_summary(persisted_summary, detector)
    if run_id is None and legacy_current is None:
        raise ValueError(f"No completed persisted optimizer run found for detector {detector}")

    if legacy_current is not None:
        return legacy_current, legacy_current, {}

    run_payload = _completed_run_payload(optimizer, detector, run_id)
    current = _attach_optimizer_run_metadata(build_optimizer_index(parallelism, detector, run_id), optimizer, detector)
    if not current.get("observation_count"):
        raise ValueError(
            f"Completed optimizer run {run_id} has no persisted completed shape observations for {detector}"
        )
    completed_ids = _completed_optimizer_run_ids(optimizer, detector)
    completed_ids.add(str(run_id))
    preferred = _attach_optimizer_run_metadata(build_optimizer_index(parallelism, detector, optimizer_run_ids=completed_ids), optimizer, detector)
    if not preferred.get("observation_count"):
        preferred = current
    current = _attach_prediction_history(results_root, current, detector)
    preferred = _attach_prediction_history(results_root, preferred, detector)
    run_metadata = run_payload.get("run_metadata") if isinstance(run_payload.get("run_metadata"), dict) else {}
    return current, preferred, run_metadata


def _completed_optimizer_detectors(results_root: Path) -> list[str]:
    optimizer_path = readable_index_path(results_root, "optimizer-index.json")
    # The optimizer index is reconstructable derived state. Persisted
    # execution-optimizer/<detector>/ reports remain valid completion evidence
    # when that index is missing (notably across results-layout migrations).
    optimizer = _read_json(optimizer_path) if optimizer_path.is_file() else {}
    candidates: set[str] = set()
    detectors = optimizer.get("detectors") if isinstance(optimizer.get("detectors"), dict) else {}
    candidates.update(str(detector) for detector in detectors if str(detector).strip())
    runs = optimizer.get("runs") if isinstance(optimizer.get("runs"), dict) else {}
    for payload in runs.values():
        if isinstance(payload, dict) and str(payload.get("detector_id") or "").strip():
            candidates.add(str(payload["detector_id"]))
    persisted = results_root / "execution-optimizer"
    if persisted.is_dir():
        candidates.update(path.name for path in persisted.iterdir() if path.is_dir() and path.name != "all")

    completed: list[str] = []
    for detector in sorted(candidates):
        try:
            _optimizer_report_components(results_root, detector)
        except (FileNotFoundError, ValueError):
            continue
        completed.append(detector)
    if not completed:
        raise ValueError("No completed persisted optimizer runs found")
    return completed


def generate_optimizer_report_all(
    results_root: Path,
    output_dir: Path,
    *,
    note: str | None = None,
) -> dict[str, Path]:
    detectors = _completed_optimizer_detectors(results_root)
    preferred_indices: list[dict[str, Any]] = []
    profiles_dir = output_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    for detector in detectors:
        _, preferred, _ = _optimizer_report_components(results_root, detector)
        preferred_indices.append(preferred)
        (profiles_dir / f"{detector}.svg").write_text(render_heatmap_svg(preferred), encoding="utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "summary.md"
    text = render_all_markdown(preferred_indices)
    if note:
        marker = "### Execution optimizer summary\n\n"
        if text.startswith(marker):
            text = marker + f"> **Note:** {note}\n\n" + text[len(marker):]
        else:
            text = f"> **Note:** {note}\n\n{text}"
    summary.write_text(text, encoding="utf-8")
    return {"summary": summary, "profiles": profiles_dir}


def generate_optimizer_report(results_root: Path, detector: str, output_dir: Path) -> dict[str, Path]:
    if detector == "all":
        return generate_optimizer_report_all(results_root, output_dir)

    try:
        current, preferred, run_metadata = _optimizer_report_components(results_root, detector)
    except ValueError as exc:
        if "No completed persisted optimizer run found for detector" not in str(exc):
            raise
        return generate_optimizer_report_all(
            results_root,
            output_dir,
            note="This is currently all available optimization data.",
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "summary.md"
    profile = output_dir / "heatmap.svg"
    summary.write_text(render_markdown(current, run_metadata, preferred_index=preferred), encoding="utf-8")
    # The single-run report's shape table is run-local, so its profile must be
    # rendered from the same current-run index.  Using the coalesced preferred
    # index here can leave the plot showing an older compatible optimization
    # landscape while the table below shows the newly completed exhaustive run.
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
