#!/usr/bin/env python3
"""Generate execution-optimizer reports from persisted results."""
from __future__ import annotations

import re
from pathlib import Path

from hth.optimizer_history import completed_optimizer_run_ids, completed_run_records
from typing import Any

from hth.optimizer_store import build_optimizer_index, render_all_markdown, render_heatmap_svg, render_markdown, select_preferred_shape
from hth.optimizer_intelligence import (
    legacy_published_optimizer_index, historical_published_optimizer_indices,
    legacy_optimizer_rows_from_indices,
)
from hth.optimizer_validity import migrate_optimizer_run, optimizer_evidence_is_valid
from hth.shape_prediction import canonical_prediction_history
from hth.persistence import ResultsRepository, read_json as _read_json



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
    if not match:
        return None
    run_id = match.group(1).strip()
    manifest_path = results_root / "execution-optimizer" / detector / "runs" / run_id / "run.json"
    if manifest_path.is_file():
        manifest = migrate_optimizer_run(_read_json(manifest_path))
        if not optimizer_evidence_is_valid(manifest):
            return None
    return run_id


def _latest_completed_run_from_index(index: dict[str, Any], detector: str) -> str | None:
    """Find a run explicitly marked complete by finalized run metadata."""
    runs = index.get("runs") if isinstance(index.get("runs"), dict) else {}
    matches: list[tuple[str, str]] = []
    for run_id, payload in runs.items():
        if not isinstance(payload, dict) or str(payload.get("detector_id")) != detector:
            continue
        if not optimizer_evidence_is_valid(migrate_optimizer_run(payload)):
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
        if not optimizer_evidence_is_valid(row):
            continue
        run_id = str(row.get("optimizer_run_id") or "").strip()
        if not run_id:
            continue
        stamp = str(row.get("captured_at_utc") or row.get("observed_at_utc") or row.get("created_at_utc") or "")
        candidates[run_id] = max(candidates.get(run_id, ""), stamp)
    if not candidates:
        return None
    return max(candidates.items(), key=lambda item: (item[1], item[0]))[0]




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



def _attach_prediction_history(
    results_root: Path, index: dict[str, Any], detector: str, optimizer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repository = ResultsRepository(results_root)
    payload = repository.load_index("optimizer-predictions.json")
    index["prediction_history"] = canonical_prediction_history(
        detector=detector, prediction_payload=payload, optimizer_index=optimizer,
    )
    return index


def _canonicalize_published_optimizer_indices(indices: list[dict[str, Any]], detector: str) -> dict[str, Any] | None:
    """Route legacy published profiles through the same row/index builder as modern evidence."""
    rows = legacy_optimizer_rows_from_indices(indices, detector)
    if not rows:
        return None
    return build_optimizer_index({"observations": rows}, detector)


def _optimizer_report_components(results_root: Path, detector: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    repository = ResultsRepository(results_root)
    if not repository.has_index("parallelism-index.json"):
        raise FileNotFoundError(f"Missing {repository.readable_index_path('parallelism-index.json')}")

    # optimizer-index.json is derived planning state. During the indexes/
    # migration it may be absent even though durable completed optimizer
    # summaries and parallelism observations still exist. Treat a missing
    # optimizer index as empty derived state and recover only from persisted
    # completion evidence; never infer completion from checkpoints.
    optimizer = repository.load_index("optimizer-index.json")
    parallelism = repository.load_index("parallelism-index.json")

    # Completed per-run records are the durable optimizer source of truth. Merge
    # them into the rebuildable aggregate views so report regeneration retains
    # every published run even if optimizer-index.json was replaced/rebuilt.
    durable_runs = completed_run_records(results_root, detector)
    if durable_runs:
        observations = list(parallelism.get("observations", [])) if isinstance(parallelism.get("observations"), list) else []
        by_id = {str(row.get("observation_id")): row for row in observations if isinstance(row, dict) and row.get("observation_id")}
        optimizer_runs = dict(optimizer.get("runs", {})) if isinstance(optimizer.get("runs"), dict) else {}
        for record in durable_runs:
            manifest = record["manifest"]
            durable_run_id = str(manifest.get("optimizer_run_id"))
            for row in record["observations"]:
                key = str(row.get("observation_id") or f"durable:{durable_run_id}:{len(by_id)}")
                by_id[key] = row
            optimizer_runs[durable_run_id] = {
                "optimizer_run_id": durable_run_id,
                "detector_id": detector,
                "run_metadata": manifest.get("run_metadata") if isinstance(manifest.get("run_metadata"), dict) else {},
                "updated_at_utc": (manifest.get("run_metadata") or {}).get("completed_at_utc", "") if isinstance(manifest.get("run_metadata"), dict) else "",
            }
        parallelism["observations"] = list(by_id.values())
        optimizer["runs"] = optimizer_runs
    persisted_report_dir = results_root / "execution-optimizer" / detector
    persisted_summary = persisted_report_dir / "summary.md"

    run_id = _completed_optimizer_run_id(results_root, detector)
    if run_id is None:
        run_id = _latest_completed_run_from_index(optimizer, detector)
    if run_id is None and persisted_summary.is_file():
        run_id = _latest_legacy_published_run_from_parallelism(parallelism, detector)

    # A published summary is itself the durable end-of-run artifact.  Older
    # results repositories may no longer retain the derived optimizer index or
    # the run-tagged parallelism rows that originally produced it, so parse the
    # published shape table as the final recovery source.  Do not require the
    # SVG: report publication can legitimately replace/regenerate that derived
    # presentation artifact independently of the completion summary.
    published_current = legacy_published_optimizer_index(persisted_summary, detector)
    if published_current is not None and run_id is not None:
        published_current["optimizer_run_id"] = str(run_id)
    if run_id is None and published_current is None:
        raise ValueError(f"No completed persisted optimizer run found for detector {detector}")

    if run_id is None:
        current_rows = legacy_optimizer_rows_from_indices([published_current], detector)
        current_legacy = build_optimizer_index(
            {"observations": current_rows}, detector, optimizer_run_id="legacy-published"
        ) if current_rows else None
        legacy_indices = historical_published_optimizer_indices(persisted_summary, detector)
        preferred_legacy = _canonicalize_published_optimizer_indices(legacy_indices, detector)
        if current_legacy is None or preferred_legacy is None:
            raise ValueError(f"No canonical optimizer evidence recovered for detector {detector}")
        current_legacy = _attach_prediction_history(results_root, current_legacy, detector, optimizer)
        preferred_legacy = _attach_prediction_history(results_root, preferred_legacy, detector, optimizer)
        return current_legacy, preferred_legacy, {}

    # Once modern completed-run evidence exists, recover distinct older
    # published profiles from results-repository history.  Do not reinterpret
    # the current working-tree summary as legacy evidence; it is the mutable
    # presentation of the latest modern run.
    historical_indices = [
        item for item in historical_published_optimizer_indices(persisted_summary, detector)
        if str(item.get("optimizer_run_id") or "") != "legacy-published"
    ]
    legacy_rows = legacy_optimizer_rows_from_indices(historical_indices, detector)
    if legacy_rows:
        observations = list(parallelism.get("observations", [])) if isinstance(parallelism.get("observations"), list) else []
        by_id = {str(row.get("observation_id")): row for row in observations if isinstance(row, dict) and row.get("observation_id")}
        for row in legacy_rows:
            if optimizer_evidence_is_valid(row):
                by_id.setdefault(str(row["observation_id"]), row)
        parallelism["observations"] = list(by_id.values())

    run_payload = _completed_run_payload(optimizer, detector, run_id)
    current = _attach_optimizer_run_metadata(build_optimizer_index(parallelism, detector, run_id), optimizer, detector)
    if not current.get("observation_count"):
        if published_current is not None:
            recovered_rows = legacy_optimizer_rows_from_indices([published_current], detector)
            recovered_current = build_optimizer_index(
                {"observations": recovered_rows}, detector, optimizer_run_id=str(run_id)
            ) if recovered_rows else None
            if recovered_current is not None and recovered_current.get("observation_count"):
                recovered_current = _attach_optimizer_run_metadata(recovered_current, optimizer, detector)
                recovered_current = _attach_prediction_history(results_root, recovered_current, detector, optimizer)
                return recovered_current, recovered_current, run_payload.get("run_metadata", {}) if isinstance(run_payload.get("run_metadata"), dict) else {}
        raise ValueError(
            f"Completed optimizer run {run_id} has no persisted completed shape observations for {detector}"
        )
    completed_ids = completed_optimizer_run_ids(optimizer, parallelism, detector)
    completed_ids.add(str(run_id))
    preferred = _attach_optimizer_run_metadata(build_optimizer_index(parallelism, detector, optimizer_run_ids=completed_ids), optimizer, detector)
    if not preferred.get("observation_count"):
        preferred = current
    current = _attach_prediction_history(results_root, current, detector, optimizer)
    preferred = _attach_prediction_history(results_root, preferred, detector, optimizer)
    run_metadata = run_payload.get("run_metadata") if isinstance(run_payload.get("run_metadata"), dict) else {}
    return current, preferred, run_metadata


def _completed_optimizer_detectors(results_root: Path) -> list[str]:
    # The optimizer index is reconstructable derived state. Persisted
    # execution-optimizer/<detector>/ reports remain valid completion evidence
    # when that index is missing (notably across results-layout migrations).
    optimizer = ResultsRepository(results_root).load_index("optimizer-index.json")
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


def _optimizer_profile_index(current: dict[str, Any], preferred: dict[str, Any]) -> dict[str, Any]:
    """Return the profile view used by aggregate Report Writer output.

    Preferred-shape selection remains compatibility scoped in ``preferred``.
    The visualization, however, must never omit the latest completed run just
    because aggregate completion metadata is stale or being reconstructed.
    Merge concrete-runner plot series from the latest completed run into the
    preferred/coalesced profile without changing any selection data.
    """
    profile = dict(preferred)
    merged: dict[str, dict[str, Any]] = {}
    for source in (preferred, current):
        for series in source.get("plot_series", []):
            if not isinstance(series, dict):
                continue
            runner_key = str(series.get("runner_key") or series.get("runner_title") or "unknown")
            run_id = str(series.get("optimizer_run_id") or "legacy-untagged")
            key = f"{runner_key}::run={run_id}"
            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(series)
                continue
            # Merge individual measured shapes by run+shape identity so a
            # repeated shape from a newly completed run remains visible rather
            # than being silently collapsed into older aggregate evidence.
            shapes: dict[tuple[str, str, int], dict[str, Any]] = {}
            for candidate in list(existing.get("shapes", [])) + list(series.get("shapes", [])):
                if not isinstance(candidate, dict):
                    continue
                shape_key = (
                    str(candidate.get("optimizer_run_id") or ""),
                    str(candidate.get("execution_shape") or ""),
                    int(candidate.get("optimizer_shape_sequence") or 0),
                )
                shapes[shape_key] = candidate
            merged_series = dict(existing)
            merged_series["shapes"] = sorted(
                shapes.values(),
                key=lambda shape: (
                    int(shape.get("pipelines") or 0),
                    int(shape.get("threads_per_pipeline") or 0),
                    str(shape.get("optimizer_run_id") or ""),
                    int(shape.get("optimizer_shape_sequence") or 0),
                ),
            )
            merged_series["best_shape"] = select_preferred_shape(merged_series["shapes"])
            merged[key] = merged_series
    profile["plot_series"] = sorted(merged.values(), key=lambda row: str(row.get("runner_title") or ""))
    return profile


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
        current, preferred, _ = _optimizer_report_components(results_root, detector)
        preferred_indices.append(preferred)
        profile_index = _optimizer_profile_index(current, preferred)
        (profiles_dir / f"{detector}.svg").write_text(render_heatmap_svg(profile_index), encoding="utf-8")

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
