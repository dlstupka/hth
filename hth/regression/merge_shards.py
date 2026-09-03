"""Merge completed interleaved regression shards into one canonical detector run."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from hth.domain.result_metrics import baseline_surpassed

from .io import create_run_directory, write_json
from .materialization import (
    build_canonical_calibration_from_summary,
    build_canonical_manifest,
    build_canonical_summary,
    derive_canonical_outcome,
    write_canonical_reports,
)
from .parameter_space import canonical_search_space
from .parameter_provenance import attach_identity, build_provenance
from .reports import normalize_result_record, ranking_key, write_rankings
from .outcome import is_winner_eligible
from .run_semantics import evidence_tier_for, legacy_run_semantics
from .runner import build_winner_page_report, file_sha256, load_pages, write_debug_artifacts
from hth.regression.result_metrics import aggregate_page_metrics


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _require_common(label: str, values: list[Any]) -> Any:
    """Return one shard invariant or reject an incompatible evidence set."""
    encoded = {
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        for value in values
    }
    if len(encoded) != 1:
        raise ValueError(f"Shard {label} mismatch: {values!r}")
    return values[0]


def _merged_pipeline_context(infos: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    contexts = [
        info.get("detector_pipeline") or summary.get("detector_pipeline")
        for info, summary in zip(infos, summaries)
    ]
    contexts = [context for context in contexts if isinstance(context, dict)]
    if not contexts:
        return None
    merged: dict[str, Any] = {"source_shards": len(infos)}
    varying_keys = {"pipeline_number", "queue_position"}
    for key in sorted(set().union(*(context.keys() for context in contexts)) - varying_keys):
        values = [context.get(key) for context in contexts]
        if all(value == values[0] for value in values):
            merged[key] = values[0]
        else:
            merged[f"{key}_values"] = values
    merged["pipeline_number"] = None
    merged["pipeline_numbers"] = sorted({
        int(context["pipeline_number"])
        for context in contexts
        if context.get("pipeline_number") is not None
    })
    merged["queue_positions"] = sorted({
        int(context["queue_position"])
        for context in contexts
        if str(context.get("queue_position") or "").isdigit()
    })
    return merged


def _results_from_raw(path: Path) -> list[dict[str, Any]]:
    compressed = path.suffix == ".gz"
    evidence_path = path.with_name("evidence.jsonl.gz" if compressed else "evidence.jsonl")
    if evidence_path.is_file():
        opener = gzip.open if compressed else open
        with opener(evidence_path, "rt", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if not all(isinstance(record, dict) for record in records):
            raise ValueError(f"Raw evidence must contain JSON objects: {evidence_path}")
        results = [
            record["result"] if isinstance(record.get("result"), dict) else record
            for record in records
        ]
        return _finalize_raw_results(results, normalize_optional=False)

    grouped: dict[str, dict[str, Any]] = {}
    opener = gzip.open if compressed else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result_envelope = json.loads(row.get("result_json") or "null")
            page_envelope = json.loads(row.get("page_json") or "null")
            parameter_id = row["parameter_set_id"]
            identity_sha = row.get("parameter_identity_sha256") or parameter_id
            legacy_result = {
                "parameter_set_id": parameter_id,
                "parameter_set_equivalence_family_id": row.get("parameter_set_equivalence_family_id") or None,
                "parameter_set_equivalence_family_sha256": row.get("parameter_set_equivalence_family_sha256") or None,
                "parameter_set_equivalence_family_size": int(row["parameter_set_equivalence_family_size"]) if row.get("parameter_set_equivalence_family_size") not in (None, "") else None,
                "parameter_identity_sha256": row.get("parameter_identity_sha256") or None,
                "parameter_schema_version": row.get("parameter_schema_version") or None,
                "parameter_grid_sha256": row.get("parameter_grid_sha256") or None,
                "parameter_grid_ordinal": int(row["parameter_grid_ordinal"]) if row.get("parameter_grid_ordinal") not in (None, "") else None,
                "profile": row.get("profile") or None,
                "search_rank": int(row["search_rank"]) if row.get("search_rank") not in (None, "") else None,
                "requested_search_member": str(row.get("requested_search_member") or "0") in {"1", "true", "True"},
                "search_space_member": str(row.get("search_space_member") or row.get("requested_search_member") or "0") in {"1", "true", "True"},
                "reference_roles": json.loads(row.get("reference_roles_json") or "[]"),
                "historic_reference": json.loads(row.get("historic_reference_json") or "{}"),
                "parameters": json.loads(row["parameters_json"]),
                "pages": [],
            }
            if isinstance(result_envelope, dict):
                result_seed = dict(result_envelope)
                result_seed["pages"] = []
                for key, value in legacy_result.items():
                    result_seed.setdefault(key, value)
            else:
                result_seed = legacy_result
            result = grouped.setdefault(identity_sha, result_seed)
            if "search_observation" not in result:
                completion_index = row.get("completion_index")
                elapsed = row.get("completion_elapsed_seconds")
                fraction = row.get("search_fraction")
                if completion_index not in (None, "") or elapsed not in (None, ""):
                    result["search_observation"] = {
                        "completion_index": int(completion_index) if completion_index not in (None, "") else None,
                        "parameter_set_number": int(completion_index) if completion_index not in (None, "") else None,
                        "elapsed_seconds": float(elapsed) if elapsed not in (None, "") else None,
                        "search_fraction": float(fraction) if fraction not in (None, "") else None,
                    }
            edge_values = {
                key.removesuffix("_error_px"): float(row[key])
                for key in ("left_error_px", "top_error_px", "right_error_px", "bottom_error_px")
                if row.get(key) not in (None, "")
            }
            error = None
            if row.get("error_type") or row.get("error_message"):
                error = {"type": row.get("error_type"), "message": row.get("error_message")}
            legacy_page = {
                "global_ordinal": int(row["global_ordinal"]),
                "label": row["label"],
                "layout_type": row["layout_type"],
                "status": row["status"],
                "iou": float(row["iou"] or 0),
                "edge_errors": edge_values,
                "edge_error_mean_px": float(row["edge_error_mean_px"]) if row.get("edge_error_mean_px") else None,
                "edge_error_maximum_px": float(row["edge_error_maximum_px"]) if row.get("edge_error_maximum_px") else None,
                "elapsed_ms": float(row["elapsed_ms"] or 0),
                "approved_bbox": json.loads(row["approved_bbox_json"]),
                "predicted_bbox": json.loads(row["predicted_bbox_json"]),
                "error": error,
            }
            result["pages"].append(page_envelope if isinstance(page_envelope, dict) else legacy_page)
    return _finalize_raw_results(list(grouped.values()))


def _finalize_raw_results(
    results: list[dict[str, Any]], *, normalize_optional: bool = True
) -> list[dict[str, Any]]:
    """Recompute canonical derived metrics without discarding source fields."""
    finalized = []
    for result in results:
        pages = result["pages"]
        successful = [page for page in pages if str(page.get("status") or "").strip().lower() in {"ok", "success"}]
        edges = [float(page["edge_error_mean_px"]) for page in successful if page.get("edge_error_mean_px") is not None]
        summary = dict(result.get("summary") or {})
        summary.update(aggregate_page_metrics(pages))
        summary.update({
            "mean_edge_error_px": round(sum(edges) / len(edges), 3) if edges else None,
            "elapsed_ms_total": round(sum(float(page.get("elapsed_ms") or 0) for page in pages), 3),
            "wall_ms": round(sum(float(page.get("elapsed_ms") or 0) for page in pages), 3),
        })
        result["summary"] = summary
        finalized.append(normalize_result_record(result) if normalize_optional else result)
    return finalized


def merge(shard_dirs: list[Path], output: Path, detector_config: Path, top: int = 20, *, expected_shard_count: int | None = None, golden_set: Path | None = None, image_root: Path | None = None, max_dimension: int = 1800, debug_level: str = "none") -> Path:
    if not shard_dirs:
        raise ValueError("No shard directories supplied")
    infos = [_read(path / "RUN-INFO.json") for path in shard_dirs]
    summaries = [_read(path / "reports" / "summary.json") for path in shard_dirs]
    detector = str(_require_common("detector", [str(info.get("detector") or "") for info in infos]))
    _require_common(
        "strategy",
        [str(info.get("strategy") or summary.get("strategy") or "") for info, summary in zip(infos, summaries)],
    )
    _require_common(
        "Golden Set identity",
        [info.get("golden_set_sha256") or summary.get("golden_set_sha256") for info, summary in zip(infos, summaries)],
    )
    _require_common(
        "detector configuration identity",
        [info.get("detector_config_sha256") or summary.get("detector_config_sha256") for info, summary in zip(infos, summaries)],
    )
    _require_common(
        "model selection",
        [info.get("model_selection", summary.get("model_selection")) for info, summary in zip(infos, summaries)],
    )
    _require_common(
        "maximum dimension",
        [info.get("max_dimension", summary.get("max_dimension")) for info, summary in zip(infos, summaries)],
    )
    _require_common(
        "Golden Set pages",
        [summary.get("page_ordinals", []) for summary in summaries],
    )
    metadata_counts = {
        int(info.get("shard_count") or summary.get("shard", {}).get("count") or 1)
        for info, summary in zip(infos, summaries)
    }
    expected = int(expected_shard_count) if expected_shard_count is not None else max(metadata_counts)
    found = {
        int(info.get("shard_index") if info.get("shard_index") is not None else summary.get("shard", {}).get("index") or 0)
        for info, summary in zip(infos, summaries)
    }
    expected_indexes = set(range(expected))
    missing = sorted(expected_indexes - found)
    extra = sorted(found - expected_indexes)
    if len(metadata_counts) != 1 or missing or extra:
        raise ValueError(
            "Invalid shard set: "
            f"expected_count={expected}, metadata_counts={sorted(metadata_counts)}, "
            f"found={sorted(found)}, missing={missing}, extra={extra}"
        )

    starts = [_parse_time(str(info["started_at_utc"])) for info in infos]
    start = min(starts)
    measured_elapsed = [max(0.0, float(info.get("elapsed_seconds") or 0.0)) for info in infos]
    elapsed = max(
        ((shard_start - start).total_seconds() + shard_elapsed)
        for shard_start, shard_elapsed in zip(starts, measured_elapsed)
    )
    finish = start + timedelta(seconds=elapsed)

    by_id: dict[str, dict[str, Any]] = {}
    completion_records: list[tuple[datetime, dict[str, Any]]] = []
    for shard_dir, info in zip(shard_dirs, infos):
        shard_start = _parse_time(str(info["started_at_utc"]))
        for result in _results_from_raw(shard_dir / "raw" / "results.csv"):
            existing = by_id.setdefault(str(result.get("parameter_identity_sha256") or result["parameter_set_id"]), result)
            if existing is not result:
                continue
            observation = result.get("search_observation") or {}
            elapsed_seconds = observation.get("elapsed_seconds")
            local_completion_index = observation.get("completion_index")
            if result.get("profile") != "baseline" and (elapsed_seconds is not None or local_completion_index is not None):
                if elapsed_seconds is not None:
                    completed_at = shard_start + timedelta(seconds=float(elapsed_seconds))
                else:
                    completed_at = shard_start + timedelta(microseconds=int(local_completion_index))
                completion_records.append((completed_at, result))

    completion_records.sort(key=lambda item: (item[0], str(item[1].get("parameter_set_id") or "")))
    completion_total = len(completion_records)
    winner_history: list[dict[str, Any]] = []
    best_result = next((result for result in by_id.values() if result.get("profile") == "baseline" and is_winner_eligible(result)), None)
    for completion_index, (completed_at, result) in enumerate(completion_records, 1):
        elapsed_seconds = (completed_at - start).total_seconds()
        observation = {
            "completion_index": completion_index,
            "parameter_set_number": completion_index,
            "elapsed_seconds": max(0.0, elapsed_seconds),
            "search_fraction": completion_index / completion_total if completion_total else None,
        }
        result["search_observation"] = observation
        if is_winner_eligible(result) and (best_result is None or ranking_key(result) < ranking_key(best_result)):
            best_result = result
            winner_history.append({
                "change_number": len(winner_history) + 1,
                "parameter_set_id": str(result.get("parameter_set_id") or "unknown"),
                "parameter_short_name": result.get("profile") or str(result.get("parameter_set_id") or "unknown")[:8],
                **observation,
            })

    detector_configuration = _read(detector_config)
    first_summary = summaries[0]
    first_info = infos[0]
    run_mode = _require_common(
        "run mode",
        [legacy_run_semantics(info, summary)[0] for info, summary in zip(infos, summaries)],
    )
    strategy = str(first_info.get("strategy") or first_summary.get("strategy") or "exhaustive")
    requested_strategy = str(first_info.get("requested_strategy") or first_summary.get("requested_strategy") or strategy)
    run_id, run_dir = create_run_directory(output, detector, None)
    for result in by_id.values():
        attach_identity(result, detector, detector_configuration, strategy=strategy)
        result["run_id"] = run_id
    outcome = derive_canonical_outcome(
        by_id.values(),
        ranking_key=ranking_key,
        winner_page_builder=build_winner_page_report,
    )
    ordered = outcome.ordered
    ranked = outcome.ranked
    winner = outcome.winner
    measurement_state = outcome.measurement_state
    identity_by_parameter_set = {
        str(result.get("parameter_set_id")): result.get("parameter_set_equivalence_family_id")
        for result in ordered
        if result.get("parameter_set_id") and result.get("parameter_set_equivalence_family_id")
    }
    for event in winner_history:
        family_id = identity_by_parameter_set.get(str(event.get("parameter_set_id")))
        if family_id:
            event["parameter_set_equivalence_family_id"] = family_id
    search_space_contract = canonical_search_space(detector_configuration, strategy)
    live_possible = int(search_space_contract["live_exhaustive_parameter_sets"])
    zombie_possible = int(search_space_contract["exhaustive_with_zombies_parameter_sets"])
    if strategy == "exhaustive-with-zombies":
        possible = zombie_possible
    elif strategy in {"exhaustive", "cartesian"}:
        possible = live_possible
    else:
        possible = int(first_info.get("possible_parameter_sets") or first_summary.get("parameter_space", {}).get("possible_parameter_sets") or live_possible)
    search_space_contract["effective_parameter_sets"] = possible
    complete_cartesian = (
        sum(1 for result in ordered if result.get("search_space_member")) >= possible
    )
    evidence_tier = evidence_tier_for(
        run_mode, exhaustive_complete=complete_cartesian
    )
    parameter_provenance = build_provenance(
        detector,
        detector_configuration,
        ordered,
        strategy=strategy,
        complete_cartesian=complete_cartesian,
    )
    write_json(run_dir / "parameter-provenance.json", parameter_provenance)
    baseline = outcome.baseline
    pages = len(first_summary.get("page_ordinals", []))
    serial_runtime_seconds = sum(
        max(0.0, float((result.get("summary") or {}).get("wall_ms") or (result.get("summary") or {}).get("elapsed_ms_total") or 0.0)) / 1000.0
        for result in ordered
    )
    effective_acceleration = serial_runtime_seconds / elapsed if elapsed > 0 else None
    shard_context = {"count": expected, "assignment": "interleaved", "source_run_ids": list(dict.fromkeys(info.get("run_id") for info in infos))}
    threads = max(int(info.get("threads") or 1) for info in infos)
    strategy_fallback_reason = first_info.get("strategy_fallback_reason", first_summary.get("strategy_fallback_reason"))
    detector_pipeline = _merged_pipeline_context(infos, summaries)
    runner_context = first_summary.get("runner", {})
    model_selection = first_info.get("model_selection") or first_summary.get("model_selection")
    resolved_max_dimension = first_info.get("max_dimension", first_summary.get("max_dimension", max_dimension))
    detector_config_sha256 = first_info.get("detector_config_sha256") or file_sha256(detector_config)
    parameter_space = {"possible_parameter_sets": possible, "live_possible_parameter_sets": live_possible, "zombie_possible_parameter_sets": zombie_possible, "canonical_search_space": search_space_contract, "planned_parameter_sets": len(ordered), "actual_parameter_sets": len(ordered), "golden_set_pages": pages, "planned_page_evaluations": len(ordered) * pages, "actual_page_evaluations": len(ordered) * pages}
    progress_payload = {"estimated_parameter_sets": completion_total, "completed_parameter_sets": completion_total, "average_eval_rate": completion_total / elapsed if elapsed else None, "failures": sum(r["summary"]["failure_count"] for r in ordered), "winner_changes": len(winner_history) if winner else 0, "winner_history": winner_history if winner else [], "winner_first_changed_elapsed_seconds": winner_history[0]["elapsed_seconds"] if winner and winner_history else None, "winner_last_changed_elapsed_seconds": winner_history[-1]["elapsed_seconds"] if winner and winner_history else None, "baseline_surpassed": baseline_surpassed(winner, baseline)}
    performance_payload = {
        "sample_count": sum(int((summary.get("performance") or {}).get("sample_count") or 0) for summary in summaries),
        "configured_threads": threads,
        "peak_rss_bytes": max((int((summary.get("performance") or {}).get("peak_rss_bytes") or 0) for summary in summaries), default=0),
        "source_shards": expected,
        "precomputed_evidence": any(bool((summary.get("performance") or {}).get("precomputed_evidence")) for summary in summaries),
        "evidence_source": next((str((summary.get("performance") or {}).get("evidence_source")) for summary in summaries if (summary.get("performance") or {}).get("evidence_source")), None),
    }
    summary = build_canonical_summary(
        outcome,
        run_id=run_id,
        detector=detector,
        run_mode=run_mode,
        evidence_tier=evidence_tier,
        strategy=strategy,
        requested_strategy=requested_strategy,
        strategy_fallback_reason=strategy_fallback_reason,
        threads=threads,
        shard=shard_context,
        detector_pipeline=detector_pipeline,
        parameter_space=parameter_space,
        page_ordinals=first_summary.get("page_ordinals", []),
        golden_set_sha256=first_info.get("golden_set_sha256"),
        detector_config_sha256=detector_config_sha256,
        model_selection=model_selection,
        max_dimension=resolved_max_dimension,
        runner=runner_context,
        source_commit=first_info.get("source_commit"),
        performance=performance_payload,
        progress=progress_payload,
        extra={
            "elapsed_seconds": round(elapsed, 3),
            "estimated_serial_runtime_seconds": round(serial_runtime_seconds, 3),
            "effective_acceleration": round(effective_acceleration, 4) if effective_acceleration is not None else None,
        },
    )
    try:
        golden_set_payload = _read(golden_set) if golden_set is not None else {}
    except (OSError, ValueError, json.JSONDecodeError):
        golden_set_payload = {}
    calibration = build_canonical_calibration_from_summary(
        outcome,
        summary=summary,
        created_at_utc=start.isoformat(),
        golden_set_configuration=golden_set if golden_set is not None else first_info.get("golden_set"),
        golden_set_payload=golden_set_payload,
        detector_configuration=detector_config,
        detector_config=detector_configuration,
        regression_metadata_extra={"shard": shard_context},
    )
    write_canonical_reports(
        run_dir,
        outcome,
        summary=summary,
        calibration=calibration,
        top=top,
    )
    debug_outputs: list[str] = []
    if debug_level != "none" and golden_set is not None and image_root is not None:
        pages_payload = load_pages(golden_set, image_root, max_dimension)
        diagnostic_ranked = ranked if ranked else ([baseline] if baseline is not None else ordered[:1])
        debug_outputs = write_debug_artifacts(output, detector, run_id, policy="winner", ranked=diagnostic_ranked, pages=pages_payload, debug_level=debug_level)
    parameters = _read(shard_dirs[0] / "parameters.json")
    parameters["shard"] = shard_context
    write_json(run_dir / "parameters.json", parameters)
    info = dict(first_info)
    info.update({
        "schema_version": "0.5",
        "run_id": run_id,
        "run_mode": run_mode,
        "evidence_tier": evidence_tier,
        "started_at_utc": start.isoformat(),
        "finished_at_utc": finish.isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "wall_elapsed_seconds": round(elapsed, 3),
        "estimated_serial_runtime_seconds": round(serial_runtime_seconds, 3),
        "effective_acceleration": round(effective_acceleration, 4) if effective_acceleration is not None else None,
        "actual_parameter_sets": len(ordered),
        "planned_parameter_sets": len(ordered),
        "status": "complete" if measurement_state["terminal_success"] else "invalid",
        "outcome": measurement_state,
        "shard_index": None,
        "shard_count": expected,
        "shard": shard_context,
    })
    write_json(run_dir / "RUN-INFO.json", info)
    manifest = build_canonical_manifest(
        outcome,
        run_id=run_id,
        detector=detector,
        run_mode=run_mode,
        evidence_tier=evidence_tier,
        strategy=strategy,
        requested_strategy=requested_strategy,
        strategy_fallback_reason=strategy_fallback_reason,
        started_at_utc=start.isoformat(),
        finished_at_utc=finish.isoformat(),
        shard=shard_context,
        debug_outputs=debug_outputs,
    )
    write_json(run_dir / "manifest.json", manifest)
    write_rankings(run_dir.parent / f"{detector}-regression-results.csv", ranked)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--expected-shard-count", type=int)
    parser.add_argument("--golden-set", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--max-dimension", type=int, default=1800)
    parser.add_argument("--debug-level", choices=("none", "basic", "verbose"), default="none")
    args = parser.parse_args(argv)
    print(merge(args.shard_dir, args.output, args.detector_config, args.top, expected_shard_count=args.expected_shard_count, golden_set=args.golden_set, image_root=args.image_root, max_dimension=args.max_dimension, debug_level=args.debug_level))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
