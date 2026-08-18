"""Merge completed interleaved regression shards into one canonical detector run."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from hth.domain.result_metrics import baseline_surpassed

from .calibration_intelligence import build_calibration_intelligence
from .io import create_run_directory, write_json
from .parameter_space import canonical_parameters, canonical_search_space
from .parameter_provenance import attach_identity, build_provenance
from .reports import normalize_result_record, ranking_key, write_rankings, write_raw_results
from .runner import build_winner_page_report, file_sha256, load_pages, write_debug_artifacts
from hth.regression.result_metrics import aggregate_page_metrics


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _results_from_raw(path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parameter_id = row["parameter_set_id"]
            identity_sha = row.get("parameter_identity_sha256") or parameter_id
            result = grouped.setdefault(identity_sha, {
                "parameter_set_id": parameter_id,
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
            })
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
            result["pages"].append({
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
            })
    results = []
    for result in grouped.values():
        pages = result["pages"]
        successful = [page for page in pages if str(page.get("status") or "").strip().lower() in {"ok", "success"}]
        edges = [float(page["edge_error_mean_px"]) for page in successful if page.get("edge_error_mean_px") is not None]
        result["summary"] = aggregate_page_metrics(pages)
        result["summary"].update({
            "mean_edge_error_px": round(sum(edges) / len(edges), 3) if edges else None,
            "elapsed_ms_total": round(sum(float(page.get("elapsed_ms") or 0) for page in pages), 3),
            "wall_ms": round(sum(float(page.get("elapsed_ms") or 0) for page in pages), 3),
        })
        results.append(normalize_result_record(result))
    return results


def merge(shard_dirs: list[Path], output: Path, detector_config: Path, top: int = 20, *, expected_shard_count: int | None = None, golden_set: Path | None = None, image_root: Path | None = None, max_dimension: int = 1800, debug_level: str = "none") -> Path:
    if not shard_dirs:
        raise ValueError("No shard directories supplied")
    infos = [_read(path / "RUN-INFO.json") for path in shard_dirs]
    summaries = [_read(path / "reports" / "summary.json") for path in shard_dirs]
    detector = str(infos[0]["detector"])
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
    best_result = next((result for result in by_id.values() if result.get("profile") == "baseline"), None)
    for completion_index, (completed_at, result) in enumerate(completion_records, 1):
        elapsed_seconds = (completed_at - start).total_seconds()
        observation = {
            "completion_index": completion_index,
            "parameter_set_number": completion_index,
            "elapsed_seconds": max(0.0, elapsed_seconds),
            "search_fraction": completion_index / completion_total if completion_total else None,
        }
        result["search_observation"] = observation
        if best_result is None or ranking_key(result) < ranking_key(best_result):
            best_result = result
            winner_history.append({
                "change_number": len(winner_history) + 1,
                "parameter_set_id": str(result.get("parameter_set_id") or "unknown"),
                "parameter_short_name": result.get("profile") or str(result.get("parameter_set_id") or "unknown")[:8],
                **observation,
            })

    ranked = sorted(by_id.values(), key=ranking_key)
    detector_configuration = _read(detector_config)
    run_id, run_dir = create_run_directory(output, detector, None)
    for rank, result in enumerate(ranked, 1):
        attach_identity(result, detector, detector_configuration)
        result["rank"] = rank
        result["run_id"] = run_id
    search_ranked = [
        result for result in ranked
        if result.get("requested_search_member")
        and not result.get("reference_roles")
    ]
    for search_rank, result in enumerate(search_ranked, 1):
        result["search_rank"] = search_rank
    historic_best = next(
        (result for result in ranked if "historic_best" in (result.get("reference_roles") or [])),
        None,
    )
    first_summary = summaries[0]
    first_info = infos[0]
    strategy = str(first_info.get("strategy") or first_summary.get("strategy") or "exhaustive")
    requested_strategy = str(first_info.get("requested_strategy") or first_summary.get("requested_strategy") or strategy)
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
    parameter_provenance = build_provenance(
        detector,
        detector_configuration,
        ranked,
        strategy=strategy,
        complete_cartesian=(sum(1 for result in ranked if result.get("search_space_member")) >= possible),
    )
    write_json(run_dir / "parameter-provenance.json", parameter_provenance)
    baseline = next((result for result in ranked if result.get("profile") == "baseline"), None)
    pages = len(first_summary.get("page_ordinals", []))
    winner_pages = build_winner_page_report(ranked[0], baseline)
    serial_runtime_seconds = sum(
        max(0.0, float((result.get("summary") or {}).get("wall_ms") or (result.get("summary") or {}).get("elapsed_ms_total") or 0.0)) / 1000.0
        for result in ranked
    )
    effective_acceleration = serial_runtime_seconds / elapsed if elapsed > 0 else None
    shard_context = {"count": expected, "assignment": "interleaved", "source_run_ids": list(dict.fromkeys(info.get("run_id") for info in infos))}
    summary = {
        "schema_version": "0.8", "run_id": run_id, "detector": detector,
        "strategy": strategy, "requested_strategy": requested_strategy,
        "threads": max(int(info.get("threads") or 1) for info in infos), "shard": shard_context,
        "parameter_space": {"possible_parameter_sets": possible, "live_possible_parameter_sets": live_possible, "zombie_possible_parameter_sets": zombie_possible, "canonical_search_space": search_space_contract, "planned_parameter_sets": len(ranked), "actual_parameter_sets": len(ranked), "golden_set_pages": pages, "planned_page_evaluations": len(ranked) * pages, "actual_page_evaluations": len(ranked) * pages},
        "page_ordinals": first_summary.get("page_ordinals", []), "parameter_set_count": len(ranked),
        "page_evaluation_count": len(ranked) * pages,
        "successful_page_evaluation_count": sum(r["summary"]["success_count"] for r in ranked),
        "fully_successful_parameter_set_count": sum(1 for r in ranked if r["summary"]["failure_count"] == 0),
        "golden_set_sha256": first_info.get("golden_set_sha256"), "winner": ranked[0], "baseline": baseline,
        "historic_best": historic_best, "top_parameter_sets": ranked[:5], "search_top_parameter_sets": search_ranked[:5], "winner_page_report": winner_pages,
        "runner": first_summary.get("runner", {}), "source_commit": first_info.get("source_commit"),
        "elapsed_seconds": round(elapsed, 3),
        "estimated_serial_runtime_seconds": round(serial_runtime_seconds, 3),
        "effective_acceleration": round(effective_acceleration, 4) if effective_acceleration is not None else None,
        "progress": {"estimated_parameter_sets": completion_total, "completed_parameter_sets": completion_total, "average_eval_rate": completion_total / elapsed if elapsed else None, "failures": sum(r["summary"]["failure_count"] for r in ranked), "winner_changes": len(winner_history), "winner_history": winner_history, "winner_first_changed_elapsed_seconds": winner_history[0]["elapsed_seconds"] if winner_history else None, "winner_last_changed_elapsed_seconds": winner_history[-1]["elapsed_seconds"] if winner_history else None, "baseline_surpassed": baseline_surpassed(ranked[0], baseline)},
    }
    write_raw_results(run_dir / "raw" / "results.csv", ranked)
    write_rankings(run_dir / "reports" / "rankings.csv", ranked)
    write_rankings(run_dir / "reports" / "top20.csv", ranked[:max(0, top)])
    write_json(run_dir / "reports" / "summary.json", summary)
    write_json(run_dir / "reports" / "winner-pages.json", winner_pages)
    try:
        golden_set_payload = _read(golden_set) if golden_set is not None else {}
    except (OSError, ValueError, json.JSONDecodeError):
        golden_set_payload = {}
    source_document = golden_set_payload.get("source_document") if isinstance(golden_set_payload, dict) else None
    golden_set_identity = {
        "configuration": str(golden_set) if golden_set is not None else first_info.get("golden_set"),
        "sha256": first_info.get("golden_set_sha256"),
        "collection_id": golden_set_payload.get("collection_id") if isinstance(golden_set_payload, dict) else None,
        "schema_version": golden_set_payload.get("schema_version") if isinstance(golden_set_payload, dict) else None,
        "description": golden_set_payload.get("description") if isinstance(golden_set_payload, dict) else None,
        "page_count": pages,
        "page_ordinals": first_summary.get("page_ordinals", []),
    }
    calibration_context = {
        "calibration_run_id": run_id,
        "calibration_schema_version": "1.1",
        "created_at_utc": start.isoformat(),
        "source_document": source_document,
        "golden_set": golden_set_identity,
        "detector_configuration": {
            "detector_id": detector,
            "configuration": str(detector_config),
            "sha256": file_sha256(detector_config),
        },
        "pipeline": {
            "commit": first_summary.get("runner", {}).get("pipeline_commit"),
            "source_commit": first_info.get("source_commit"),
            "python": first_summary.get("runner", {}).get("python_version"),
            "opencv": first_summary.get("runner", {}).get("opencv_version"),
        },
    }
    calibration = build_calibration_intelligence(
        ranked,
        detector=detector,
        strategy=strategy,
        possible_parameter_sets=possible,
        calibration_context=calibration_context,
        regression_context={
            "requested_strategy": requested_strategy,
            "resolved_strategy": strategy,
            "configured_threads": summary["threads"],
            "possible_parameter_sets": possible,
            "live_possible_parameter_sets": live_possible,
            "zombie_possible_parameter_sets": zombie_possible,
            "baseline_parameters": dict(detector_configuration.get("profiles", {}).get("baseline", {})),
            "fixed_parameter_policy": "baseline",
            "zombie_parameters": list(search_space_contract["configured_zombie_parameters"]),
            "zombie_parameter_evidence": {str(name): dict(spec.get("last_measured", {})) for name, spec in (detector_configuration.get("zombie_parameters", {}) if isinstance(detector_configuration.get("zombie_parameters"), dict) else {}).items() if isinstance(spec, dict) and isinstance(spec.get("last_measured"), dict)},
            "canonical_search_space": search_space_contract,
            "planned_parameter_sets": len(ranked),
            "evaluated_parameter_sets": len(ranked),
            "golden_set_pages": pages,
            "page_evaluations": len(ranked) * pages,
            "failed_page_evaluations": summary["progress"]["failures"],
            "average_eval_rate": summary["progress"]["average_eval_rate"],
            "execution_environment": first_summary.get("runner", {}),
            "shard": shard_context,
        },
    )
    write_json(run_dir / "reports" / "calibration-intelligence.json", calibration)
    debug_outputs: list[str] = []
    if debug_level != "none" and golden_set is not None and image_root is not None:
        pages_payload = load_pages(golden_set, image_root, max_dimension)
        debug_outputs = write_debug_artifacts(output, detector, run_id, policy="winner", ranked=ranked, pages=pages_payload, debug_level=debug_level)
    parameters = _read(shard_dirs[0] / "parameters.json")
    parameters["shard"] = shard_context
    write_json(run_dir / "parameters.json", parameters)
    info = dict(first_info)
    info.update({
        "run_id": run_id,
        "started_at_utc": start.isoformat(),
        "finished_at_utc": finish.isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "wall_elapsed_seconds": round(elapsed, 3),
        "estimated_serial_runtime_seconds": round(serial_runtime_seconds, 3),
        "effective_acceleration": round(effective_acceleration, 4) if effective_acceleration is not None else None,
        "actual_parameter_sets": len(ranked),
        "planned_parameter_sets": len(ranked),
        "shard_index": None,
        "shard_count": expected,
        "shard": shard_context,
    })
    write_json(run_dir / "RUN-INFO.json", info)
    write_json(run_dir / "manifest.json", {"schema_version": "0.1", "run_id": run_id, "detector": detector, "strategy": strategy, "status": "complete", "started_at_utc": start.isoformat(), "finished_at_utc": finish.isoformat(), "shard": shard_context, "outputs": ["RUN-INFO.json", "parameters.json", "parameter-provenance.json", "raw/results.csv", "reports/summary.json", "reports/winner-pages.json", "reports/calibration-intelligence.json", "reports/rankings.csv", "reports/top20.csv"], "debug_outputs": debug_outputs})
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
