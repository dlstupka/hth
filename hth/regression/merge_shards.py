"""Merge completed interleaved regression shards into one canonical detector run."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from .calibration_intelligence import build_calibration_intelligence
from .io import create_run_directory, write_json
from .parameter_space import canonical_parameters
from .reports import ranking_key, write_rankings, write_raw_results
from .runner import build_winner_page_report, load_pages, write_debug_artifacts


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _results_from_raw(path: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parameter_id = row["parameter_set_id"]
            result = grouped.setdefault(parameter_id, {
                "parameter_set_id": parameter_id,
                "profile": row.get("profile") or None,
                "parameters": json.loads(row["parameters_json"]),
                "pages": [],
            })
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
        successful = [page for page in pages if page["status"] == "success"]
        ious = [float(page["iou"]) for page in successful] or [0.0]
        edges = [float(page["edge_error_mean_px"]) for page in successful if page.get("edge_error_mean_px") is not None]
        result["summary"] = {
            "page_count": len(pages),
            "success_count": len(successful),
            "failure_count": len(pages) - len(successful),
            "mean_iou": round(sum(ious) / len(ious), 8),
            "minimum_iou": round(min(ious), 8),
            "stddev_iou": round(statistics.pstdev(ious), 8),
            "mean_edge_error_px": round(sum(edges) / len(edges), 3) if edges else None,
            "elapsed_ms_total": round(sum(float(page.get("elapsed_ms") or 0) for page in pages), 3),
            "wall_ms": round(sum(float(page.get("elapsed_ms") or 0) for page in pages), 3),
        }
        results.append(result)
    return results


def merge(shard_dirs: list[Path], output: Path, detector_config: Path, top: int = 20, *, golden_set: Path | None = None, image_root: Path | None = None, max_dimension: int = 1800, debug_level: str = "none") -> Path:
    if not shard_dirs:
        raise ValueError("No shard directories supplied")
    infos = [_read(path / "RUN-INFO.json") for path in shard_dirs]
    summaries = [_read(path / "reports" / "summary.json") for path in shard_dirs]
    detector = str(infos[0]["detector"])
    expected = int(infos[0].get("shard_count") or summaries[0].get("shard", {}).get("count") or 1)
    found = {int(info.get("shard_index") or summary.get("shard", {}).get("index") or 0) for info, summary in zip(infos, summaries)}
    if found != set(range(expected)):
        raise ValueError(f"Incomplete shard set: expected {expected}, found {sorted(found)}")

    by_id: dict[str, dict[str, Any]] = {}
    for shard_dir in shard_dirs:
        for result in _results_from_raw(shard_dir / "raw" / "results.csv"):
            by_id.setdefault(result["parameter_set_id"], result)
    ranked = sorted(by_id.values(), key=ranking_key)
    run_id, run_dir = create_run_directory(output, detector, None)
    for rank, result in enumerate(ranked, 1):
        result["rank"] = rank
        result["run_id"] = run_id
    baseline = next((result for result in ranked if result.get("profile") == "baseline"), None)
    first_summary = summaries[0]
    first_info = infos[0]
    start = min(_parse_time(str(info["started_at_utc"])) for info in infos)
    finish = max(_parse_time(str(info["finished_at_utc"])) for info in infos)
    elapsed = (finish - start).total_seconds()
    pages = len(first_summary.get("page_ordinals", []))
    possible = int(first_info.get("possible_parameter_sets") or first_summary.get("parameter_space", {}).get("possible_parameter_sets") or len(ranked))
    winner_pages = build_winner_page_report(ranked[0], baseline)
    shard_context = {"count": expected, "assignment": "interleaved", "source_run_ids": [info.get("run_id") for info in infos]}
    summary = {
        "schema_version": "0.8", "run_id": run_id, "detector": detector,
        "strategy": "exhaustive", "requested_strategy": "exhaustive",
        "threads": max(int(info.get("threads") or 1) for info in infos), "shard": shard_context,
        "parameter_space": {"possible_parameter_sets": possible, "planned_parameter_sets": len(ranked), "actual_parameter_sets": len(ranked), "golden_set_pages": pages, "planned_page_evaluations": len(ranked) * pages, "actual_page_evaluations": len(ranked) * pages},
        "page_ordinals": first_summary.get("page_ordinals", []), "parameter_set_count": len(ranked),
        "page_evaluation_count": len(ranked) * pages,
        "successful_page_evaluation_count": sum(r["summary"]["success_count"] for r in ranked),
        "fully_successful_parameter_set_count": sum(1 for r in ranked if r["summary"]["failure_count"] == 0),
        "golden_set_sha256": first_info.get("golden_set_sha256"), "winner": ranked[0], "baseline": baseline,
        "top_parameter_sets": ranked[:5], "winner_page_report": winner_pages,
        "runner": first_summary.get("runner", {}), "source_commit": first_info.get("source_commit"),
        "progress": {"estimated_parameter_sets": len(ranked), "completed_parameter_sets": len(ranked), "average_eval_rate": len(ranked) / elapsed if elapsed else None, "failures": sum(r["summary"]["failure_count"] for r in ranked)},
    }
    write_raw_results(run_dir / "raw" / "results.csv", ranked)
    write_rankings(run_dir / "reports" / "rankings.csv", ranked)
    write_rankings(run_dir / "reports" / "top20.csv", ranked[:max(0, top)])
    write_json(run_dir / "reports" / "summary.json", summary)
    write_json(run_dir / "reports" / "winner-pages.json", winner_pages)
    calibration = build_calibration_intelligence(ranked, detector=detector, strategy="exhaustive", possible_parameter_sets=possible, calibration_context={"calibration_run_id": run_id, "calibration_schema_version": "1.1", "created_at_utc": start.isoformat(), "golden_set": {"sha256": first_info.get("golden_set_sha256"), "page_count": pages, "page_ordinals": first_summary.get("page_ordinals", [])}, "detector_configuration": {"detector_id": detector, "configuration": str(detector_config)}}, regression_context={"requested_strategy": "exhaustive", "resolved_strategy": "exhaustive", "configured_threads": summary["threads"], "possible_parameter_sets": possible, "planned_parameter_sets": len(ranked), "evaluated_parameter_sets": len(ranked), "golden_set_pages": pages, "page_evaluations": len(ranked) * pages, "average_eval_rate": summary["progress"]["average_eval_rate"], "shard": shard_context})
    write_json(run_dir / "reports" / "calibration-intelligence.json", calibration)
    debug_outputs: list[str] = []
    if debug_level != "none" and golden_set is not None and image_root is not None:
        pages_payload = load_pages(golden_set, image_root, max_dimension)
        debug_outputs = write_debug_artifacts(output, detector, run_id, policy="winner", ranked=ranked, pages=pages_payload, debug_level=debug_level)
    parameters = _read(shard_dirs[0] / "parameters.json")
    parameters["shard"] = shard_context
    write_json(run_dir / "parameters.json", parameters)
    info = dict(first_info)
    info.update({"run_id": run_id, "started_at_utc": start.isoformat(), "finished_at_utc": finish.isoformat(), "elapsed_seconds": round(elapsed, 3), "actual_parameter_sets": len(ranked), "planned_parameter_sets": len(ranked), "shard_index": None, "shard_count": expected, "shard": shard_context})
    write_json(run_dir / "RUN-INFO.json", info)
    write_json(run_dir / "manifest.json", {"schema_version": "0.1", "run_id": run_id, "detector": detector, "strategy": "exhaustive", "status": "complete", "started_at_utc": start.isoformat(), "finished_at_utc": finish.isoformat(), "shard": shard_context, "outputs": ["RUN-INFO.json", "parameters.json", "raw/results.csv", "reports/summary.json", "reports/winner-pages.json", "reports/calibration-intelligence.json", "reports/rankings.csv", "reports/top20.csv"], "debug_outputs": debug_outputs})
    write_rankings(run_dir.parent / f"{detector}-regression-results.csv", ranked)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-config", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--golden-set", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--max-dimension", type=int, default=1800)
    parser.add_argument("--debug-level", choices=("none", "basic", "verbose"), default="none")
    args = parser.parse_args(argv)
    print(merge(args.shard_dir, args.output, args.detector_config, args.top, golden_set=args.golden_set, image_root=args.image_root, max_dimension=args.max_dimension, debug_level=args.debug_level))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
