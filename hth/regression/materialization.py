"""Canonical construction of derived detector-regression run artifacts.

Execution paths may collect evidence differently (direct, sharded, or
historical), but they must reduce that evidence and describe the resulting run
through this module.  Keeping these builders pure makes parity testable without
requiring detector execution or filesystem fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .calibration_intelligence import build_calibration_intelligence
from .io import write_json
from .outcome import reduce_regression_outcome, unavailable_winner_page_report
from .reports import write_rankings, write_raw_results


CANONICAL_SUMMARY_SCHEMA_VERSION = "0.8"
CANONICAL_MANIFEST_SCHEMA_VERSION = "0.2"
CANONICAL_CALIBRATION_SCHEMA_VERSION = "1.1"
CANONICAL_REPORT_OUTPUTS = (
    "RUN-INFO.json",
    "parameters.json",
    "parameter-provenance.json",
    "raw/results.csv",
    "reports/summary.json",
    "reports/winner-pages.json",
    "reports/calibration-intelligence.json",
    "reports/rankings.csv",
    "reports/top20.csv",
)


@dataclass(frozen=True)
class CanonicalOutcome:
    ordered: list[dict[str, Any]]
    ranked: list[dict[str, Any]]
    winner: dict[str, Any] | None
    baseline: dict[str, Any] | None
    historic_best: dict[str, Any] | None
    search_ranked: list[dict[str, Any]]
    measurement_state: dict[str, Any]
    winner_pages: dict[str, Any]


def canonical_outcome_summary_fields(outcome: CanonicalOutcome) -> dict[str, Any]:
    """Return every summary field derived exclusively from page evidence."""
    state = outcome.measurement_state
    return {
        "parameter_set_count": len(outcome.ordered),
        "page_evaluation_count": state["page_evaluation_count"],
        "successful_page_evaluation_count": state["successful_page_evaluation_count"],
        "fully_successful_parameter_set_count": sum(
            1
            for result in outcome.ordered
            if int(result["summary"].get("failure_count", 0) or 0) == 0
            and int(result["summary"].get("success_count", 0) or 0) > 0
        ),
        "measurement_state": state,
        "winner": outcome.winner,
        "baseline": outcome.baseline,
        "historic_best": outcome.historic_best,
        "top_parameter_sets": outcome.ranked[:5],
        "search_top_parameter_sets": outcome.search_ranked[:5],
        "winner_page_report": outcome.winner_pages,
    }


def derive_canonical_outcome(
    results: Iterable[dict[str, Any]],
    *,
    ranking_key: Callable[[dict[str, Any]], Any],
    winner_page_builder: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]],
) -> CanonicalOutcome:
    """Apply the one canonical ranking/reference/winner-page policy."""
    ordered, ranked, winner, measurement_state = reduce_regression_outcome(
        results, ranking_key=ranking_key
    )
    search_ranked = [
        result
        for result in ranked
        if result.get("requested_search_member") and not result.get("reference_roles")
    ]
    for result in ordered:
        result.pop("search_rank", None)
    for search_rank, result in enumerate(search_ranked, 1):
        result["search_rank"] = search_rank

    baseline = next(
        (
            result
            for result in ordered
            if result.get("profile") == "baseline"
            or "baseline" in (result.get("reference_roles") or [])
        ),
        None,
    )
    historic_best = next(
        (
            result
            for result in ordered
            if "historic_best" in (result.get("reference_roles") or [])
        ),
        None,
    )
    winner_pages = (
        winner_page_builder(winner, baseline)
        if winner is not None
        else unavailable_winner_page_report(measurement_state)
    )
    return CanonicalOutcome(
        ordered=ordered,
        ranked=ranked,
        winner=winner,
        baseline=baseline,
        historic_best=historic_best,
        search_ranked=search_ranked,
        measurement_state=measurement_state,
        winner_pages=winner_pages,
    )


def build_canonical_summary(
    outcome: CanonicalOutcome,
    *,
    run_id: str,
    detector: str,
    strategy: str,
    requested_strategy: str,
    strategy_fallback_reason: str | None,
    threads: int,
    shard: Mapping[str, Any],
    detector_pipeline: Mapping[str, Any] | None,
    parameter_space: Mapping[str, Any],
    page_ordinals: Iterable[int],
    golden_set_sha256: str | None,
    detector_config_sha256: str | None,
    model_selection: Mapping[str, Any] | None,
    max_dimension: int | None,
    runner: Mapping[str, Any],
    source_commit: str | None,
    progress: Mapping[str, Any],
    performance: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared summary schema for every execution shape."""
    summary: dict[str, Any] = {
        "schema_version": CANONICAL_SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "detector": detector,
        "strategy": strategy,
        "requested_strategy": requested_strategy,
        "strategy_fallback_reason": strategy_fallback_reason,
        "threads": int(threads),
        "shard": dict(shard),
        "detector_pipeline": dict(detector_pipeline) if detector_pipeline else None,
        "parameter_space": dict(parameter_space),
        "page_ordinals": [int(value) for value in page_ordinals],
        "golden_set_sha256": golden_set_sha256,
        "detector_config_sha256": detector_config_sha256,
        "model_selection": dict(model_selection) if model_selection else None,
        "max_dimension": max_dimension,
        "runner": dict(runner),
        "source_commit": source_commit,
        "performance": dict(performance) if performance else None,
        "progress": dict(progress),
    }
    summary.update(canonical_outcome_summary_fields(outcome))
    if extra:
        overlap = set(summary).intersection(extra)
        if overlap:
            raise ValueError(
                "Canonical summary fields cannot be overridden: " + ", ".join(sorted(overlap))
            )
        summary.update(extra)
    return summary


def build_calibration_identity(
    *,
    run_id: str,
    created_at_utc: str,
    source_document: Any,
    golden_set: Mapping[str, Any],
    detector: str,
    detector_configuration: str,
    detector_config_sha256: str,
    model_selection: Mapping[str, Any] | None,
    pipeline_commit: Any,
    source_commit: Any,
    python_version: Any,
    opencv_version: Any,
) -> dict[str, Any]:
    return {
        "calibration_run_id": run_id,
        "calibration_schema_version": CANONICAL_CALIBRATION_SCHEMA_VERSION,
        "created_at_utc": created_at_utc,
        "source_document": source_document,
        "golden_set": dict(golden_set),
        "detector_configuration": {
            "detector_id": detector,
            "configuration": detector_configuration,
            "sha256": detector_config_sha256,
        },
        "model_selection": dict(model_selection) if model_selection else None,
        "pipeline": {
            "commit": pipeline_commit,
            "source_commit": source_commit,
            "python": python_version,
            "opencv": opencv_version,
        },
    }


def build_regression_metadata(
    *,
    requested_strategy: str,
    resolved_strategy: str,
    strategy_fallback_reason: str | None,
    configured_threads: int,
    detector_pipeline: Mapping[str, Any] | None,
    possible_parameter_sets: int,
    planned_parameter_sets: int | None,
    evaluated_parameter_sets: int,
    golden_set_pages: int,
    page_evaluations: int,
    failed_page_evaluations: int,
    average_eval_rate: float | None,
    execution_environment: Mapping[str, Any],
    baseline_parameters: Mapping[str, Any],
    live_possible_parameter_sets: int,
    zombie_possible_parameter_sets: int,
    canonical_search_space: Mapping[str, Any],
    zombie_parameters: Iterable[str],
    zombie_parameter_evidence: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "requested_strategy": requested_strategy,
        "resolved_strategy": resolved_strategy,
        "strategy_fallback_reason": strategy_fallback_reason,
        "configured_threads": int(configured_threads),
        "detector_pipeline": dict(detector_pipeline) if detector_pipeline else None,
        "possible_parameter_sets": int(possible_parameter_sets),
        "planned_parameter_sets": planned_parameter_sets,
        "evaluated_parameter_sets": int(evaluated_parameter_sets),
        "golden_set_pages": int(golden_set_pages),
        "page_evaluations": int(page_evaluations),
        "failed_page_evaluations": int(failed_page_evaluations),
        "average_eval_rate": average_eval_rate,
        "execution_environment": dict(execution_environment),
        "baseline_parameters": dict(baseline_parameters),
        "fixed_parameter_policy": "baseline",
        "zombie_parameters": sorted(str(name) for name in zombie_parameters),
        "zombie_parameter_evidence": dict(zombie_parameter_evidence),
        "live_possible_parameter_sets": int(live_possible_parameter_sets),
        "zombie_possible_parameter_sets": int(zombie_possible_parameter_sets),
        "canonical_search_space": dict(canonical_search_space),
    }
    if extra:
        overlap = set(metadata).intersection(extra)
        if overlap:
            raise ValueError(
                "Canonical regression metadata fields cannot be overridden: "
                + ", ".join(sorted(overlap))
            )
        metadata.update(extra)
    return metadata


def build_canonical_calibration(
    outcome: CanonicalOutcome,
    *,
    detector: str,
    strategy: str,
    possible_parameter_sets: int,
    calibration_identity: Mapping[str, Any],
    regression_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return build_calibration_intelligence(
        outcome.ordered,
        detector=detector,
        strategy=strategy,
        possible_parameter_sets=possible_parameter_sets,
        calibration_context=dict(calibration_identity),
        regression_context=dict(regression_metadata),
    )


def build_canonical_manifest(
    outcome: CanonicalOutcome,
    *,
    run_id: str,
    detector: str,
    strategy: str,
    started_at_utc: str,
    finished_at_utc: str,
    shard: Mapping[str, Any] | None = None,
    additional_outputs: Iterable[str] = (),
    debug_outputs: Iterable[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": CANONICAL_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "detector": detector,
        "strategy": strategy,
        "status": "complete" if outcome.measurement_state["terminal_success"] else "invalid",
        "outcome": outcome.measurement_state,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "outputs": list(dict.fromkeys((*CANONICAL_REPORT_OUTPUTS, *additional_outputs))),
        "debug_outputs": list(debug_outputs),
    }
    if shard is not None:
        manifest["shard"] = dict(shard)
    if extra:
        overlap = set(manifest).intersection(extra)
        if overlap:
            raise ValueError(
                "Canonical manifest fields cannot be overridden: " + ", ".join(sorted(overlap))
            )
        manifest.update(extra)
    return manifest


def write_canonical_reports(
    run_dir: Path,
    outcome: CanonicalOutcome,
    *,
    summary: Mapping[str, Any],
    calibration: Mapping[str, Any],
    top: int,
    write_raw: bool = True,
) -> None:
    """Write every report derived solely from normalized regression evidence."""
    if write_raw:
        write_raw_results(run_dir / "raw" / "results.csv", outcome.ordered)
    write_rankings(run_dir / "reports" / "rankings.csv", outcome.ranked)
    write_rankings(run_dir / "reports" / "top20.csv", outcome.ranked[:max(0, top)])
    write_json(run_dir / "reports" / "summary.json", dict(summary))
    write_json(run_dir / "reports" / "winner-pages.json", outcome.winner_pages)
    write_json(run_dir / "reports" / "calibration-intelligence.json", dict(calibration))
