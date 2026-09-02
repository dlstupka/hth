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
from .reports import write_rankings, write_raw_evidence, write_raw_results
from .run_semantics import validate_run_semantics


CANONICAL_SUMMARY_SCHEMA_VERSION = "0.9"
CANONICAL_MANIFEST_SCHEMA_VERSION = "0.3"
CANONICAL_CALIBRATION_SCHEMA_VERSION = "1.1"
CANONICAL_REPORT_OUTPUTS = (
    "RUN-INFO.json",
    "parameters.json",
    "parameter-provenance.json",
    "raw/results.csv",
    "raw/evidence.jsonl",
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


def _merge_extra_fields(
    target: dict[str, Any],
    extra: Mapping[str, Any] | None,
    *,
    contract: str,
) -> None:
    """Add extension fields without allowing canonical contract overrides."""
    if not extra:
        return
    overlap = set(target).intersection(extra)
    if overlap:
        raise ValueError(
            f"Canonical {contract} fields cannot be overridden: "
            + ", ".join(sorted(overlap))
        )
    target.update(extra)


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
    run_mode: str,
    evidence_tier: str,
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
    run_mode, evidence_tier = validate_run_semantics(run_mode, evidence_tier)
    summary: dict[str, Any] = {
        "schema_version": CANONICAL_SUMMARY_SCHEMA_VERSION,
        "run_id": run_id,
        "detector": detector,
        "run_mode": run_mode,
        "evidence_tier": evidence_tier,
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
    _merge_extra_fields(summary, extra, contract="summary")
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
    run_mode: str,
    evidence_tier: str,
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
    run_mode, evidence_tier = validate_run_semantics(run_mode, evidence_tier)
    metadata: dict[str, Any] = {
        "run_mode": run_mode,
        "evidence_tier": evidence_tier,
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
    _merge_extra_fields(metadata, extra, contract="regression metadata")
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
    run_mode, evidence_tier = validate_run_semantics(
        regression_metadata.get("run_mode"),
        regression_metadata.get("evidence_tier"),
    )
    calibration = build_calibration_intelligence(
        outcome.ordered,
        detector=detector,
        strategy=strategy,
        possible_parameter_sets=possible_parameter_sets,
        calibration_context=dict(calibration_identity),
        regression_context=dict(regression_metadata),
    )
    calibration["run_mode"] = run_mode
    calibration["evidence_tier"] = evidence_tier
    return calibration


def build_golden_set_identity(
    *,
    configuration: Any,
    sha256: Any,
    payload: Mapping[str, Any] | None,
    page_ordinals: Iterable[int],
) -> dict[str, Any]:
    """Describe the Golden Set once for every calibration-producing path."""
    golden_set = dict(payload or {})
    ordinals = [int(value) for value in page_ordinals]
    return {
        "configuration": str(configuration) if configuration is not None else None,
        "sha256": sha256,
        "collection_id": golden_set.get("collection_id"),
        "schema_version": golden_set.get("schema_version"),
        "description": golden_set.get("description"),
        "page_count": len(ordinals),
        "page_ordinals": ordinals,
    }


def build_canonical_calibration_from_summary(
    outcome: CanonicalOutcome,
    *,
    summary: Mapping[str, Any],
    created_at_utc: str,
    golden_set_configuration: Any,
    golden_set_payload: Mapping[str, Any] | None,
    detector_configuration: Any,
    detector_config: Mapping[str, Any],
    regression_metadata_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive calibration identity and metadata from one canonical summary."""
    detector = str(summary["detector"])
    strategy = str(summary["strategy"])
    runner = dict(summary.get("runner") or {})
    parameter_space = dict(summary.get("parameter_space") or {})
    progress = dict(summary.get("progress") or {})
    page_ordinals = list(summary.get("page_ordinals") or [])
    golden_set_payload = dict(golden_set_payload or {})
    calibration_identity = build_calibration_identity(
        run_id=str(summary["run_id"]),
        created_at_utc=created_at_utc,
        source_document=golden_set_payload.get("source_document"),
        golden_set=build_golden_set_identity(
            configuration=golden_set_configuration,
            sha256=summary.get("golden_set_sha256"),
            payload=golden_set_payload,
            page_ordinals=page_ordinals,
        ),
        detector=detector,
        detector_configuration=str(detector_configuration),
        detector_config_sha256=str(summary.get("detector_config_sha256") or ""),
        model_selection=summary.get("model_selection"),
        pipeline_commit=runner.get("pipeline_commit"),
        source_commit=summary.get("source_commit"),
        python_version=runner.get("python_version"),
        opencv_version=runner.get("opencv_version"),
    )
    search_space = dict(parameter_space.get("canonical_search_space") or {})
    zombie_specs = (
        detector_config.get("zombie_parameters", {})
        if isinstance(detector_config.get("zombie_parameters"), dict)
        else {}
    )
    profiles = detector_config.get("profiles", {})
    baseline_parameters = profiles.get("baseline", {}) if isinstance(profiles, dict) else {}
    regression_metadata = build_regression_metadata(
        run_mode=str(summary["run_mode"]),
        evidence_tier=str(summary["evidence_tier"]),
        requested_strategy=str(summary.get("requested_strategy") or strategy),
        resolved_strategy=strategy,
        strategy_fallback_reason=summary.get("strategy_fallback_reason"),
        configured_threads=int(summary.get("threads") or 1),
        detector_pipeline=summary.get("detector_pipeline"),
        possible_parameter_sets=int(parameter_space.get("possible_parameter_sets") or 0),
        planned_parameter_sets=parameter_space.get("planned_parameter_sets"),
        evaluated_parameter_sets=int(summary.get("parameter_set_count") or 0),
        golden_set_pages=len(page_ordinals),
        page_evaluations=int(
            parameter_space.get("actual_page_evaluations")
            or summary.get("page_evaluation_count")
            or 0
        ),
        failed_page_evaluations=int(progress.get("failures") or 0),
        average_eval_rate=progress.get("average_eval_rate"),
        execution_environment=runner,
        baseline_parameters=baseline_parameters,
        live_possible_parameter_sets=int(parameter_space.get("live_possible_parameter_sets") or 0),
        zombie_possible_parameter_sets=int(parameter_space.get("zombie_possible_parameter_sets") or 0),
        canonical_search_space=search_space,
        zombie_parameters=search_space.get("configured_zombie_parameters", zombie_specs),
        zombie_parameter_evidence={
            str(parameter_name): dict(spec.get("last_measured", {}))
            for parameter_name, spec in zombie_specs.items()
            if isinstance(spec, dict) and isinstance(spec.get("last_measured"), dict)
        },
        extra=regression_metadata_extra,
    )
    return build_canonical_calibration(
        outcome,
        detector=detector,
        strategy=strategy,
        possible_parameter_sets=int(parameter_space.get("possible_parameter_sets") or 0),
        calibration_identity=calibration_identity,
        regression_metadata=regression_metadata,
    )


def build_canonical_manifest(
    outcome: CanonicalOutcome,
    *,
    run_id: str,
    detector: str,
    run_mode: str,
    evidence_tier: str,
    strategy: str,
    requested_strategy: str | None = None,
    strategy_fallback_reason: str | None = None,
    started_at_utc: str,
    finished_at_utc: str,
    shard: Mapping[str, Any] | None = None,
    additional_outputs: Iterable[str] = (),
    debug_outputs: Iterable[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    run_mode, evidence_tier = validate_run_semantics(run_mode, evidence_tier)
    manifest: dict[str, Any] = {
        "schema_version": CANONICAL_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "detector": detector,
        "run_mode": run_mode,
        "evidence_tier": evidence_tier,
        "strategy": strategy,
        "requested_strategy": requested_strategy or strategy,
        "strategy_fallback_reason": strategy_fallback_reason,
        "status": "complete" if outcome.measurement_state["terminal_success"] else "invalid",
        "outcome": outcome.measurement_state,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "outputs": list(dict.fromkeys((*CANONICAL_REPORT_OUTPUTS, *additional_outputs))),
        "debug_outputs": list(debug_outputs),
    }
    if shard is not None:
        manifest["shard"] = dict(shard)
    _merge_extra_fields(manifest, extra, contract="manifest")
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
        write_raw_evidence(run_dir / "raw" / "evidence.jsonl", outcome.ordered)
    write_rankings(run_dir / "reports" / "rankings.csv", outcome.ranked)
    write_rankings(run_dir / "reports" / "top20.csv", outcome.ranked[:max(0, top)])
    write_json(run_dir / "reports" / "summary.json", dict(summary))
    write_json(run_dir / "reports" / "winner-pages.json", outcome.winner_pages)
    write_json(run_dir / "reports" / "calibration-intelligence.json", dict(calibration))
