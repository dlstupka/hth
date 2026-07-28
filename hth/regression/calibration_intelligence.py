"""Derive calibration intelligence from complete detector regression results."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from typing import Any, Iterable


NEAR_BEST_ABSOLUTE_TOLERANCE = 0.001
EQUIVALENT_ABSOLUTE_TOLERANCE = 0.0001
INTERACTION_SAMPLE_LIMIT = 50_000
INTERACTION_PARAMETER_LIMIT = 6


def _value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _display_value(key: str) -> str:
    try:
        value = json.loads(key)
    except json.JSONDecodeError:
        return key
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _quantile_desc(values: list[float], quantile: float) -> float | None:
    """Return a quantile from values already ordered from highest to lowest."""
    if not values:
        return None
    ascending_position = quantile * (len(values) - 1)
    descending_position = (len(values) - 1) - ascending_position
    lower = int(math.floor(descending_position))
    upper = int(math.ceil(descending_position))
    if lower == upper:
        return values[lower]
    fraction = descending_position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _eta_squared(groups: dict[str, list[float]], overall_mean: float, total_ss: float) -> float:
    if total_ss <= 0.0:
        return 0.0
    between = sum(bucket[0] * ((bucket[1] / bucket[0]) - overall_mean) ** 2 for bucket in groups.values() if bucket[0])
    return max(0.0, min(1.0, between / total_ss))


def _influence_class(eta_squared: float, mean_range: float) -> str:
    if mean_range < EQUIVALENT_ABSOLUTE_TOLERANCE or eta_squared < 0.001:
        return "Dormant"
    if eta_squared >= 0.25:
        return "Critical"
    if eta_squared >= 0.10:
        return "Important"
    if eta_squared >= 0.03:
        return "Moderate"
    return "Low"


def _confidence(*, exhaustive_complete: bool, success_rate: float, near_best_share: float, set_count: int) -> tuple[str, list[str]]:
    reasons: list[str] = []
    score = 0
    if exhaustive_complete:
        score += 2
        reasons.append("complete exhaustive coverage")
    else:
        reasons.append("partial or adaptive search")
    if success_rate >= 0.90:
        score += 1
        reasons.append("high fully-successful-set rate")
    elif success_rate < 0.50:
        reasons.append("many parameter sets failed at least one page")
    if near_best_share >= 0.01:
        score += 1
        reasons.append("broad near-best basin")
    elif set_count > 20:
        reasons.append("narrow near-best basin")
    if set_count < 20:
        reasons.append("small calibration sample")
    if score >= 4:
        return "High", reasons
    if score >= 2:
        return "Medium", reasons
    return "Low", reasons


def build_calibration_intelligence(
    ranked: list[dict[str, Any]],
    *,
    detector: str,
    strategy: str,
    possible_parameter_sets: int | None,
) -> dict[str, Any]:
    """Build a compact, machine-readable characterization of a calibration run.

    The conclusions are intentionally scoped to the evaluated Golden Set and
    parameter grid. They do not claim detector behavior outside that corpus.
    """
    if not ranked:
        return {
            "schema_version": "1.0",
            "detector": detector,
            "available": False,
            "reason": "no evaluated parameter sets",
        }

    scores = [float(result.get("summary", {}).get("mean_iou", 0.0) or 0.0) for result in ranked]
    best_score = scores[0]
    count = len(scores)
    overall_mean = sum(scores) / count
    total_ss = sum((score - overall_mean) ** 2 for score in scores)
    successful_count = sum(
        1 for result in ranked
        if int(result.get("summary", {}).get("failure_count", 0) or 0) == 0
    )
    near_best_count = sum(1 for score in scores if best_score - score <= NEAR_BEST_ABSOLUTE_TOLERANCE)
    equivalent_count = sum(1 for score in scores if best_score - score <= EQUIVALENT_ABSOLUTE_TOLERANCE)

    group_accumulators: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, -math.inf]))
    near_best_values: dict[str, set[str]] = defaultdict(set)
    page_accumulators: dict[int, dict[str, float]] = defaultdict(lambda: {
        "count": 0.0, "success": 0.0, "sum": 0.0, "sum_sq": 0.0,
        "minimum": math.inf, "maximum": -math.inf,
    })

    for result, score in zip(ranked, scores):
        parameters = result.get("parameters", {})
        if isinstance(parameters, dict):
            for name, value in parameters.items():
                key = _value_key(value)
                bucket = group_accumulators[str(name)][key]
                bucket[0] += 1.0
                bucket[1] += score
                bucket[2] += score * score
                bucket[3] = max(bucket[3], score)
                if best_score - score <= NEAR_BEST_ABSOLUTE_TOLERANCE:
                    near_best_values[str(name)].add(key)
        for page in result.get("pages", []):
            try:
                ordinal = int(page.get("global_ordinal"))
            except (TypeError, ValueError):
                continue
            accumulator = page_accumulators[ordinal]
            accumulator["count"] += 1.0
            page_iou = float(page.get("iou", 0.0) or 0.0)
            if str(page.get("status", "")) == "ok":
                accumulator["success"] += 1.0
            accumulator["sum"] += page_iou
            accumulator["sum_sq"] += page_iou * page_iou
            accumulator["minimum"] = min(accumulator["minimum"], page_iou)
            accumulator["maximum"] = max(accumulator["maximum"], page_iou)

    parameters_report: list[dict[str, Any]] = []
    for name, groups in group_accumulators.items():
        means = [(key, bucket[1] / bucket[0]) for key, bucket in groups.items() if bucket[0]]
        means.sort(key=lambda item: (-item[1], item[0]))
        mean_values = [mean for _, mean in means]
        mean_range = max(mean_values) - min(mean_values) if mean_values else 0.0
        eta = _eta_squared(groups, overall_mean, total_ss)
        parameters_report.append({
            "parameter": name,
            "value_count": len(groups),
            "eta_squared": eta,
            "mean_iou_range": mean_range,
            "classification": _influence_class(eta, mean_range),
            "near_best_value_coverage": (
                len(near_best_values.get(name, set())) / len(groups) if groups else 0.0
            ),
            "best_values": [
                {"value": _display_value(key), "mean_iou": mean, "count": int(groups[key][0])}
                for key, mean in means[:3]
            ],
        })
    parameters_report.sort(key=lambda item: (-float(item["eta_squared"]), -float(item["mean_iou_range"]), str(item["parameter"])))

    interaction_parameters = [item["parameter"] for item in parameters_report[:INTERACTION_PARAMETER_LIMIT]]
    sample_step = max(1, math.ceil(count / INTERACTION_SAMPLE_LIMIT))
    sample = ranked[::sample_step]
    sample_scores = [float(result.get("summary", {}).get("mean_iou", 0.0) or 0.0) for result in sample]
    sample_mean = sum(sample_scores) / len(sample_scores) if sample_scores else 0.0
    sample_ss = sum((score - sample_mean) ** 2 for score in sample_scores)
    sample_individual_eta: dict[str, float] = {}
    for parameter in interaction_parameters:
        groups: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for result, score in zip(sample, sample_scores):
            parameters = result.get("parameters", {})
            if not isinstance(parameters, dict) or parameter not in parameters:
                continue
            key = _value_key(parameters[parameter])
            groups[key][0] += 1.0
            groups[key][1] += score
        sample_individual_eta[parameter] = _eta_squared(groups, sample_mean, sample_ss)

    interactions: list[dict[str, Any]] = []
    for left_index, left in enumerate(interaction_parameters):
        for right in interaction_parameters[left_index + 1:]:
            groups: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
            for result, score in zip(sample, sample_scores):
                parameters = result.get("parameters", {})
                if not isinstance(parameters, dict) or left not in parameters or right not in parameters:
                    continue
                key = _value_key([parameters[left], parameters[right]])
                groups[key][0] += 1.0
                groups[key][1] += score
            pair_eta = _eta_squared(groups, sample_mean, sample_ss)
            incremental = max(0.0, pair_eta - max(sample_individual_eta.get(left, 0.0), sample_individual_eta.get(right, 0.0)))
            interactions.append({
                "parameters": [left, right],
                "eta_squared": pair_eta,
                "incremental_importance": incremental,
                "sample_size": len(sample),
            })
    interactions.sort(key=lambda item: (-float(item["incremental_importance"]), -float(item["eta_squared"]), item["parameters"]))

    page_report: list[dict[str, Any]] = []
    for ordinal, accumulator in sorted(page_accumulators.items()):
        page_count = int(accumulator["count"])
        mean = accumulator["sum"] / page_count if page_count else 0.0
        variance = max(0.0, accumulator["sum_sq"] / page_count - mean * mean) if page_count else 0.0
        page_report.append({
            "global_ordinal": ordinal,
            "mean_iou": mean,
            "minimum_iou": accumulator["minimum"] if page_count else None,
            "maximum_iou": accumulator["maximum"] if page_count else None,
            "stddev_iou": math.sqrt(variance),
            "success_rate": accumulator["success"] / page_count if page_count else 0.0,
        })

    exhaustive_complete = bool(
        strategy in {"exhaustive", "cartesian"}
        and possible_parameter_sets is not None
        and count >= int(possible_parameter_sets)
    )
    success_rate = successful_count / count
    near_best_share = near_best_count / count
    confidence, confidence_reasons = _confidence(
        exhaustive_complete=exhaustive_complete,
        success_rate=success_rate,
        near_best_share=near_best_share,
        set_count=count,
    )
    dormant = [item["parameter"] for item in parameters_report if item["classification"] == "Dormant"]

    return {
        "schema_version": "1.0",
        "detector": detector,
        "available": True,
        "scope_note": "All conclusions are specific to the evaluated Golden Set and configured parameter grid.",
        "search": {
            "strategy": strategy,
            "parameter_sets": count,
            "possible_parameter_sets": possible_parameter_sets,
            "exhaustive_complete": exhaustive_complete,
            "fully_successful_parameter_sets": successful_count,
            "fully_successful_rate": success_rate,
        },
        "landscape": {
            "best_mean_iou": best_score,
            "mean_mean_iou": overall_mean,
            "minimum_mean_iou": scores[-1],
            "stddev_mean_iou": math.sqrt(total_ss / count) if count else 0.0,
            "median_mean_iou": _quantile_desc(scores, 0.50),
            "p90_mean_iou": _quantile_desc(scores, 0.90),
            "p95_mean_iou": _quantile_desc(scores, 0.95),
            "p99_mean_iou": _quantile_desc(scores, 0.99),
            "equivalent_winner_count": equivalent_count,
            "equivalent_winner_share": equivalent_count / count,
            "near_best_count": near_best_count,
            "near_best_share": near_best_share,
            "equivalent_tolerance": EQUIVALENT_ABSOLUTE_TOLERANCE,
            "near_best_tolerance": NEAR_BEST_ABSOLUTE_TOLERANCE,
        },
        "parameter_influence": parameters_report,
        "interactions": interactions[:10],
        "interaction_method": {
            "parameters_considered": interaction_parameters,
            "sample_size": len(sample),
            "sample_step": sample_step,
            "note": "Pairwise interaction importance is estimated from a deterministic sample and is exploratory, not causal.",
        },
        "page_sensitivity": page_report,
        "recommendations": {
            "dormant_parameters": dormant,
            "retain_for_revalidation": bool(dormant),
            "note": "Dormant parameters may be omitted from future searches for this Golden Set, but should be re-evaluated when the Golden Set changes.",
        },
        "calibration_confidence": {
            "rating": confidence,
            "reasons": confidence_reasons,
        },
    }
