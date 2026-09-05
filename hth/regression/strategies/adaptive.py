"""Deterministic, detector-agnostic adaptive parameter-space search."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable

from ..parameter_space import canonical_parameters

Result = dict[str, Any]


@dataclass(frozen=True)
class AdaptiveSearchOutcome:
    results: list[Result]
    telemetry: dict[str, Any]


def _score(result: Result) -> float:
    return float((result.get("summary") or {}).get("mean_iou") or 0.0)


def _coordinates(parameters: dict[str, Any], domains: dict[str, list[Any]]) -> tuple[float, ...]:
    coordinates: list[float] = []
    for name, values in domains.items():
        try:
            index = values.index(parameters[name])
        except ValueError:
            index = min(range(len(values)), key=lambda item: abs(float(values[item]) - float(parameters[name])))
        coordinates.append(index / max(1, len(values) - 1))
    return tuple(coordinates)


def _distance(left: tuple[float, ...], right: tuple[float, ...], weights: list[float]) -> float:
    return math.sqrt(sum(weight * (a - b) ** 2 for a, b, weight in zip(left, right, weights)))


def _eta_squared(results: Iterable[Result], name: str) -> float:
    rows = list(results)
    if len(rows) < 2:
        return 0.0
    scores = [_score(row) for row in rows]
    overall = sum(scores) / len(scores)
    total = sum((value - overall) ** 2 for value in scores)
    if total <= 1e-15:
        return 0.0
    groups: dict[str, list[float]] = {}
    for row, value in zip(rows, scores):
        key = repr((row.get("parameters") or {}).get(name))
        groups.setdefault(key, []).append(value)
    between = sum(len(values) * ((sum(values) / len(values)) - overall) ** 2 for values in groups.values())
    return max(0.0, min(1.0, between / total))


def _initial_design(
    candidates: list[dict[str, Any]],
    coordinates: dict[str, tuple[float, ...]],
    count: int,
) -> list[dict[str, Any]]:
    if count >= len(candidates):
        return list(candidates)
    ordered = sorted(candidates, key=canonical_parameters)
    selected = [ordered[0]]
    if count > 1:
        selected.append(ordered[-1])
    selected_keys = {canonical_parameters(row) for row in selected}
    while len(selected) < count:
        choice = max(
            (row for row in ordered if canonical_parameters(row) not in selected_keys),
            key=lambda row: (
                min(
                    _distance(coordinates[canonical_parameters(row)], coordinates[canonical_parameters(existing)], [1.0] * len(coordinates[canonical_parameters(row)]))
                    for existing in selected
                ),
                canonical_parameters(row),
            ),
        )
        selected.append(choice)
        selected_keys.add(canonical_parameters(choice))
    return selected


def search(
    config: dict[str, Any],
    candidates: list[dict[str, Any]],
    evaluate_batch: Callable[[list[dict[str, Any]]], list[Result]],
    ranking_key: Callable[[Result], tuple],
    *,
    seed_results: Iterable[Result] = (),
    budget: int | None = None,
) -> AdaptiveSearchOutcome:
    """Adaptively choose candidate batches using measured quality telemetry.

    The first batch is a deterministic space-filling design.  Later batches
    balance coverage, incumbent-local refinement, marginal eta-squared influence,
    unseen values/pairs, and boundary pressure.  The search never treats prior
    evidence as an implicit bound and never invents values outside the declared
    adaptive candidate universe.
    """
    settings = dict(config.get("adaptive_search") or {})
    unique = {canonical_parameters(row): dict(row) for row in candidates}
    ordered = [unique[key] for key in sorted(unique)]
    maximum = int(settings.get("max_parameter_sets", min(len(ordered), 64)))
    target = min(len(ordered), maximum, int(budget) if budget is not None else maximum)
    if target <= 0:
        return AdaptiveSearchOutcome([], {"candidate_parameter_sets": len(ordered), "budget": 0, "rounds": []})
    names = list(config.get("parameters", {}))
    domains = {
        name: list(config["parameters"][name].get("adaptive_values", config["parameters"][name].get("values", [])))
        for name in names
    }
    coordinates = {canonical_parameters(row): _coordinates(row, domains) for row in ordered}
    initial_count = min(target, int(settings.get("initial_parameter_sets", max(6, 2 * len(names) + 1))))
    batch_size = max(1, int(settings.get("batch_size", 4)))
    pending = _initial_design(ordered, coordinates, initial_count)
    evaluated: list[Result] = []
    learning: list[Result] = [dict(row) for row in seed_results]
    evaluated_keys = {canonical_parameters(row.get("parameters") or {}) for row in learning}
    rounds: list[dict[str, Any]] = []

    while pending and len(evaluated) < target:
        pending = [row for row in pending if canonical_parameters(row) not in evaluated_keys]
        pending = pending[: target - len(evaluated)]
        if not pending:
            break
        new_results = evaluate_batch(pending)
        evaluated.extend(new_results)
        learning.extend(new_results)
        evaluated_keys.update(canonical_parameters(row.get("parameters") or {}) for row in new_results)
        eta = {name: _eta_squared(learning, name) for name in names}
        rounds.append({
            "round": len(rounds) + 1,
            "evaluated_parameter_sets": len(new_results),
            "cumulative_parameter_sets": len(evaluated),
            "best_mean_iou": max((_score(row) for row in learning), default=None),
            "eta_squared": eta,
        })
        remaining = [row for row in ordered if canonical_parameters(row) not in evaluated_keys]
        if not remaining or len(evaluated) >= target:
            break
        ranked = sorted(learning, key=ranking_key)
        elite = ranked[: max(1, min(3, len(ranked)))]
        max_eta = max(eta.values(), default=0.0)
        weights = [0.15 + (eta[name] / max_eta if max_eta > 0 else 1.0) for name in names]
        observed_coordinates = [coordinates[key] for key in evaluated_keys if key in coordinates]
        elite_coordinates = [
            _coordinates(dict(row.get("parameters") or {}), domains)
            for row in elite
            if all(name in (row.get("parameters") or {}) for name in names)
        ]
        incumbent = elite[0].get("parameters") or {}
        boundary_targets: dict[str, float] = {}
        for index, name in enumerate(names):
            if name not in incumbent or eta[name] <= 0:
                continue
            coordinate = _coordinates({name: incumbent[name]}, {name: domains[name]})[0]
            if coordinate in {0.0, 1.0}:
                boundary_targets[name] = coordinate
        observed_values = {
            name: {(row.get("parameters") or {}).get(name) for row in learning}
            for name in names
        }
        observed_pairs = {
            (left, right): {
                ((row.get("parameters") or {}).get(left), (row.get("parameters") or {}).get(right))
                for row in learning
            }
            for index, left in enumerate(names)
            for right in names[index + 1:]
        }

        def acquisition(row: dict[str, Any]) -> tuple[float, str]:
            point = coordinates[canonical_parameters(row)]
            exploration = min((_distance(point, other, weights) for other in observed_coordinates), default=1.0)
            exploitation = 1.0 - min((_distance(point, other, weights) for other in elite_coordinates), default=1.0) / max(1.0, math.sqrt(sum(weights)))
            unseen_values = sum(row[name] not in observed_values[name] for name in names) / max(1, len(names))
            pair_total = max(1, len(observed_pairs))
            unseen_pairs = sum((row[left], row[right]) not in pairs for (left, right), pairs in observed_pairs.items()) / pair_total
            boundary_pressure = (
                sum(
                    (eta[name] / max_eta) * (1.0 - abs(point[names.index(name)] - target))
                    for name, target in boundary_targets.items()
                ) / max(1, len(boundary_targets))
                if max_eta > 0 else 0.0
            )
            return (
                0.34 * exploration
                + 0.30 * exploitation
                + 0.13 * unseen_values
                + 0.13 * unseen_pairs
                + 0.10 * boundary_pressure,
                canonical_parameters(row),
            )

        pending = sorted(remaining, key=acquisition, reverse=True)[: min(batch_size, target - len(evaluated))]

    final_eta = {name: _eta_squared(learning, name) for name in names}
    telemetry = {
        "schema_version": "1.0",
        "strategy": "adaptive",
        "candidate_parameter_sets": len(ordered),
        "budget": target,
        "evaluated_parameter_sets": len(evaluated),
        "initial_parameter_sets": initial_count,
        "batch_size": batch_size,
        "eta_squared": final_eta,
        "rounds": rounds,
        "selection_policy": "space-filling then eta-weighted coverage/incumbent/boundary refinement",
        "deterministic": True,
    }
    return AdaptiveSearchOutcome(evaluated, telemetry)
