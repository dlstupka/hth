"""Deterministic, detector-agnostic adaptive parameter-space search."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Callable, Iterable

from ..parameter_space import canonical_parameters

Result = dict[str, Any]


@dataclass(frozen=True)
class AdaptiveSearchOutcome:
    results: list[Result]
    telemetry: dict[str, Any]


def default_parameter_budget(config: dict[str, Any], candidate_count: int) -> int:
    """Return the shared dimension-aware adaptive budget for a detector grid."""
    settings = dict(config.get("adaptive_search") or {})
    varying_numeric_dimensions = sum(
        str(spec.get("type")) in {"float", "int"}
        and len(set(spec.get("adaptive_values", spec.get("values", [])))) > 1
        for spec in (config.get("parameters") or {}).values()
        if isinstance(spec, dict)
    )
    default_maximum = max(64, 24 * varying_numeric_dimensions)
    return min(max(0, int(candidate_count)), int(settings.get("max_parameter_sets", default_maximum)))


def _score(result: Result) -> float:
    return float((result.get("summary") or {}).get("mean_iou") or 0.0)


def _coordinates(parameters: dict[str, Any], domains: dict[str, list[Any]]) -> tuple[float, ...]:
    coordinates: list[float] = []
    for name, values in domains.items():
        if values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            low, high = float(min(values)), float(max(values))
            coordinates.append(0.0 if high == low else (float(parameters[name]) - low) / (high - low))
            continue
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


def _dynamic_refinement_candidates(
    config: dict[str, Any],
    incumbents: list[dict[str, Any]],
    eta: dict[str, float],
    refinement_domains: dict[str, list[Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[Any]]]:
    """Generate bounded local and uncertainty midpoints around several elites."""
    settings = dict((config.get("adaptive_search") or {}).get("dynamic_refinement") or {})
    if not settings.get("enabled", True):
        return [], {}
    if not incumbents:
        return [], {}
    minimum_eta = float(settings.get("minimum_eta_squared", 0.0))
    maximum_dimensions = max(1, int(settings.get("maximum_dimensions", 3)))
    numeric_names = [
        name for name in refinement_domains
        if str(config.get("parameters", {}).get(name, {}).get("type")) in {"float", "int"}
        and len(set(refinement_domains[name])) > 1
    ]
    names = [
        name for name, _ in sorted(eta.items(), key=lambda item: (-item[1], item[0]))
        if eta[name] >= minimum_eta
        and name in numeric_names
    ][:maximum_dimensions]
    # At the beginning of a search marginal eta-squared may be zero or unstable.
    # Preserve global numeric coverage instead of disabling refinement precisely
    # when the current samples contain the least information.
    if not names:
        names = numeric_names[:maximum_dimensions]
    generated_values: dict[str, list[Any]] = {}
    for name in names:
        values = sorted({float(value) for value in refinement_domains[name]})
        midpoints: list[Any] = []
        centers = [float(row[name]) for row in incumbents if name in row]
        for center in centers:
            lower = max((value for value in values if value < center), default=None)
            upper = min((value for value in values if value > center), default=None)
            for neighbor in (lower, upper):
                if neighbor is not None:
                    midpoints.append(round((center + neighbor) / 2.0, 12))
        # Independently subdivide the widest declared interval. This maintains
        # exploration even when every current elite lies in the wrong basin.
        if len(values) > 1:
            left, right = max(zip(values, values[1:]), key=lambda pair: (pair[1] - pair[0], -pair[0]))
            midpoints.append(round((left + right) / 2.0, 12))
        if str(config["parameters"][name].get("type")) == "int":
            midpoints = [int(round(value)) for value in midpoints]
        midpoints = [value for value in midpoints if value not in refinement_domains[name]]
        if midpoints:
            generated_values[name] = sorted(set(midpoints))
            refinement_domains[name] = sorted(set(refinement_domains[name]) | set(midpoints))

    candidates: list[dict[str, Any]] = []
    for incumbent in incumbents:
        for name, values in generated_values.items():
            for value in values:
                candidate = dict(incumbent)
                candidate[name] = value
                candidates.append(candidate)
        interacting = list(generated_values)
        if len(interacting) > 1:
            # Pairwise combinations expose narrow interactions without creating
            # the full refined Cartesian product.
            for left_index, left in enumerate(interacting):
                for right in interacting[left_index + 1:]:
                    for left_value, right_value in itertools.product(generated_values[left], generated_values[right]):
                        candidate = dict(incumbent)
                        candidate[left] = left_value
                        candidate[right] = right_value
                        candidates.append(candidate)
    unique = {canonical_parameters(candidate): candidate for candidate in candidates}
    return [unique[key] for key in sorted(unique)], generated_values


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
    balance coverage, multi-elite refinement, marginal eta-squared influence,
    unseen values/pairs, and boundary pressure. The search never treats prior
    evidence as an implicit bound. Dynamic refinement adds numeric midpoint
    candidates inside the declared adaptive bounds while preserving exploration.
    """
    settings = dict(config.get("adaptive_search") or {})
    unique = {canonical_parameters(row): dict(row) for row in candidates}
    ordered = [unique[key] for key in sorted(unique)]
    names = list(config.get("parameters", {}))
    domains = {
        name: list(config["parameters"][name].get("adaptive_values", config["parameters"][name].get("values", [])))
        for name in names
    }
    maximum = default_parameter_budget(config, len(ordered))
    target = min(len(ordered), maximum, int(budget) if budget is not None else maximum)
    if target <= 0:
        return AdaptiveSearchOutcome([], {"candidate_parameter_sets": len(ordered), "budget": 0, "rounds": []})
    refinement_domains = {name: list(values) for name, values in domains.items()}
    coordinates = {canonical_parameters(row): _coordinates(row, domains) for row in ordered}
    initial_count = min(target, int(settings.get("initial_parameter_sets", max(9, 4 * len(names) + 1))))
    batch_size = max(1, int(settings.get("batch_size", 4)))
    pending = _initial_design(ordered, coordinates, initial_count)
    evaluated: list[Result] = []
    learning: list[Result] = [dict(row) for row in seed_results]
    evaluated_keys = {canonical_parameters(row.get("parameters") or {}) for row in learning}
    rounds: list[dict[str, Any]] = []
    generated_keys: set[str] = set()
    generated_values: dict[str, set[Any]] = {name: set() for name in names}

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
        round_record = {
            "round": len(rounds) + 1,
            "evaluated_parameter_sets": len(new_results),
            "cumulative_parameter_sets": len(evaluated),
            "best_mean_iou": max((_score(row) for row in learning), default=None),
            "eta_squared": eta,
        }
        rounds.append(round_record)
        if len(evaluated) >= target:
            break
        ranked = sorted(learning, key=ranking_key)
        elite = ranked[: max(1, min(3, len(ranked)))]
        incumbents = [dict(row.get("parameters") or {}) for row in elite]
        incumbent = incumbents[0]
        refined, new_values = _dynamic_refinement_candidates(
            config, incumbents, eta, refinement_domains,
        )
        added = []
        for row in refined:
            key = canonical_parameters(row)
            if key not in unique and key not in evaluated_keys:
                unique[key] = row
                ordered.append(row)
                coordinates[key] = _coordinates(row, domains)
                generated_keys.add(key)
                added.append(row)
        for name, values in new_values.items():
            generated_values[name].update(values)
        round_record["generated_refinement_candidates"] = len(added)
        round_record["generated_values"] = new_values
        remaining = [row for row in ordered if canonical_parameters(row) not in evaluated_keys]
        if not remaining:
            break
        max_eta = max(eta.values(), default=0.0)
        weights = [0.15 + (eta[name] / max_eta if max_eta > 0 else 1.0) for name in names]
        observed_coordinates = [coordinates[key] for key in evaluated_keys if key in coordinates]
        elite_coordinates = [
            _coordinates(dict(row.get("parameters") or {}), domains)
            for row in elite
            if all(name in (row.get("parameters") or {}) for name in names)
        ]
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

        next_count = min(batch_size, target - len(evaluated))
        refinement_share = max(0.0, min(0.75, float(settings.get("dynamic_refinement", {}).get("batch_share", 0.5))))
        refinement_count = min(len(generated_keys), math.ceil(next_count * refinement_share))
        generated_remaining = [row for row in remaining if canonical_parameters(row) in generated_keys]
        base_remaining = [row for row in remaining if canonical_parameters(row) not in generated_keys]

        def refinement_acquisition(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
            point = coordinates[canonical_parameters(row)]
            changed = [name for name in names if row.get(name) != incumbent.get(name)]
            changed_influence = sum(eta.get(name, 0.0) for name in changed)
            incumbent_distance = (
                _distance(point, elite_coordinates[0], [1.0] * len(weights))
                if elite_coordinates else math.inf
            )
            adaptive_score, canonical = acquisition(row)
            # Prefer one-axis tests on the current winner, prioritizing the
            # dimensions with measured influence. This obtains the evidence
            # needed before spending refinement budget on interaction products.
            return (-len(changed), changed_influence, -incumbent_distance, adaptive_score, canonical)

        selected = sorted(generated_remaining, key=refinement_acquisition, reverse=True)[:refinement_count]
        selected_keys = {canonical_parameters(row) for row in selected}
        selected.extend(
            row for row in sorted(base_remaining + generated_remaining, key=acquisition, reverse=True)
            if canonical_parameters(row) not in selected_keys
        )
        pending = selected[:next_count]

    final_eta = {name: _eta_squared(learning, name) for name in names}
    telemetry = {
        "schema_version": "1.0",
        "strategy": "adaptive",
        "candidate_parameter_sets": len(ordered),
        "initial_candidate_parameter_sets": len(candidates),
        "generated_refinement_parameter_sets": len(generated_keys),
        "generated_values": {name: sorted(values) for name, values in generated_values.items() if values},
        "budget": target,
        "evaluated_parameter_sets": len(evaluated),
        "initial_parameter_sets": initial_count,
        "batch_size": batch_size,
        "eta_squared": final_eta,
        "rounds": rounds,
        "selection_policy": "space-filling then eta-weighted coverage/multi-elite refinement with global interval subdivision and bounded dynamic midpoints",
        "deterministic": True,
    }
    return AdaptiveSearchOutcome(evaluated, telemetry)
