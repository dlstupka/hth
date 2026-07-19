"""Detector-agnostic interval refinement followed by local Cartesian search."""
from __future__ import annotations
import itertools
from typing import Any, Callable
from ..parameter_space import canonical_parameters, value_index

Result = dict[str, Any]

def search(config: dict[str, Any], evaluate: Callable[[dict[str, Any]], Result], ranking_key: Callable[[Result], tuple]) -> list[Result]:
    specs = config["parameters"]
    current = dict(config["profiles"]["baseline"])
    evaluated: dict[str, Result] = {}
    def cached(params: dict[str, Any]) -> Result:
        key = canonical_parameters(params)
        if key not in evaluated:
            evaluated[key] = evaluate(dict(params))
        return evaluated[key]
    cached(current)
    for _ in range(int(config.get("binary_refine", {}).get("passes", 3))):
        changed = False
        for name, spec in specs.items():
            values = list(spec["values"])
            if len(values) < 2: continue
            low, high = 0, len(values)-1
            best_params, best_result = dict(current), cached(current)
            while low <= high:
                mid = (low+high)//2
                trials=[]
                for idx in sorted({low, mid, high}):
                    trial=dict(current); trial[name]=values[idx]
                    trials.append((idx, trial, cached(trial)))
                idx, trial, result = min(trials, key=lambda item: ranking_key(item[2]))
                if ranking_key(result) < ranking_key(best_result):
                    best_params, best_result = trial, result
                if high-low <= 2: break
                if idx <= mid: high = mid
                else: low = mid
            if canonical_parameters(best_params) != canonical_parameters(current):
                current=best_params; changed=True
        if not changed: break
    radius=int(config.get("binary_refine", {}).get("local_exhaustive_radius", 1))
    names=list(specs); local=[]
    for name in names:
        values=list(specs[name]["values"]); center=value_index(values,current[name])
        local.append(values[max(0,center-radius):min(len(values),center+radius+1)])
    for combo in itertools.product(*local):
        cached(dict(zip(names,combo,strict=True)))
    return list(evaluated.values())
