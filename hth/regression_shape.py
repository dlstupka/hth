#!/usr/bin/env python3
"""Resolve regression execution shapes from optimizer intelligence or manual input."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from hth.domain.multidetector_schedule import recommended_schedule, workload_class
from hth.optimizer_intelligence import (
    compatible_optimizer_rows as intelligence_compatible_optimizer_rows,
    logical_cpus_from_capacity_label,
    resolve_optimizer_intelligence,
    resolve_selector_intelligence,
    runner_from_row as intelligence_runner_from_row,
)
from hth.shape_prediction import merge_prediction
from hth.regression.sharding import runner_max_threads


@dataclass(frozen=True)
class RunnerProfile:
    name: str
    label: str
    cpu_model: str
    physical_cores: int | None
    logical_cpus: int


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return "unknown"


def _physical_cores() -> int | None:
    try:
        completed = subprocess.run(
            ["lscpu", "-p=CORE,SOCKET"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    pairs = {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip() and not line.startswith("#")
    }
    return len(pairs) or None


def current_runner_profile(*, name: str | None = None, label: str | None = None) -> RunnerProfile:
    return RunnerProfile(
        name=(name or os.environ.get("HTH_RUNNER_NAME") or os.environ.get("RUNNER_NAME") or "unknown").strip(),
        label=(label or os.environ.get("HTH_RUNNER_LABEL") or "unknown").strip(),
        cpu_model=_cpu_model(),
        physical_cores=_physical_cores(),
        logical_cpus=max(1, os.cpu_count() or 1),
    )




_RUNNER_TARGETS: dict[str, list[str]] = {
    "github-hosted": ["ubuntu-latest"],
    "hth": ["self-hosted", "Linux", "X64", "hth"],
    "rhel8": ["self-hosted", "Linux", "X64", "rhel8"],
    "e7k": ["self-hosted", "Linux", "X64", "e7k"],
    "e9k": ["self-hosted", "Linux", "X64", "e9k"],
    "windows": ["self-hosted", "Windows", "X64"],
}


def _requested_runner_target(
    *, runner: str, specific_runner: str, custom_runner_label: str | None,
) -> tuple[list[str], str]:
    if specific_runner == "custom" and str(custom_runner_label or "").strip():
        label = str(custom_runner_label).strip()
        return ["self-hosted", label], label
    mapping = {
        "github-hosted": (["ubuntu-latest"], "github-hosted"),
        "self-hosted-hth": (_RUNNER_TARGETS["hth"], "hth"),
        "self-hosted-rhel8": (_RUNNER_TARGETS["rhel8"], "rhel8"),
        "self-hosted-e7k": (_RUNNER_TARGETS["e7k"], "e7k"),
        "self-hosted-e9k": (_RUNNER_TARGETS["e9k"], "e9k"),
        "self-hosted-windows": (_RUNNER_TARGETS["windows"], "windows"),
    }
    labels, label = mapping.get(runner, mapping["github-hosted"])
    return list(labels), label


def parse_manual_shape(value: str) -> tuple[int, int]:
    text = value.strip().lower().replace(" ", "")
    patterns = (
        r"^(\d+)p/(\d+)t$",
        r"^(\d+)/(\d+)$",
        r"^(\d+)p[,x:](\d+)t?$",
    )
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            pipelines, threads = (int(match.group(1)), int(match.group(2)))
            if pipelines < 1 or threads < 1:
                break
            return pipelines, threads
    raise ValueError(f"Manual execution shape must look like 8p/48t (got {value!r})")


def _runner_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return intelligence_runner_from_row(row)


def _normalize_model(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def compatible_optimizer_rows(
    *, parallelism_index: Path, detector_config: Path, golden_set: Path, max_dimension: int,
) -> tuple[str, list[dict[str, Any]]]:
    """Backward-compatible wrapper around shared optimizer intelligence."""
    return intelligence_compatible_optimizer_rows(
        parallelism_index=parallelism_index, detector_config=detector_config,
        golden_set=golden_set, max_dimension=max_dimension,
    )

def resolve_predicted_shape(
    *,
    parallelism_index: Path,
    predictions_index: Path | None,
    detector_config: Path,
    golden_set: Path,
    max_dimension: int,
    profile: RunnerProfile,
) -> dict[str, Any] | None:
    detector, rows = compatible_optimizer_rows(
        parallelism_index=parallelism_index,
        detector_config=detector_config,
        golden_set=golden_set,
        max_dimension=max_dimension,
    )
    if not rows:
        return None
    result = resolve_optimizer_intelligence(
        detector=detector, rows=rows, target_runner_name=profile.name,
        target_runner_label=profile.label, target_cpu_model=profile.cpu_model,
        target_physical_cores=profile.physical_cores, target_logical_cpus=profile.logical_cpus,
        predictions_index=predictions_index,
    )
    if result is None or result.get("relation") != "scaled-vcpu":
        return None
    predicted = result.get("predicted_shape") if isinstance(result.get("predicted_shape"), dict) else {}
    result["pipelines"] = int(predicted["pipelines"])
    result["threads_per_pipeline"] = int(predicted["threads_per_pipeline"])
    result["allocated_threads"] = int(predicted["allocated_threads"])
    result["workload"] = {
        "detector_config_sha256": _sha256(detector_config),
        "golden_set_sha256": _sha256(golden_set),
        "max_dimension": max_dimension,
        "mode": "full",
        "strategy": "exhaustive",
    }
    result["source"] = f"predicted-{result.get('confidence', 'unknown')}"
    return result


def resolve_preferred_shape(
    *,
    parallelism_index: Path,
    detector_config: Path,
    golden_set: Path,
    max_dimension: int,
    profile: RunnerProfile,
) -> dict[str, Any] | None:
    """Compatibility wrapper for measured/equivalent preferred-shape callers."""
    detector, rows = compatible_optimizer_rows(
        parallelism_index=parallelism_index,
        detector_config=detector_config,
        golden_set=golden_set,
        max_dimension=max_dimension,
    )
    result = resolve_optimizer_intelligence(
        detector=detector, rows=rows, target_runner_name=profile.name,
        target_runner_label=profile.label, target_cpu_model=profile.cpu_model,
        target_physical_cores=profile.physical_cores, target_logical_cpus=profile.logical_cpus,
    )
    if result is None or result.get("relation") == "scaled-vcpu":
        return None
    predicted = result["predicted_shape"]
    return {
        "detector": detector,
        "pipelines": int(predicted["pipelines"]),
        "threads_per_pipeline": int(predicted["threads_per_pipeline"]),
        "allocated_threads": int(predicted["allocated_threads"]),
        "source": str(result["relation"]),
        "matched_observations": len(rows),
        "runner_name": profile.name,
        "runner_label": profile.label,
        "logical_cpus": profile.logical_cpus,
        "cpu_model": profile.cpu_model,
    }



def _print_shell(result: dict[str, Any]) -> None:
    print(int(result["pipelines"]), int(result["threads_per_pipeline"]), str(result.get("source") or "unknown"))


def _append_github_env(path: Path | None, values: dict[str, Any]) -> None:
    if not path:
        return
    with path.open("a", encoding="utf-8") as stream:
        for key, value in values.items():
            stream.write(f"{key}={value}\n")




def resolve_preferred_dispatch(
    *,
    shape_mode: str,
    regression_mode: str,
    strategy: str,
    limit: str | None,
    detector: str,
    parallelism_index: Path,
    detector_config_root: Path,
    golden_set: Path,
    max_dimension: int,
    requested_runner: str,
    specific_runner: str,
    custom_runner_label: str | None,
) -> dict[str, Any]:
    """Resolve runner + shape together before GitHub dispatches the regression job."""
    requested_labels, requested_label = _requested_runner_target(
        runner=requested_runner,
        specific_runner=specific_runner,
        custom_runner_label=custom_runner_label,
    )
    fallback = {
        "runs_on": requested_labels,
        "runner_label": requested_label,
        "runner_name": "requested",
        "exact": False,
        "source": "requested-runner",
        "runner_budget": runner_max_threads(requested_label),
    }
    mode = str(shape_mode or "auto").strip().lower()
    if (
        mode != "preferred"
        or regression_mode != "full"
        or strategy != "exhaustive"
        or str(limit or "").strip()
        or detector in {"all", "all-without-exhaustive"}
    ):
        return fallback

    detector_config = detector_config_root / f"{detector}.json"
    detector_id, rows = compatible_optimizer_rows(
        parallelism_index=parallelism_index,
        detector_config=detector_config,
        golden_set=golden_set,
        max_dimension=max_dimension,
    )
    if not rows:
        fallback["source"] = "requested-runner-no-preferred-history"
        return fallback

    target_logical = logical_cpus_from_capacity_label(requested_label)
    intelligence = resolve_selector_intelligence(
        detector=detector_id,
        rows=rows,
        required_labels=requested_labels,
        target_runner_label=requested_label,
        target_logical_cpus=target_logical,
    )
    if not intelligence:
        fallback["source"] = "requested-runner-no-compatible-preferred-history"
        return fallback

    shape = intelligence.get("predicted_shape") if isinstance(intelligence.get("predicted_shape"), dict) else {}
    pipelines = _as_int(shape.get("pipelines"))
    threads = _as_int(shape.get("threads_per_pipeline"))
    allocated = _as_int(shape.get("allocated_threads"))
    if not pipelines or not threads:
        fallback["source"] = "requested-runner-no-preferred-shape"
        return fallback
    allocated = allocated or pipelines * threads

    if intelligence.get("provenance") == "measured":
        row = intelligence.get("evidence_row") if isinstance(intelligence.get("evidence_row"), dict) else {}
        runner = _runner_from_row(row)
        observed_logical = _as_int(runner.get("logical_cpu_count"))
        policy_budget = runner_max_threads(requested_label, observed_logical)
        return {
            "runs_on": requested_labels,
            "runner_label": requested_label,
            "runner_name": str(runner.get("runner_name") or runner.get("name") or "preferred"),
            "exact": True,
            "pipelines": pipelines,
            "threads_per_pipeline": threads,
            "allocated_threads": allocated,
            "runner_budget": max(policy_budget, allocated),
            "source": "preferred-dispatch-optimizer",
            "detector": detector_id,
            "provenance": "measured",
        }

    # Before GitHub assigns a concrete machine, capacity labels such as 192t are
    # enough to make the same linear vCPU projection used inside the job.  Keep
    # the requested runner target; this is a prediction, not authority to reroute
    # the job onto the historical source runner.
    if not target_logical:
        fallback["source"] = "requested-runner-no-compatible-preferred-history"
        return fallback
    budget = runner_max_threads(requested_label, target_logical)
    if allocated > budget:
        fallback["source"] = "requested-runner-prediction-exceeds-budget"
        return fallback
    confidence = str(intelligence.get("confidence") or "low")
    return {
        "runs_on": requested_labels,
        "runner_label": requested_label,
        "runner_name": "requested",
        "exact": True,
        "pipelines": pipelines,
        "threads_per_pipeline": threads,
        "allocated_threads": allocated,
        "runner_budget": budget,
        "source": f"predicted-{confidence}-linear-vcpu-dispatch",
        "detector": detector_id,
        "provenance": "predicted",
        "anchor_logical_cpus": intelligence.get("anchor_logical_cpus"),
    }


def dispatch_output_env(result: dict[str, Any]) -> dict[str, str]:
    labels = [str(value) for value in result.get("runs_on", [])]
    values = {
        "runs_on": json.dumps(labels, separators=(",", ":")),
        "runner_label": str(result.get("runner_label") or "unknown"),
        "runner_name": str(result.get("runner_name") or "unknown"),
        "runner_labels": ",".join(labels),
        "runner_budget": str(result.get("runner_budget") or ""),
        "exact": "1" if result.get("exact") else "0",
        "source": str(result.get("source") or "unknown"),
        "pipelines": str(result.get("pipelines") or ""),
        "threads_per_pipeline": str(result.get("threads_per_pipeline") or ""),
    }
    values["github_hosted"] = "1" if labels == ["ubuntu-latest"] else "0"
    return values


def resolve_workflow_shape(
    *, shape_mode: str, regression_mode: str, strategy: str, limit: str | None,
    detector: str, manual_shape: str | None, parallelism_index: Path,
    predictions_index: Path | None, detector_config_root: Path, golden_set: Path,
    max_dimension: int, profile: RunnerProfile, prediction_out: Path | None,
    multidetector_index: Path | None = None,
    runner_budget: int | None = None, pre_resolved_pipelines: int | None = None,
    pre_resolved_threads: int | None = None, pre_resolved_source: str | None = None,
) -> dict[str, Any]:
    """Resolve all workflow shape policy in Python; YAML only supplies inputs."""
    budget = max(1, runner_budget or runner_max_threads(profile.label, profile.logical_cpus))

    def exact(pipelines: int, threads: int, source: str, prediction_file: Path | None = None) -> dict[str, Any]:
        allocated = pipelines * threads
        if allocated > budget:
            raise ValueError(
                f"Execution shape {pipelines}p/{threads}t allocates {allocated} "
                f"threads against detected runner budget {budget}"
            )
        result = {
            "exact": True, "pipelines": pipelines, "threads_per_pipeline": threads,
            "allocated_threads": allocated, "runner_budget": budget, "source": source,
        }
        if prediction_file:
            result["prediction_file"] = str(prediction_file)
        return result

    mode = (shape_mode or "auto").strip().lower()
    if pre_resolved_pipelines and pre_resolved_threads and not str(pre_resolved_source or "").startswith("predicted-"):
        return exact(
            int(pre_resolved_pipelines), int(pre_resolved_threads),
            str(pre_resolved_source or "preferred-dispatch"),
        )
    if mode == "auto":
        return {"exact": False, "source": "auto", "runner_budget": budget}

    if mode == "manual":
        if not manual_shape:
            raise ValueError("Manual execution shape is required (example: 8p/48t)")
        pipelines, threads = parse_manual_shape(manual_shape)
        return exact(pipelines, threads, "manual")

    if mode != "preferred":
        raise ValueError(f"Unknown execution shape mode: {shape_mode}")

    is_multidetector = detector in {"all", "all-without-exhaustive"}
    if is_multidetector and workload_class(regression_mode, strategy, limit) == "short":
        golden_sha = hashlib.sha256(golden_set.read_bytes()).hexdigest() if golden_set.is_file() else None
        detector_count = len(list(detector_config_root.glob("*.json")))
        preferred_multi = recommended_schedule(
            index_path=multidetector_index, detector_count=detector_count,
            runner_thread_budget=budget, runner_label=profile.label,
            golden_set_sha256=golden_sha, mode=regression_mode,
            strategy=strategy, limit=limit,
        )
        result = exact(
            int(preferred_multi["pipelines"]), int(preferred_multi["threads_per_pipeline"]),
            f"preferred-{preferred_multi['source']}",
        )
        result["multidetector"] = True
        result["evidence_observation_id"] = preferred_multi.get("evidence_observation_id")
        return result

    if regression_mode != "full" or strategy != "exhaustive" or str(limit or "").strip():
        return {"exact": False, "source": "auto-fallback-incompatible-workload", "runner_budget": budget}
    if is_multidetector:
        return {"exact": False, "source": "auto-fallback-all-full-exhaustive", "runner_budget": budget}

    detector_config = detector_config_root / f"{detector}.json"
    detector_id, optimizer_rows = compatible_optimizer_rows(
        parallelism_index=parallelism_index,
        detector_config=detector_config,
        golden_set=golden_set,
        max_dimension=max_dimension,
    )
    resolved = resolve_optimizer_intelligence(
        detector=detector_id, rows=optimizer_rows, target_runner_name=profile.name,
        target_runner_label=profile.label, target_cpu_model=profile.cpu_model,
        target_physical_cores=profile.physical_cores, target_logical_cpus=profile.logical_cpus,
        predictions_index=predictions_index,
    )
    if resolved:
        predicted = resolved["predicted_shape"]
        relation = str(resolved.get("relation") or "unknown")
        source = f"preferred-{relation}" if relation != "scaled-vcpu" else f"predicted-{resolved.get('confidence', 'low')}-linear-vcpu"
        prediction_file = None
        if relation == "scaled-vcpu":
            resolved["workload"] = {
                "detector_config_sha256": _sha256(detector_config),
                "golden_set_sha256": _sha256(golden_set),
                "max_dimension": max_dimension,
                "mode": "full",
                "strategy": "exhaustive",
            }
            resolved["source"] = source
            if prediction_out:
                prediction_out.parent.mkdir(parents=True, exist_ok=True)
                prediction_out.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                prediction_file = prediction_out
        return exact(
            int(predicted["pipelines"]), int(predicted["threads_per_pipeline"]),
            source, prediction_file,
        )
    return {"exact": False, "source": "auto-fallback-no-shape-history", "runner_budget": budget}


def workflow_shape_env(result: dict[str, Any]) -> dict[str, Any]:
    env = {
        "HTH_EXACT_EXECUTION_SHAPE_SOURCE": result.get("source", "unknown"),
        "HTH_EXECUTION_THREAD_BUDGET": result.get("runner_budget", ""),
        "HTH_EXECUTION_SHAPE_RESOLVED": "exact" if result.get("exact") else "auto",
    }
    if result.get("exact"):
        env.update({
            "THREADS": int(result["threads_per_pipeline"]),
            "DETECTOR_PIPELINES": int(result["pipelines"]),
            "HTH_EXACT_EXECUTION_SHAPE": "1",
            "HTH_ALLOW_THREAD_OVERSUBSCRIPTION": "false",
        })
    if result.get("prediction_file"):
        env["HTH_SHAPE_PREDICTION_FILE"] = result["prediction_file"]
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    manual = sub.add_parser("manual", help="Parse a manual Np/Mt execution shape")
    manual.add_argument("shape")

    preferred = sub.add_parser("preferred", help="Resolve persisted optimizer preference for this runner profile")
    preferred.add_argument("--parallelism-index", type=Path, required=True)
    preferred.add_argument("--detector-config", type=Path, required=True)
    preferred.add_argument("--golden-set", type=Path, required=True)
    preferred.add_argument("--max-dimension", type=int, required=True)
    preferred.add_argument("--runner-name")
    preferred.add_argument("--runner-label")

    predicted = sub.add_parser("predicted", help="Predict a shape from same-detector optimizer history when no compatible preference exists")
    predicted.add_argument("--parallelism-index", type=Path, required=True)
    predicted.add_argument("--predictions-index", type=Path)
    predicted.add_argument("--prediction-out", type=Path)
    predicted.add_argument("--detector-config", type=Path, required=True)
    predicted.add_argument("--golden-set", type=Path, required=True)
    predicted.add_argument("--max-dimension", type=int, required=True)
    predicted.add_argument("--runner-name")
    predicted.add_argument("--runner-label")

    record = sub.add_parser("record-prediction", help="Merge a generated shape prediction into the persistent prediction history")
    record.add_argument("--prediction-file", type=Path, required=True)
    record.add_argument("--predictions-index", type=Path, required=True)

    dispatch = sub.add_parser("dispatch-resolve", help="Resolve the runner and preferred shape before GitHub job dispatch")
    dispatch.add_argument("--shape-mode", required=True)
    dispatch.add_argument("--regression-mode", required=True)
    dispatch.add_argument("--strategy", required=True)
    dispatch.add_argument("--limit", default="")
    dispatch.add_argument("--detector", required=True)
    dispatch.add_argument("--parallelism-index", type=Path, required=True)
    dispatch.add_argument("--detector-config-root", type=Path, required=True)
    dispatch.add_argument("--golden-set", type=Path, required=True)
    dispatch.add_argument("--max-dimension", type=int, required=True)
    dispatch.add_argument("--requested-runner", required=True)
    dispatch.add_argument("--specific-runner", default="any")
    dispatch.add_argument("--custom-runner-label", default="")
    dispatch.add_argument("--github-output", type=Path)

    workflow = sub.add_parser("workflow-resolve", help="Resolve workflow execution shape and write GitHub environment")
    workflow.add_argument("--shape-mode", required=True)
    workflow.add_argument("--regression-mode", required=True)
    workflow.add_argument("--strategy", required=True)
    workflow.add_argument("--limit", default="")
    workflow.add_argument("--detector", required=True)
    workflow.add_argument("--manual-shape", default="")
    workflow.add_argument("--parallelism-index", type=Path, required=True)
    workflow.add_argument("--predictions-index", type=Path)
    workflow.add_argument("--multidetector-index", type=Path)
    workflow.add_argument("--detector-config-root", type=Path, required=True)
    workflow.add_argument("--golden-set", type=Path, required=True)
    workflow.add_argument("--max-dimension", type=int, required=True)
    workflow.add_argument("--runner-name")
    workflow.add_argument("--runner-label")
    workflow.add_argument("--prediction-out", type=Path)
    workflow.add_argument("--github-env", type=Path)
    workflow.add_argument("--runner-budget", type=int)
    workflow.add_argument("--pre-resolved-pipelines", type=int)
    workflow.add_argument("--pre-resolved-threads", type=int)
    workflow.add_argument("--pre-resolved-source")

    args = parser.parse_args()
    if args.command == "dispatch-resolve":
        result = resolve_preferred_dispatch(
            shape_mode=args.shape_mode, regression_mode=args.regression_mode,
            strategy=args.strategy, limit=args.limit, detector=args.detector,
            parallelism_index=args.parallelism_index, detector_config_root=args.detector_config_root,
            golden_set=args.golden_set, max_dimension=args.max_dimension,
            requested_runner=args.requested_runner, specific_runner=args.specific_runner,
            custom_runner_label=args.custom_runner_label,
        )
        values = dispatch_output_env(result)
        if args.github_output:
            _append_github_env(args.github_output, values)
        print(
            f"Dispatch execution: runner={values['runner_label']} runs-on={values['runs_on']} "
            f"shape={values['pipelines'] or 'auto'}p/{values['threads_per_pipeline'] or 'auto'}t "
            f"source={values['source']}"
        )
        return 0

    if args.command == "workflow-resolve":
        profile = current_runner_profile(name=args.runner_name, label=args.runner_label)
        result = resolve_workflow_shape(
            shape_mode=args.shape_mode, regression_mode=args.regression_mode,
            strategy=args.strategy, limit=args.limit, detector=args.detector,
            manual_shape=args.manual_shape, parallelism_index=args.parallelism_index,
            predictions_index=args.predictions_index, multidetector_index=args.multidetector_index,
            detector_config_root=args.detector_config_root, golden_set=args.golden_set,
            max_dimension=args.max_dimension, profile=profile,
            prediction_out=args.prediction_out, runner_budget=args.runner_budget,
            pre_resolved_pipelines=args.pre_resolved_pipelines,
            pre_resolved_threads=args.pre_resolved_threads,
            pre_resolved_source=args.pre_resolved_source,
        )
        _append_github_env(args.github_env, workflow_shape_env(result))
        if result.get("exact"):
            free_threads = max(0, int(result["runner_budget"]) - int(result["allocated_threads"]))
            print(
                f"Resolved execution shape: {result['pipelines']}p/"
                f"{result['threads_per_pipeline']}t ({result['source']}; threads "
                f"{result['allocated_threads']} allocated / {result['runner_budget']} max; "
                f"{free_threads} free)"
            )
        else:
            print(f"Execution shape: auto planner ({result['source']})")
        return 0

    if args.command == "manual":
        pipelines, threads = parse_manual_shape(args.shape)
        _print_shell({"pipelines": pipelines, "threads_per_pipeline": threads, "source": "manual"})
        return 0

    if args.command == "record-prediction":
        prediction = _read_json(args.prediction_file)
        merge_prediction(args.predictions_index, prediction)
        return 0

    profile = current_runner_profile(name=args.runner_name, label=args.runner_label)
    if args.command == "predicted":
        result = resolve_predicted_shape(
            parallelism_index=args.parallelism_index,
            predictions_index=args.predictions_index,
            detector_config=args.detector_config,
            golden_set=args.golden_set,
            max_dimension=args.max_dimension,
            profile=profile,
        )
        if result is None:
            return 2
        if args.prediction_out:
            args.prediction_out.parent.mkdir(parents=True, exist_ok=True)
            args.prediction_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _print_shell(result)
        return 0

    result = resolve_preferred_shape(
        parallelism_index=args.parallelism_index,
        detector_config=args.detector_config,
        golden_set=args.golden_set,
        max_dimension=args.max_dimension,
        profile=profile,
    )
    if result is None:
        return 2
    _print_shell(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
