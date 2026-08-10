#!/usr/bin/env python3
"""Dispatch one ordinary execution-optimizer run per selected detector."""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def configured_detectors(detector_dir: Path) -> list[str]:
    return sorted(path.stem for path in detector_dir.glob("*.json") if path.is_file())


def preferred_detectors(index_path: Path) -> set[str]:
    if not index_path.is_file():
        return set()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return {
        str(row["detector_id"])
        for row in payload.get("preferred_executor_configurations", [])
        if isinstance(row, dict) and row.get("detector_id")
    }


def resolve_targets(detector_dir: Path, index_path: Path, mode: str) -> list[str]:
    detectors = configured_detectors(detector_dir)
    if not detectors:
        raise ValueError(f"No detector configurations found beneath {detector_dir}")
    if mode == "all":
        return detectors
    if mode == "all-without-preference":
        preferred = preferred_detectors(index_path)
        return [detector for detector in detectors if detector not in preferred]
    raise ValueError(f"Unsupported optimizer target mode: {mode}")


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _dispatch(endpoint: str, token: str, ref: str, detector: str, common_inputs: dict[str, Any]) -> None:
    inputs = dict(common_inputs)
    inputs["algorithm"] = detector
    body = json.dumps({"ref": ref, "inputs": inputs}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2026-03-10",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"Unexpected dispatch status {response.status} for {detector}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Failed to dispatch {detector}: HTTP {exc.code}: {detail}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "all-without-preference"), required=True)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--optimizer-index", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--workflow", default="execution-optimizer.yml")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--specific-runner", required=True)
    parser.add_argument("--custom-runner-label", default="")
    parser.add_argument("--pipeline-enumeration", required=True)
    parser.add_argument("--pipeline-min", required=True)
    parser.add_argument("--pipeline-max", required=True)
    parser.add_argument("--thread-min", required=True)
    parser.add_argument("--thread-max", required=True)
    parser.add_argument("--allow-thread-oversubscription", required=True)
    parser.add_argument("--early-stop", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--results-repository", required=True)
    parser.add_argument("--golden-set", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--max-dimension", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--debug-level", required=True)
    args = parser.parse_args()

    targets = resolve_targets(args.detector_dir, args.optimizer_index, args.mode)
    print(f"Optimizer target mode: {args.mode}")
    print(f"Detector count: {len(targets)}")
    for detector in targets:
        print(f"  {detector}")

    common_inputs: dict[str, Any] = {
        "runner": args.runner,
        "specific_runner": args.specific_runner,
        "custom_runner_label": args.custom_runner_label,
        "pipeline_enumeration": args.pipeline_enumeration,
        "pipeline_min": args.pipeline_min,
        "pipeline_max": args.pipeline_max,
        "thread_min": args.thread_min,
        "thread_max": args.thread_max,
        "allow_thread_oversubscription": _bool(args.allow_thread_oversubscription),
        "early_stop": _bool(args.early_stop),
        "resume": args.resume,
        "results_repository": args.results_repository,
        "golden_set": args.golden_set,
        "image_root": args.image_root,
        "max_dimension": args.max_dimension,
        "strategy": args.strategy,
        "debug_level": args.debug_level,
    }

    token = os.environ.get("GH_TOKEN", "").strip()
    if targets and not token:
        raise SystemExit("GH_TOKEN is required to dispatch optimizer workflows")
    endpoint = f"https://api.github.com/repos/{args.repository}/actions/workflows/{args.workflow}/dispatches"
    for detector in targets:
        _dispatch(endpoint, token, args.ref, detector, common_inputs)
        print(f"Dispatched optimizer: {detector}")

    if args.summary:
        lines = [
            "### Execution optimizer dispatcher",
            "",
            f"Mode: `{args.mode}`  ",
            f"Search method: `{args.pipeline_enumeration}`  ",
        ]
        if targets:
            lines.extend([f"Dispatched **{len(targets)}** detector optimizer run(s):", ""])
            lines.extend(f"- `{detector}`" for detector in targets)
        else:
            lines.append("All configured detectors already have a collected preferred shape.")
        lines.append("")
        args.summary.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
