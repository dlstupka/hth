#!/usr/bin/env python3
"""Dispatch ordinary full exhaustive detector regressions that are still missing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from hth.detector_catalog import configured_detectors as catalog_configured_detectors
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configured_detectors(detector_dir: Path) -> list[str]:
    return catalog_configured_detectors(detector_dir, automatic_only=True)


def exhaustive_detectors(
    detector_dir: Path,
    calibration_index: Path,
    golden_set: Path,
) -> set[str]:
    if not calibration_index.is_file():
        return set()
    payload = json.loads(calibration_index.read_text(encoding="utf-8"))
    golden_sha = _sha256(golden_set)
    config_sha = {path.stem: _sha256(path) for path in detector_dir.glob("*.json") if path.is_file()}
    complete: set[str] = set()
    for row in payload.get("entries", []):
        if not isinstance(row, dict):
            continue
        detector = str(row.get("detector_id") or "")
        search = row.get("search") if isinstance(row.get("search"), dict) else {}
        if not detector or not search.get("exhaustive_complete"):
            continue
        if row.get("golden_set_sha256") != golden_sha:
            continue
        if row.get("detector_config_sha256") != config_sha.get(detector):
            continue
        complete.add(detector)
    return complete


def resolve_targets(
    detector_dir: Path,
    calibration_index: Path,
    golden_set: Path,
    mode: str,
) -> list[str]:
    detectors = configured_detectors(detector_dir)
    if not detectors:
        raise ValueError(f"No detector configurations found beneath {detector_dir}")
    if mode == "all-without-exhaustive":
        complete = exhaustive_detectors(detector_dir, calibration_index, golden_set)
        return [detector for detector in detectors if detector not in complete]
    raise ValueError(f"Unsupported regression target mode: {mode}")


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
    parser.add_argument("--mode", choices=("all-without-exhaustive",), required=True)
    parser.add_argument("--detector-dir", type=Path, required=True)
    parser.add_argument("--calibration-index", type=Path, required=True)
    parser.add_argument("--golden-set", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--workflow", default="regress-detector.yml")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--results-repository", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--execution-shape", required=True)
    parser.add_argument("--manual-execution-shape", default="")
    parser.add_argument("--runner", required=True)
    parser.add_argument("--specific-runner", required=True)
    parser.add_argument("--custom-runner-label", default="")
    parser.add_argument("--max-dimension", required=True)
    parser.add_argument("--shards", default="")
    parser.add_argument("--shard-target-minutes", required=True)
    parser.add_argument("--shard-lease-minutes", required=True)
    parser.add_argument("--detector-loading-strategy", required=True)
    parser.add_argument("--pipeline-stagger-minutes", required=True)
    parser.add_argument("--top", required=True)
    parser.add_argument("--debug-level", required=True)
    args = parser.parse_args()

    targets = resolve_targets(args.detector_dir, args.calibration_index, args.golden_set, args.mode)
    print(f"Regression target mode: {args.mode}")
    print(f"Detector count: {len(targets)}")
    for detector in targets:
        print(f"  {detector}")

    # This mode exists specifically to fill missing authoritative exhaustive evidence,
    # so child runs are always full, exhaustive, and unlimited regardless of stale
    # values left in the other workflow-dispatch controls.
    common_inputs: dict[str, Any] = {
        "mode": "full",
        "strategy": "exhaustive",
        "limit": "",
        "golden_set": str(args.golden_set).removeprefix("hth-pipeline/"),
        "results_repository": args.results_repository,
        "image_root": args.image_root,
        "execution_shape": args.execution_shape,
        "manual_execution_shape": args.manual_execution_shape,
        "runner": args.runner,
        "specific_runner": args.specific_runner,
        "custom_runner_label": args.custom_runner_label,
        "max_dimension": args.max_dimension,
        "shards": args.shards,
        "shard_target_minutes": args.shard_target_minutes,
        "shard_lease_minutes": args.shard_lease_minutes,
        "detector_loading_strategy": args.detector_loading_strategy,
        "pipeline_stagger_minutes": args.pipeline_stagger_minutes,
        "top": args.top,
        "debug_level": args.debug_level,
    }

    token = os.environ.get("GH_TOKEN", "").strip()
    if targets and not token:
        raise SystemExit("GH_TOKEN is required to dispatch regression workflows")
    endpoint = f"https://api.github.com/repos/{args.repository}/actions/workflows/{args.workflow}/dispatches"
    for detector in targets:
        _dispatch(endpoint, token, args.ref, detector, common_inputs)
        print(f"Dispatched exhaustive regression: {detector}")

    if args.summary:
        lines = [
            "### Exhaustive regression dispatcher",
            "",
            "Mode: `all-without-exhaustive`  ",
            "Child runs: `full` / `exhaustive` / unlimited  ",
        ]
        if targets:
            lines.extend([f"Dispatched **{len(targets)}** detector regression run(s):", ""])
            lines.extend(f"- `{detector}`" for detector in targets)
        else:
            lines.append("Every configured detector already has compatible exhaustive calibration evidence.")
        lines.append("")
        args.summary.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
