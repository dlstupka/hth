#!/usr/bin/env python3
"""Run the bounded paired layout smoke with one Kraken process per view."""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import json
import os
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _command(kraken: str, pages: list[dict], inputs_root: Path, results: Path, view: str, threads: int) -> list[str]:
    command = [kraken, "-d", "cpu", "--threads", str(threads), "--raise-on-error"]
    for page in pages:
        ordinal = int(page["global_ordinal"])
        source = inputs_root / page[f"{view}_file"]
        if not source.is_file():
            raise FileNotFoundError(source)
        command += ["-i", str(source.resolve()), str((results / f"fs_{ordinal:04d}.json").resolve())]
    return command + ["segment", "-bl"]


def _run_view(command: list[str], results: Path, log: Path, page_count: int) -> dict:
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    with log.open("wb") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
        while True:
            try:
                exit_code = process.wait(timeout=15)
                break
            except subprocess.TimeoutExpired:
                print(f"Layout smoke: {results.name} {len(list(results.glob('fs_*.json')))}/{page_count} pages", flush=True)
    produced = len(list(results.glob("fs_*.json")))
    output = log.read_text(encoding="utf-8", errors="replace")
    print(output[-4000:], flush=True)
    if exit_code != 0 or produced != page_count:
        raise RuntimeError(f"Kraken {results.name} failed: exit={exit_code}, outputs={produced}/{page_count}; see {log}")
    return {
        "view": results.name,
        "pages": produced,
        "batch_wall_seconds": round(time.perf_counter() - started, 3),
        "polygonizer_warnings": output.count("Polygonizer failed"),
    }


def run(paired_inputs: Path, output: Path, threads: int, expected_kraken_version: str, github_summary: Path | None = None) -> dict:
    if threads <= 0:
        raise ValueError("Positive thread count required")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}")
    inputs = json.loads(paired_inputs.read_text(encoding="utf-8"))
    pages = inputs["pages"]
    if inputs.get("golden_set_id") != "HTH-GOLDEN-0002" or len(pages) != 18:
        raise ValueError("This smoke requires the frozen 18-page GS0002 input")
    if len({int(page["global_ordinal"]) for page in pages}) != len(pages):
        raise ValueError("Paired inputs contain duplicate page ordinals")
    version = metadata.version("kraken")
    if version != expected_kraken_version:
        raise ValueError(f"Expected Kraken {expected_kraken_version}, found {version}")
    model = Path(str(importlib.resources.files("kraken").joinpath("blla.mlmodel")))
    if not model.is_file():
        raise FileNotFoundError(model)
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    kraken = shutil.which("kraken")
    if kraken is None:
        raise FileNotFoundError("Kraken CLI is not on PATH")

    output.mkdir(parents=True)
    shutil.copyfile(paired_inputs, output / "paired-inputs.json")
    batches = []
    for view in ("source", "normalized"):
        results = output / view
        results.mkdir()
        command = _command(kraken, pages, paired_inputs.parent, results, view, threads)
        batches.append(_run_view(command, results, output / f"{view}.log", len(pages)))
    execution = {"schema_version": "1.0", "batches": batches}
    (output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
    report_command = [
        sys.executable, str(ROOT / "tools" / "layout-smoke-report.py"),
        "--paired-inputs", str(output / "paired-inputs.json"),
        "--source-results", str(output / "source"),
        "--normalized-results", str(output / "normalized"),
        "--model-sha256", model_sha256,
        "--engine-version", version,
        "--device", "cpu",
        "--threads", str(threads),
        "--execution", str(output / "execution.json"),
        "--output", str(output / "layout-smoke-metrics.json"),
    ]
    if github_summary is not None:
        report_command += ["--github-summary", str(github_summary)]
    subprocess.run(report_command, check=True)
    return execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--expected-kraken-version", default="7.0.2")
    parser.add_argument("--github-summary", type=Path)
    args = parser.parse_args()
    run(args.paired_inputs, args.output, args.threads, args.expected_kraken_version, args.github_summary)


if __name__ == "__main__":
    main()
