#!/usr/bin/env python3
"""Regenerate HTH human-facing reports from persisted results."""
from __future__ import annotations

import argparse
from pathlib import Path

from hth.calibration_report import (
    calibration_run_dirs,
    generate_calibration_manifest,
    smoke_run_dirs,
)
from hth.optimizer_report import generate_optimizer_report, generate_optimizer_report_all

__all__ = [
    "calibration_run_dirs",
    "smoke_run_dirs",
    "generate_calibration_manifest",
    "generate_optimizer_report",
    "generate_optimizer_report_all",
]


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="report", required=True)
    calibration = subcommands.add_parser("detector-calibration-manifest")
    calibration.add_argument("--results-root", type=Path, required=True)
    calibration.add_argument("--golden-set", type=Path)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--pipeline-repository", default="")
    calibration.add_argument("--results-repository", default="")
    calibration.add_argument("--results-commit", default="")
    calibration.add_argument("--run-url", default="")
    optimizer = subcommands.add_parser("execution-optimizer")
    optimizer.add_argument("--results-root", type=Path, required=True)
    optimizer.add_argument("--detector", required=True)
    optimizer.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.report == "detector-calibration-manifest":
        path = generate_calibration_manifest(
            args.results_root,
            args.output,
            golden_set=args.golden_set,
            pipeline_repository=args.pipeline_repository,
            results_repository=args.results_repository,
            results_commit=args.results_commit,
            run_url=args.run_url,
        )
        print(path)
        return 0
    paths = generate_optimizer_report(args.results_root, args.detector, args.output_dir)
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
