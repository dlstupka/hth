#!/usr/bin/env python3
"""Record durable HTH stage timestamps and elapsed times for CI logs and reports."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def display_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 10:
        return f"{seconds:.1f}s"
    rounded = int(round(seconds))
    minutes, secs = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def write_output(name: str, value: str) -> None:
    destination = os.environ.get("GITHUB_OUTPUT", "").strip()
    if destination:
        with Path(destination).open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{name}={value}\n")


def start(stage: str) -> int:
    epoch = time.time()
    started_at = utc_now()
    print("=" * 60)
    print(f"HTH :: {stage}")
    print("=" * 60)
    print(f"Started: {started_at}")
    write_output("epoch", f"{epoch:.6f}")
    write_output("started_at", started_at)
    return 0


def finish(stage: str, start_epoch: str, started_at: str, status: str, timings_file: str) -> int:
    ended_epoch = time.time()
    ended_at = utc_now()
    try:
        start_value = float(start_epoch)
    except (TypeError, ValueError):
        start_value = ended_epoch
    elapsed = max(0.0, ended_epoch - start_value)
    normalized_status = (status or "unknown").strip().lower()
    record = {
        "stage": stage,
        "status": normalized_status,
        "started_at_utc": started_at or "unknown",
        "completed_at_utc": ended_at,
        "elapsed_seconds": round(elapsed, 3),
    }
    if timings_file:
        path = Path(timings_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print("-" * 60)
    print(f"Completed: {stage}")
    print(f"Status:    {normalized_status}")
    print(f"Finished:  {ended_at}")
    print(f"Elapsed:   {display_duration(elapsed)}")
    print("-" * 60)
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    start_p = sub.add_parser("start")
    start_p.add_argument("--stage", required=True)
    finish_p = sub.add_parser("finish")
    finish_p.add_argument("--stage", required=True)
    finish_p.add_argument("--start-epoch", default="")
    finish_p.add_argument("--started-at", default="")
    finish_p.add_argument("--status", default="unknown")
    finish_p.add_argument("--timings-file", default="")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.command == "start":
        return start(args.stage)
    return finish(args.stage, args.start_epoch, args.started_at, args.status, args.timings_file)


if __name__ == "__main__":
    raise SystemExit(main())
