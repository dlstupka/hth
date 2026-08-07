#!/usr/bin/env python3
"""Sample lightweight Linux runner metrics from /proc for optimizer heartbeats."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_proc_stat(path: Path = Path("/proc/stat")) -> dict[str, int] | None:
    if not path.is_file():
        return None
    first = path.read_text(encoding="utf-8").splitlines()[0].split()
    if not first or first[0] != "cpu" or len(first) < 6:
        return None
    values = [int(value) for value in first[1:]]
    names = ["user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal", "guest", "guest_nice"]
    return {name: values[index] if index < len(values) else 0 for index, name in enumerate(names)}


def _read_meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, int] | None:
    if not path.is_file():
        return None
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        token = raw.strip().split()[0] if raw.strip() else "0"
        try:
            values[key] = int(token) * 1024
        except ValueError:
            continue
    return values


def _read_load(path: Path = Path("/proc/loadavg")) -> tuple[float, float, float] | None:
    if not path.is_file():
        return None
    values = path.read_text(encoding="utf-8").split()
    if len(values) < 3:
        return None
    return float(values[0]), float(values[1]), float(values[2])


def _cpu_delta(current: dict[str, int] | None, previous: dict[str, int] | None) -> tuple[float | None, float | None]:
    if not current or not previous:
        return None, None
    deltas = {key: max(0, current.get(key, 0) - previous.get(key, 0)) for key in current}
    total = sum(deltas.values())
    if total <= 0:
        return None, None
    idle = deltas.get("idle", 0)
    iowait = deltas.get("iowait", 0)
    cpu_pct = 100.0 * (total - idle - iowait) / total
    iowait_pct = 100.0 * iowait / total
    return cpu_pct, iowait_pct


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    if amount == 0:
        return "0"
    units = ("B", "K", "M", "G", "T", "P")
    unit = units[0]
    for candidate in units:
        unit = candidate
        if abs(amount) < 1024.0 or candidate == units[-1]:
            break
        amount /= 1024.0
    return f"{amount:.1f}{unit}" if unit not in {"B", "K"} else f"{amount:.0f}{unit}"


def sample_runner_metrics(
    *,
    runner_label: str,
    runner_name: str,
    optimizer_run_id: str,
    state_file: Path,
    output_log: Path,
    cpu_state_file: Path,
) -> tuple[dict[str, Any], str]:
    state = _read_state(state_file)
    current_cpu = _read_proc_stat()
    previous_cpu = None
    if cpu_state_file.is_file():
        try:
            payload = json.loads(cpu_state_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                previous_cpu = {str(k): int(v) for k, v in payload.items()}
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            previous_cpu = None
    if current_cpu is not None:
        cpu_state_file.parent.mkdir(parents=True, exist_ok=True)
        cpu_state_file.write_text(json.dumps(current_cpu, sort_keys=True) + "\n", encoding="utf-8")
    cpu_pct, iowait_pct = _cpu_delta(current_cpu, previous_cpu)

    load = _read_load()
    mem = _read_meminfo()
    total = mem.get("MemTotal") if mem else None
    available = mem.get("MemAvailable") if mem else None
    used = (total - available) if total is not None and available is not None else None
    swap_total = mem.get("SwapTotal") if mem else None
    swap_free = mem.get("SwapFree") if mem else None
    swap_used = (swap_total - swap_free) if swap_total is not None and swap_free is not None else None

    record: dict[str, Any] = {
        "observed_at_utc": _now(),
        "optimizer_run_id": str(optimizer_run_id),
        "runner_label": runner_label,
        "runner_name": runner_name,
        "phase": state.get("phase"),
        "shape_sequence": state.get("shape_sequence"),
        "shape_total": state.get("shape_total"),
        "detector_id": state.get("detector_id"),
        "pipelines": state.get("pipelines"),
        "shards": state.get("shards"),
        "threads_per_pipeline": state.get("threads_per_pipeline"),
        "load1": load[0] if load else None,
        "load5": load[1] if load else None,
        "load15": load[2] if load else None,
        "cpu_pct": cpu_pct,
        "iowait_pct": iowait_pct,
        "ram_used_bytes": used,
        "ram_total_bytes": total,
        "swap_used_bytes": swap_used,
        "swap_total_bytes": swap_total,
    }
    output_log.parent.mkdir(parents=True, exist_ok=True)
    with output_log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")

    label = runner_label or "unknown"
    name = runner_name or "unknown"
    cpu_text = f"{cpu_pct:.1f}%" if cpu_pct is not None else "n/a"
    io_text = f"{iowait_pct:.1f}%" if iowait_pct is not None else "n/a"
    load_text = f"{load[0]:.1f}" if load else "unknown"
    line = (
        f"[runner {label}/{name}] load={load_text} cpu={cpu_text} iowait={io_text} "
        f"ram={_format_bytes(used)}/{_format_bytes(total)} swap={_format_bytes(swap_used)}"
    )
    return record, line


def summarize_runner_metrics(log_path: Path, *, optimizer_run_id: str, shape_sequence: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if str(row.get("optimizer_run_id")) != str(optimizer_run_id):
                continue
            if row.get("shape_sequence") != shape_sequence:
                continue
            rows.append(row)

    def values(key: str) -> list[float]:
        result: list[float] = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)):
                result.append(float(value))
        return result

    result: dict[str, Any] = {"sample_count": len(rows)}
    for key, output_prefix in (
        ("load1", "load1"),
        ("cpu_pct", "cpu_pct"),
        ("iowait_pct", "iowait_pct"),
        ("ram_used_bytes", "ram_used_bytes"),
        ("swap_used_bytes", "swap_used_bytes"),
    ):
        found = values(key)
        result[f"avg_{output_prefix}"] = (sum(found) / len(found)) if found else None
        result[f"peak_{output_prefix}"] = max(found) if found else None
    if rows:
        result["first_sample_utc"] = rows[0].get("observed_at_utc")
        result["last_sample_utc"] = rows[-1].get("observed_at_utc")
    return result


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-label", required=True)
    parser.add_argument("--runner-name", required=True)
    parser.add_argument("--optimizer-run-id", required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument("--cpu-state-file", type=Path, required=True)
    args = parser.parse_args()
    _, line = sample_runner_metrics(
        runner_label=args.runner_label,
        runner_name=args.runner_name,
        optimizer_run_id=args.optimizer_run_id,
        state_file=args.state_file,
        output_log=args.output_log,
        cpu_state_file=args.cpu_state_file,
    )
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
