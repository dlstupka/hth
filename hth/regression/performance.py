"""Runner performance telemetry for detector regression."""
from __future__ import annotations

import json
import os
try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]
import threading
import time
from pathlib import Path
from typing import Any, Callable


def available_cpu_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


def physical_core_count() -> int | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return None
    physical: set[tuple[str, str]] = set()
    current: dict[str, str] = {}
    for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines() + [""]:
        if not line.strip():
            if "physical id" in current and "core id" in current:
                physical.add((current["physical id"], current["core id"]))
            current = {}
        elif ":" in line:
            key, value = line.split(":", 1)
            current[key.strip().lower()] = value.strip()
    return len(physical) or None


def peak_rss_bytes() -> int | None:
    try:
        if resource is None:
            return None
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        return value if hasattr(os, "uname") and os.uname().sysname == "Darwin" else value * 1024
    except (AttributeError, OSError, ValueError):
        return None


class PerformanceSampler:
    """Write periodic process-performance samples without affecting progress output."""

    def __init__(
        self,
        path: Path,
        *,
        snapshot: Callable[[], dict[str, Any]],
        interval_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        process_clock: Callable[[], float] = time.process_time,
    ) -> None:
        self.path = path
        self.snapshot_callback = snapshot
        self.interval_seconds = float(interval_seconds)
        self.clock = clock
        self.process_clock = process_clock
        self.started = clock()
        self.last_wall = self.started
        self.last_cpu = process_clock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.samples: list[dict[str, Any]] = []

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sample()
        if self.interval_seconds > 0:
            self._thread = threading.Thread(target=self._loop, name="regression-performance", daemon=True)
            self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample()

    def sample(self) -> dict[str, Any]:
        with self._lock:
            now = self.clock()
            cpu_now = self.process_clock()
            wall_delta = max(0.0, now - self.last_wall)
            cpu_delta = max(0.0, cpu_now - self.last_cpu)
            payload = {
                "elapsed_seconds": round(max(0.0, now - self.started), 3),
                "process_cpu_seconds": round(cpu_now, 3),
                "process_cpu_percent": round(100.0 * cpu_delta / wall_delta, 2) if wall_delta > 0 else None,
                "peak_rss_bytes": peak_rss_bytes(),
                **self.snapshot_callback(),
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
            self.samples.append(payload)
            self.last_wall = now
            self.last_cpu = cpu_now
            return payload

    def finish(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(1.0, self.interval_seconds))
        self.sample()
        return list(self.samples)
