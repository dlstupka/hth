"""Compact progress telemetry for long-running regression jobs."""
from __future__ import annotations

from dataclasses import dataclass
import time
import threading
from typing import Any, Callable, TextIO
import sys


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--"
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class ProgressSnapshot:
    completed: int
    total: int
    elapsed_seconds: float
    eta_seconds: float | None
    eval_rate: float | None
    best_mean_iou: float | None
    worst_iou: float | None
    failures: int
    current_profile: str


class ProgressReporter:
    """Emit one fixed-width heartbeat line at a time, plus sparse milestones."""

    HEADER = (
        "Elapsed    ETA       Complete             Eval Rate   Best Mean IoU   "
        "Worst IoU   Failures   Current Profile"
    )

    def __init__(
        self,
        *,
        total: int,
        interval_seconds: float = 60.0,
        stream: TextIO = sys.stdout,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.total = max(0, int(total))
        self.interval_seconds = float(interval_seconds)
        self.stream = stream
        self.clock = clock
        self.started = clock()
        self.last_emit = self.started
        self.completed = 0
        self.failures = 0
        self.best_result: dict[str, Any] | None = None
        self.current_profile = "--"
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_progress_at = self.started
        self._last_stall_warning_at: float | None = None

    def start(self) -> None:
        print("Regression Progress", file=self.stream)
        print(self.HEADER, file=self.stream)
        print("-" * len(self.HEADER), file=self.stream)
        self.emit(force=True)
        if self.interval_seconds > 0:
            self._thread = threading.Thread(target=self._heartbeat_loop, name="regression-heartbeat", daemon=True)
            self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            with self._lock:
                now = self.clock()
                stalled_for = now - self._last_progress_at
                if stalled_for >= 2 * self.interval_seconds and (
                    self._last_stall_warning_at is None or now - self._last_stall_warning_at >= 2 * self.interval_seconds
                ):
                    print(
                        f"\n>>> No forward progress for {_duration(stalled_for)}; detector evaluation may be stalled",
                        file=self.stream,
                        flush=True,
                    )
                    self._last_stall_warning_at = now
                self.emit(force=True)
                if stalled_for >= 2 * self.interval_seconds:
                    print(file=self.stream, flush=True)

    def observe(self, result: dict[str, Any], profile: str | None = None) -> None:
        with self._lock:
            self.completed += 1
            self._last_progress_at = self.clock()
            summary = result.get("summary", {})
            self.failures += int(summary.get("failure_count", 0) or 0)
            candidate_key = (
                -float(summary.get("mean_iou", 0.0) or 0.0),
                -float(summary.get("minimum_iou", 0.0) or 0.0),
                int(summary.get("failure_count", 0) or 0),
            )
            old_key = None
            if self.best_result is not None:
                old = self.best_result.get("summary", {})
                old_key = (
                    -float(old.get("mean_iou", 0.0) or 0.0),
                    -float(old.get("minimum_iou", 0.0) or 0.0),
                    int(old.get("failure_count", 0) or 0),
                )
            is_new_best = old_key is None or candidate_key < old_key
            if is_new_best:
                self.best_result = result
                self.current_profile = profile or f"ps:{str(result.get('parameter_set_id', 'unknown'))[:8]}"
                if self.completed > 1:
                    print(f"\n>>> New best profile {self.current_profile}", file=self.stream)
                    self.emit(force=True)
                    print(file=self.stream)
                    return
            self.emit()

    def snapshot(self) -> ProgressSnapshot:
        elapsed = max(0.0, self.clock() - self.started)
        rate = self.completed / elapsed if elapsed > 0 and self.completed else None
        remaining = max(0, self.total - self.completed)
        eta = remaining / rate if rate else None
        summary = self.best_result.get("summary", {}) if self.best_result else {}
        return ProgressSnapshot(
            completed=self.completed,
            total=self.total,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            eval_rate=rate,
            best_mean_iou=float(summary["mean_iou"]) if "mean_iou" in summary else None,
            worst_iou=float(summary["minimum_iou"]) if "minimum_iou" in summary else None,
            failures=self.failures,
            current_profile=self.current_profile,
        )

    def emit(self, *, force: bool = False) -> bool:
        now = self.clock()
        if not force and now - self.last_emit < self.interval_seconds:
            return False
        snap = self.snapshot()
        percent = (100.0 * snap.completed / snap.total) if snap.total else 0.0
        complete = f"{snap.completed}/{snap.total} {percent:5.1f}%"
        rate = f"{snap.eval_rate:8.2f}/s" if snap.eval_rate is not None else "       --"
        best = f"{snap.best_mean_iou:13.4f}" if snap.best_mean_iou is not None else "           --"
        worst = f"{snap.worst_iou:9.4f}" if snap.worst_iou is not None else "       --"
        print(
            f"{_duration(snap.elapsed_seconds):10} {_duration(snap.eta_seconds):9} "
            f"{complete:20} {rate:11} {best}   {worst}   {snap.failures:8d}   {snap.current_profile}",
            file=self.stream,
            flush=True,
        )
        self.last_emit = now
        return True

    def finish(self) -> ProgressSnapshot:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=min(1.0, self.interval_seconds))
        with self._lock:
            self.emit(force=True)
            return self.snapshot()
