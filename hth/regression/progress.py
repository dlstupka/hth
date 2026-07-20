"""Compact progress telemetry for long-running regression jobs."""
from __future__ import annotations

from dataclasses import dataclass
import sys
import threading
import time
from typing import Any, Callable, TextIO


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
    worst_page_iou: float | None
    failures: int
    evaluating: str
    last_improvement_seconds: float | None
    last_improvement_elapsed_seconds: float | None


class ProgressReporter:
    """Emit one fixed-width heartbeat line per interval plus sparse milestones."""

    COLUMN_WIDTHS = {
        "elapsed": 8,
        "eta": 8,
        "complete": 13,
        "rate": 9,
        "best": 13,
        "worst": 9,
        "failures": 8,
        "last_improvement": 16,
    }
    HEADER = (
        f"{'Elapsed':<{COLUMN_WIDTHS['elapsed']}}  "
        f"{'ETA':<{COLUMN_WIDTHS['eta']}}  "
        f"{'Complete':<{COLUMN_WIDTHS['complete']}}  "
        f"{'Eval Rate':>{COLUMN_WIDTHS['rate']}}  "
        f"{'Best Mean IoU':>{COLUMN_WIDTHS['best']}}  "
        f"{'Worst IoU':>{COLUMN_WIDTHS['worst']}}  "
        f"{'Failures':>{COLUMN_WIDTHS['failures']}}  "
        f"{'Last Improvement':<{COLUMN_WIDTHS['last_improvement']}}  "
        "Evaluating"
    )

    def __init__(
        self,
        *,
        total: int,
        interval_seconds: float = 60.0,
        stream: TextIO = sys.stdout,
        clock: Callable[[], float] = time.monotonic,
        eta_min_completed: int = 1,
        eta_min_elapsed_seconds: float = 0.0,
    ) -> None:
        self.total = max(0, int(total))
        self.interval_seconds = float(interval_seconds)
        self.stream = stream
        self.clock = clock
        self.eta_min_completed = max(1, int(eta_min_completed))
        self.eta_min_elapsed_seconds = max(0.0, float(eta_min_elapsed_seconds))
        self.started = clock()
        self.last_emit = self.started
        self.completed = 0
        self.failures = 0
        self.best_mean_result: dict[str, Any] | None = None
        self.best_worst_result: dict[str, Any] | None = None
        self.baseline_mean_iou: float | None = None
        self.baseline_surpassed = False
        self.evaluating = "--"
        self._last_improvement_at: float | None = None
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
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name="regression-heartbeat",
                daemon=True,
            )
            self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            with self._lock:
                now = self.clock()
                stalled_for = now - self._last_progress_at
                if stalled_for >= 2 * self.interval_seconds and (
                    self._last_stall_warning_at is None
                    or now - self._last_stall_warning_at >= 2 * self.interval_seconds
                ):
                    print(
                        f"\n{_duration(now - self.started)} >>> No forward progress for "
                        f"{_duration(stalled_for)}; detector evaluation may be stalled",
                        file=self.stream,
                        flush=True,
                    )
                    self._last_stall_warning_at = now
                self.emit(force=True)
                if stalled_for >= 2 * self.interval_seconds:
                    print(file=self.stream, flush=True)

    @staticmethod
    def _profile_name(result: dict[str, Any], profile: str | None) -> str:
        return profile or f"ps:{str(result.get('parameter_set_id', 'unknown'))[:8]}"

    def announce(self, labels: str | list[str], profile: str | None = None) -> None:
        """Interleave one or more sparse milestone labels with a single status row."""
        milestone_labels = [labels] if isinstance(labels, str) else list(labels)
        if not milestone_labels:
            return
        now = self.clock()
        suffix = f" {profile}" if profile else ""
        print(file=self.stream)
        for label in milestone_labels:
            print(f"{_duration(now - self.started)} >>> {label}{suffix}", file=self.stream)
        self.emit(force=True)
        print(file=self.stream, flush=True)


    def begin_evaluation(self, profile: str) -> None:
        """Record the parameter set currently being evaluated for heartbeat telemetry."""
        with self._lock:
            self.evaluating = profile

    def observe(self, result: dict[str, Any], profile: str | None = None) -> None:
        with self._lock:
            self.completed += 1
            now = self.clock()
            self._last_progress_at = now
            summary = result.get("summary", {})
            self.failures += int(summary.get("failure_count", 0) or 0)
            mean_iou = float(summary.get("mean_iou", 0.0) or 0.0)
            worst_iou = float(summary.get("minimum_iou", 0.0) or 0.0)
            profile_name = self._profile_name(result, profile)

            old_best_mean = (
                float(self.best_mean_result.get("summary", {}).get("mean_iou", 0.0) or 0.0)
                if self.best_mean_result is not None
                else None
            )
            old_best_worst = (
                float(self.best_worst_result.get("summary", {}).get("minimum_iou", 0.0) or 0.0)
                if self.best_worst_result is not None
                else None
            )

            new_best_mean = old_best_mean is None or mean_iou > old_best_mean
            new_best_worst = old_best_worst is None or worst_iou > old_best_worst

            if new_best_mean:
                self.best_mean_result = result
            if new_best_worst:
                self.best_worst_result = result
            if new_best_mean or new_best_worst:
                self._last_improvement_at = now

            if profile == "baseline":
                self.baseline_mean_iou = mean_iou

            milestones: list[str] = []
            if self.completed > 1 and new_best_mean:
                milestones.append("New best mean IoU")
            if self.completed > 1 and new_best_worst:
                milestones.append("New worst-page IoU")

            if (
                not self.baseline_surpassed
                and self.baseline_mean_iou is not None
                and self.best_mean_result is not None
                and float(self.best_mean_result.get("summary", {}).get("mean_iou", 0.0) or 0.0)
                > self.baseline_mean_iou
            ):
                self.baseline_surpassed = True
                milestones.append("Baseline surpassed")

            if milestones:
                self.announce(milestones, profile_name)
            else:
                self.emit()

    def snapshot(self) -> ProgressSnapshot:
        now = self.clock()
        elapsed = max(0.0, now - self.started)
        rate = self.completed / elapsed if elapsed > 0 and self.completed else None
        remaining = max(0, self.total - self.completed)
        # A first completed evaluation is enough for a useful operational estimate.
        # The estimate will naturally stabilize as additional profiles complete.
        eta = remaining / rate if rate else None
        mean_summary = self.best_mean_result.get("summary", {}) if self.best_mean_result else {}
        worst_summary = self.best_worst_result.get("summary", {}) if self.best_worst_result else {}
        since_improvement = (
            max(0.0, now - self._last_improvement_at)
            if self._last_improvement_at is not None
            else None
        )
        improvement_elapsed = (
            max(0.0, self._last_improvement_at - self.started)
            if self._last_improvement_at is not None
            else None
        )
        return ProgressSnapshot(
            completed=self.completed,
            total=self.total,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            eval_rate=rate,
            best_mean_iou=(
                float(mean_summary["mean_iou"]) if "mean_iou" in mean_summary else None
            ),
            worst_page_iou=(
                float(worst_summary["minimum_iou"])
                if "minimum_iou" in worst_summary
                else None
            ),
            failures=self.failures,
            evaluating=self.evaluating,
            last_improvement_seconds=since_improvement,
            last_improvement_elapsed_seconds=improvement_elapsed,
        )

    @staticmethod
    def _percent(completed: int, total: int) -> str:
        if total <= 0:
            return "0.0%"
        percent = 100.0 * completed / total
        return f"{percent:.2f}%" if 0 < percent < 1 else f"{percent:.1f}%"

    def emit(self, *, force: bool = False) -> bool:
        now = self.clock()
        if not force and now - self.last_emit < self.interval_seconds:
            return False
        snap = self.snapshot()
        complete = f"{snap.completed}/{snap.total} {self._percent(snap.completed, snap.total)}"
        widths = self.COLUMN_WIDTHS
        rate = f"{snap.eval_rate:.2f}/s" if snap.eval_rate is not None else "--"
        best = f"{snap.best_mean_iou:.4f}" if snap.best_mean_iou is not None else "--"
        worst = f"{snap.worst_page_iou:.4f}" if snap.worst_page_iou is not None else "--"
        eta = _duration(snap.eta_seconds) if snap.eta_seconds is not None else "TBD"
        last_improvement = (
            _duration(snap.last_improvement_elapsed_seconds)
            if snap.last_improvement_elapsed_seconds is not None
            else "--"
        )
        print(
            f"{_duration(snap.elapsed_seconds):<{widths['elapsed']}}  "
            f"{eta:<{widths['eta']}}  "
            f"{complete:<{widths['complete']}}  "
            f"{rate:>{widths['rate']}}  "
            f"{best:>{widths['best']}}  "
            f"{worst:>{widths['worst']}}  "
            f"{snap.failures:>{widths['failures']}d}  "
            f"{last_improvement:<{widths['last_improvement']}}  "
            f"{snap.evaluating}",
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
            # The final row describes a completed run, not an evaluation still in flight.
            self.evaluating = "--"
            self.emit(force=True)
            return self.snapshot()
