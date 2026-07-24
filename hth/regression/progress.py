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


def _evaluation_time(milliseconds: Any) -> str:
    try:
        value = float(milliseconds)
    except (TypeError, ValueError):
        return "--"
    if value < 1000:
        return f"{value:.1f}ms"
    seconds = value / 1000.0
    if seconds < 60:
        return f"{seconds:.1f}s"
    return _duration(seconds)


def _ranking_key(result: dict[str, Any]) -> tuple[float, float, int, float]:
    summary = result.get("summary", {})
    edge = summary.get("mean_edge_error_px")
    return (
        -float(summary.get("mean_iou", 0.0) or 0.0),
        -float(summary.get("minimum_iou", 0.0) or 0.0),
        int(summary.get("failure_count", 0) or 0),
        float(edge) if edge is not None else float("inf"),
    )


@dataclass
class ProgressSnapshot:
    completed: int
    total: int
    elapsed_seconds: float
    eta_seconds: float | None
    eval_rate: float | None
    current_mean_iou: float | None
    best_mean_iou: float | None
    current_minimum_page_iou: float | None
    best_minimum_page_iou: float | None
    current_stddev_iou: float | None
    best_stddev_iou: float | None
    current_evaluation_ms: float | None
    failures: int
    evaluating: str
    mean_iou_improvements: int
    minimum_iou_improvements: int
    stddev_improvements: int
    winner_changes: int
    parameter_sets_with_improvements: int
    last_improvement_seconds: float | None
    last_improvement_elapsed_seconds: float | None


class ProgressReporter:
    """Emit one fixed-width heartbeat line per interval plus sparse milestones."""

    COLUMN_WIDTHS = {
        "elapsed": 8,
        "eta": 8,
        "progress": 8,
        "percent": 6,
        "rate": 7,
        "average": 7,
        "average_best": 7,
        "minimum": 7,
        "minimum_best": 7,
        "stddev": 7,
        "stddev_best": 7,
        "failures": 4,
        "time": 9,
    }
    HEADER_TOP = (
        f"{'Elapsed':<{COLUMN_WIDTHS['elapsed']}}  "
        f"{'ETA':<{COLUMN_WIDTHS['eta']}}  "
        f"{'Progress':<{COLUMN_WIDTHS['progress']}}  "
        f"{'%':>{COLUMN_WIDTHS['percent']}}  "
        f"{'Rate':>{COLUMN_WIDTHS['rate']}}  "
        f"{'Avg':>{COLUMN_WIDTHS['average']}}  "
        f"{'Best':>{COLUMN_WIDTHS['average_best']}}  "
        f"{'Min':>{COLUMN_WIDTHS['minimum']}}  "
        f"{'Best':>{COLUMN_WIDTHS['minimum_best']}}  "
        f"{'':>{COLUMN_WIDTHS['stddev']}}  "
        f"{'Best':>{COLUMN_WIDTHS['stddev_best']}}  "
        f"{'Fail':>{COLUMN_WIDTHS['failures']}}  "
        f"{'Eval':>{COLUMN_WIDTHS['time']}}  "
        "Parameter"
    )
    HEADER_BOTTOM = (
        f"{'':<{COLUMN_WIDTHS['elapsed']}}  "
        f"{'':<{COLUMN_WIDTHS['eta']}}  "
        f"{'':<{COLUMN_WIDTHS['progress']}}  "
        f"{'':>{COLUMN_WIDTHS['percent']}}  "
        f"{'':>{COLUMN_WIDTHS['rate']}}  "
        f"{'IoU':>{COLUMN_WIDTHS['average']}}  "
        f"{'IoU':>{COLUMN_WIDTHS['average_best']}}  "
        f"{'IoU':>{COLUMN_WIDTHS['minimum']}}  "
        f"{'IoU':>{COLUMN_WIDTHS['minimum_best']}}  "
        f"{'SD':>{COLUMN_WIDTHS['stddev']}}  "
        f"{'SD':>{COLUMN_WIDTHS['stddev_best']}}  "
        f"{'':>{COLUMN_WIDTHS['failures']}}  "
        f"{'Time':>{COLUMN_WIDTHS['time']}}  "
        "Set"
    )
    HEADER = HEADER_TOP + "\n" + HEADER_BOTTOM

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
        self.current_result: dict[str, Any] | None = None
        self.best_mean_result: dict[str, Any] | None = None
        self.best_worst_result: dict[str, Any] | None = None
        self.best_stddev_result: dict[str, Any] | None = None
        self.overall_best_result: dict[str, Any] | None = None
        self.baseline_mean_iou: float | None = None
        self.baseline_surpassed = False
        self.evaluating = "--"
        self.mean_iou_improvements = 0
        self.minimum_iou_improvements = 0
        self.stddev_improvements = 0
        self.winner_changes = 0
        self.parameter_sets_with_improvements = 0
        self._last_improvement_at: float | None = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_progress_at = self.started
        self._search_started_at: float | None = None
        self._last_stall_warning_at: float | None = None

    def start(self) -> None:
        print("Regression Progress", file=self.stream)
        print(self.HEADER_TOP, file=self.stream)
        print(self.HEADER_BOTTOM, file=self.stream)
        print("-" * max(len(self.HEADER_TOP), len(self.HEADER_BOTTOM)), file=self.stream)
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
                        f"{_duration(now - self.started)} >>> No forward progress for "
                        f"{_duration(stalled_for)}; detector evaluation may be stalled",
                        file=self.stream,
                        flush=True,
                    )
                    self._last_stall_warning_at = now
                self.emit(force=True)

    @staticmethod
    def _normalize_profile(profile: str) -> str:
        return profile[3:] if profile.startswith("ps:") else profile

    @classmethod
    def _profile_name(cls, result: dict[str, Any], profile: str | None) -> str:
        value = profile or str(result.get("parameter_set_id", "unknown"))[:8]
        return cls._normalize_profile(value)

    def announce(
        self,
        labels: str | list[str],
        profile: str | None = None,
        *,
        emit_status: bool = True,
    ) -> None:
        """Emit sparse milestone notes without forcing them into table columns."""
        milestone_labels = [labels] if isinstance(labels, str) else list(labels)
        if not milestone_labels:
            return
        now = self.clock()
        suffix = f" {profile}" if profile else ""
        for label in milestone_labels:
            print(f"{_duration(now - self.started)} >>> {label}{suffix}", file=self.stream)
        if emit_status:
            self.emit(force=True)
        self.stream.flush()

    def begin_evaluation(self, profile: str) -> None:
        with self._lock:
            self.evaluating = self._normalize_profile(profile)

    def observe_baseline(self, result: dict[str, Any]) -> None:
        """Seed best-so-far values without counting baseline as an improvement."""
        with self._lock:
            now = self.clock()
            summary = result.get("summary", {})
            self.current_result = result
            self.best_mean_result = result
            self.best_worst_result = result
            self.best_stddev_result = result
            self.overall_best_result = result
            self.baseline_mean_iou = float(summary.get("mean_iou", 0.0) or 0.0)
            self.failures += int(summary.get("failure_count", 0) or 0)
            self.evaluating = "baseline"
            self._last_progress_at = now
            self._search_started_at = now
            self.emit(force=True)

    def observe(self, result: dict[str, Any], profile: str | None = None) -> None:
        with self._lock:
            self.completed += 1
            now = self.clock()
            self._last_progress_at = now
            summary = result.get("summary", {})
            self.failures += int(summary.get("failure_count", 0) or 0)
            mean_iou = float(summary.get("mean_iou", 0.0) or 0.0)
            worst_iou = float(summary.get("minimum_iou", 0.0) or 0.0)
            stddev_iou = float(summary.get("stddev_iou", 0.0) or 0.0)
            self.current_result = result
            profile_name = self._profile_name(result, profile)

            old_best_mean = float(self.best_mean_result.get("summary", {}).get("mean_iou", 0.0) or 0.0) if self.best_mean_result else None
            old_best_worst = float(self.best_worst_result.get("summary", {}).get("minimum_iou", 0.0) or 0.0) if self.best_worst_result else None
            old_best_stddev = float(self.best_stddev_result.get("summary", {}).get("stddev_iou", 0.0) or 0.0) if self.best_stddev_result else None

            new_best_mean = old_best_mean is None or mean_iou > old_best_mean
            new_best_worst = old_best_worst is None or worst_iou > old_best_worst
            new_best_stddev = old_best_stddev is None or stddev_iou < old_best_stddev
            new_overall_winner = self.overall_best_result is None or _ranking_key(result) < _ranking_key(self.overall_best_result)

            if new_best_mean:
                self.best_mean_result = result
                self.mean_iou_improvements += 1
            if new_best_worst:
                self.best_worst_result = result
                self.minimum_iou_improvements += 1
            if new_best_stddev:
                self.best_stddev_result = result
                self.stddev_improvements += 1
            if new_overall_winner:
                self.overall_best_result = result
                self.winner_changes += 1
            if new_best_mean or new_best_worst or new_best_stddev:
                self.parameter_sets_with_improvements += 1
                self._last_improvement_at = now

            milestones: list[str] = []
            if new_best_mean:
                milestones.append("New best average page IoU")
            if new_best_worst:
                milestones.append("New minimum page IoU")
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
        search_elapsed = max(0.0, now - self._search_started_at) if self._search_started_at is not None else 0.0
        rate = self.completed / search_elapsed if search_elapsed > 0 and self.completed else None
        remaining = max(0, self.total - self.completed)
        eta = remaining / rate if rate else None
        current_summary = self.current_result.get("summary", {}) if self.current_result else {}
        best_mean_summary = self.best_mean_result.get("summary", {}) if self.best_mean_result else {}
        best_worst_summary = self.best_worst_result.get("summary", {}) if self.best_worst_result else {}
        best_stddev_summary = self.best_stddev_result.get("summary", {}) if self.best_stddev_result else {}
        since_improvement = max(0.0, now - self._last_improvement_at) if self._last_improvement_at is not None else None
        improvement_elapsed = max(0.0, self._last_improvement_at - self.started) if self._last_improvement_at is not None else None
        evaluation_ms = current_summary.get("wall_ms", current_summary.get("elapsed_ms_total"))
        try:
            current_evaluation_ms = float(evaluation_ms) if evaluation_ms is not None else None
        except (TypeError, ValueError):
            current_evaluation_ms = None
        return ProgressSnapshot(
            completed=self.completed,
            total=self.total,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            eval_rate=rate,
            current_mean_iou=float(current_summary["mean_iou"]) if "mean_iou" in current_summary else None,
            best_mean_iou=float(best_mean_summary["mean_iou"]) if "mean_iou" in best_mean_summary else None,
            current_minimum_page_iou=float(current_summary["minimum_iou"]) if "minimum_iou" in current_summary else None,
            best_minimum_page_iou=float(best_worst_summary["minimum_iou"]) if "minimum_iou" in best_worst_summary else None,
            current_stddev_iou=float(current_summary["stddev_iou"]) if "stddev_iou" in current_summary else None,
            best_stddev_iou=float(best_stddev_summary["stddev_iou"]) if "stddev_iou" in best_stddev_summary else None,
            current_evaluation_ms=current_evaluation_ms,
            failures=self.failures,
            evaluating=self.evaluating,
            mean_iou_improvements=self.mean_iou_improvements,
            minimum_iou_improvements=self.minimum_iou_improvements,
            stddev_improvements=self.stddev_improvements,
            winner_changes=self.winner_changes,
            parameter_sets_with_improvements=self.parameter_sets_with_improvements,
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
        progress = f"{snap.completed}/{snap.total}"
        percent = self._percent(snap.completed, snap.total)
        widths = self.COLUMN_WIDTHS
        rate = f"{snap.eval_rate:.3f}/s" if snap.eval_rate is not None else "--"
        average = f"{snap.current_mean_iou:.4f}" if snap.current_mean_iou is not None else "--"
        average_best = f"{snap.best_mean_iou:.4f}" if snap.best_mean_iou is not None else "--"
        minimum = f"{snap.current_minimum_page_iou:.4f}" if snap.current_minimum_page_iou is not None else "--"
        minimum_best = f"{snap.best_minimum_page_iou:.4f}" if snap.best_minimum_page_iou is not None else "--"
        stddev = f"{snap.current_stddev_iou:.4f}" if snap.current_stddev_iou is not None else "--"
        stddev_best = f"{snap.best_stddev_iou:.4f}" if snap.best_stddev_iou is not None else "--"
        eta = _duration(snap.eta_seconds) if snap.eta_seconds is not None else "TBD"
        evaluation_time = _evaluation_time(snap.current_evaluation_ms)
        print(
            f"{_duration(snap.elapsed_seconds):<{widths['elapsed']}}  "
            f"{eta:<{widths['eta']}}  "
            f"{progress:<{widths['progress']}}  "
            f"{percent:>{widths['percent']}}  "
            f"{rate:>{widths['rate']}}  "
            f"{average:>{widths['average']}}  "
            f"{average_best:>{widths['average_best']}}  "
            f"{minimum:>{widths['minimum']}}  "
            f"{minimum_best:>{widths['minimum_best']}}  "
            f"{stddev:>{widths['stddev']}}  "
            f"{stddev_best:>{widths['stddev_best']}}  "
            f"{snap.failures:>{widths['failures']}d}  "
            f"{evaluation_time:>{widths['time']}}  "
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
            return self.snapshot()
