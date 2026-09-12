"""Deterministic page-lane execution for feedback-directed regressions."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Iterable

from .metrics import bbox_iou, edge_errors
from .parameter_space import parameter_set_id
from hth.domain.result_metrics import aggregate_page_metrics
from hth.geometry.common import scale_bbox


NON_SHARDABLE_STRATEGIES = frozenset({"adaptive", "binary-refine"})


def evaluate_page(
    detector: Callable[..., Any], parameters: dict[str, Any], page: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one immutable page view and preserve failures as page evidence."""
    started = time.perf_counter()
    try:
        candidate = detector(
            image_bgr=page["image"], mask=page["mask"], parameters=parameters,
        )
        elapsed = (time.perf_counter() - started) * 1000
        if candidate.bbox is None:
            return {
                "global_ordinal": page["global_ordinal"], "label": page["label"],
                "layout_type": page["layout_type"],
                "status": candidate.status if candidate.status != "ok" else "no_candidate",
                "iou": 0.0, "edge_error_mean_px": None,
                "edge_error_maximum_px": None, "elapsed_ms": round(elapsed, 3),
                "candidate": asdict(candidate),
            }
        predicted = scale_bbox(
            candidate.bbox, 1.0 / page["scale"],
            page["original_width"], page["original_height"],
        )
        approved = page["approved_bbox"]
        errors = edge_errors(predicted, approved)
        return {
            "global_ordinal": page["global_ordinal"], "label": page["label"],
            "layout_type": page["layout_type"], "status": "ok",
            "approved_bbox": approved, "predicted_bbox": predicted,
            "iou": round(bbox_iou(predicted, approved), 8),
            "edge_errors": errors,
            "edge_error_mean_px": round(float(errors["mean"]), 3),
            "edge_error_maximum_px": int(errors["maximum"]),
            "elapsed_ms": round(elapsed, 3), "candidate": asdict(candidate),
        }
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "global_ordinal": page["global_ordinal"], "label": page["label"],
            "layout_type": page["layout_type"], "status": "error", "iou": 0.0,
            "edge_error_mean_px": None, "edge_error_maximum_px": None,
            "elapsed_ms": round(elapsed, 3),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def reduce_pages(
    parameters: dict[str, Any], page_results: Iterable[dict[str, Any]],
    *, wall_ms: float,
) -> dict[str, Any]:
    """Reduce a complete page set in canonical Golden Set ordinal order."""
    pages = sorted((dict(row) for row in page_results), key=lambda row: int(row["global_ordinal"]))
    successful = [row for row in pages if row["status"] == "ok"]
    edges = [float(row["edge_error_mean_px"]) for row in successful]
    summary = aggregate_page_metrics(pages)
    summary.update({
        "mean_edge_error_px": round(sum(edges) / len(edges), 3) if edges else None,
        "elapsed_ms_total": round(sum(float(row["elapsed_ms"]) for row in pages), 3),
        "wall_ms": round(float(wall_ms), 3),
    })
    return {
        "parameter_set_id": parameter_set_id(parameters),
        "parameters": parameters, "summary": summary, "pages": pages,
    }


def evaluate_set(
    detector: Callable[..., Any], parameters: dict[str, Any], pages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compatibility serial evaluator used by shardable search strategies."""
    started = time.perf_counter()
    results = [evaluate_page(detector, parameters, page) for page in pages]
    return reduce_pages(parameters, results, wall_ms=(time.perf_counter() - started) * 1000)


class GoldenSetCoordinator:
    """Own page-lane control and full-set barriers for one detector search."""

    def __init__(
        self, detector: Callable[..., Any], pages: list[dict[str, Any]], *,
        lanes: int, total_threads: int, threads_per_lane: int,
        diagnostics_path: Path | None = None, verbose: bool = False,
    ) -> None:
        if not pages:
            raise ValueError("Golden Set coordinator requires at least one page")
        if int(lanes) < 1 or int(total_threads) < 1 or int(threads_per_lane) < 1:
            raise ValueError("Golden Set lane and thread counts must be positive")
        if int(lanes) * int(threads_per_lane) > int(total_threads):
            raise ValueError("Golden Set lane reservation exceeds total threads")
        self.detector = detector
        self.pages = tuple(dict(page) for page in pages)
        self.lanes = min(int(lanes), len(self.pages))
        self.total_threads = int(total_threads)
        self.threads_per_lane = int(threads_per_lane)
        self.diagnostics_path = diagnostics_path
        self.verbose = bool(verbose)
        self._lock = threading.Lock()
        self._started = time.perf_counter()
        self._rounds = 0
        self._candidates = 0
        self._pages = 0
        self._page_seconds = 0.0
        self._active = 0
        self._peak_active = 0
        if diagnostics_path is not None:
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostics_path.write_text("", encoding="utf-8")

    def _record_page(
        self, *, candidate: str, lane: int, result: dict[str, Any],
    ) -> None:
        elapsed = float(result.get("elapsed_ms") or 0.0) / 1000.0
        with self._lock:
            self._pages += 1
            self._page_seconds += elapsed
            if self.verbose and self.diagnostics_path is not None:
                payload = {
                    "event": "page-finish", "candidate": candidate,
                    "lane": lane + 1, "golden_set_page": result.get("global_ordinal"),
                    "status": result.get("status"), "elapsed_seconds": round(elapsed, 6),
                }
                with self.diagnostics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _evaluate_partition(
        self, parameters: dict[str, Any], partition: tuple[dict[str, Any], ...], lane: int,
    ) -> tuple[list[dict[str, Any]], float, float]:
        started = time.perf_counter()
        with self._lock:
            self._active += 1
            self._peak_active = max(self._peak_active, self._active)
        candidate = parameter_set_id(parameters)[:12]
        try:
            rows = []
            for page in partition:
                result = evaluate_page(self.detector, parameters, page)
                self._record_page(candidate=candidate, lane=lane, result=result)
                rows.append(result)
            return rows, started, time.perf_counter()
        finally:
            with self._lock:
                self._active -= 1

    def evaluate_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Evaluate candidates with one complete-Golden-Set barrier per batch."""
        if not batch:
            return []
        round_started = time.perf_counter()
        partitions = tuple(
            tuple(self.pages[index::self.lanes]) for index in range(self.lanes)
        )
        indexed: list[list[dict[str, Any]]] = [[] for _ in batch]
        candidate_started: list[float | None] = [None] * len(batch)
        candidate_finished: list[float | None] = [None] * len(batch)
        jobs = [
            (candidate_index, lane, parameters, partition)
            for candidate_index, parameters in enumerate(batch)
            for lane, partition in enumerate(partitions)
            if partition
        ]
        workers = min(self.total_threads, len(jobs))
        if workers <= 1:
            for candidate_index, lane, parameters, partition in jobs:
                rows, started, finished = self._evaluate_partition(parameters, partition, lane)
                indexed[candidate_index].extend(rows)
                candidate_started[candidate_index] = started
                candidate_finished[candidate_index] = finished
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="golden-set") as executor:
                futures = {
                    executor.submit(self._evaluate_partition, parameters, partition, lane): candidate_index
                    for candidate_index, lane, parameters, partition in jobs
                }
                try:
                    for future in as_completed(futures):
                        candidate_index = futures[future]
                        rows, started, finished = future.result()
                        indexed[candidate_index].extend(rows)
                        prior_started = candidate_started[candidate_index]
                        prior_finished = candidate_finished[candidate_index]
                        candidate_started[candidate_index] = min(prior_started, started) if prior_started else started
                        candidate_finished[candidate_index] = max(prior_finished, finished) if prior_finished else finished
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        round_finished = time.perf_counter()
        with self._lock:
            self._rounds += 1
            self._candidates += len(batch)
        return [
            reduce_pages(
                parameters, indexed[index],
                wall_ms=(
                    ((candidate_finished[index] or round_finished)
                     - (candidate_started[index] or round_started)) * 1000
                ),
            )
            for index, parameters in enumerate(batch)
        ]

    def evaluate(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return self.evaluate_batch([parameters])[0]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = max(0.0, time.perf_counter() - self._started)
            utilization = (
                self._page_seconds / (elapsed * self.total_threads)
                if elapsed > 0 and self.total_threads > 0 else 0.0
            )
            return {
                "schema_version": "1.0", "lanes": self.lanes,
                "threads_per_lane": self.threads_per_lane,
                "reserved_threads": self.total_threads,
                "rounds": self._rounds, "candidate_parameter_sets": self._candidates,
                "page_evaluations": self._pages,
                "page_work_seconds": round(self._page_seconds, 6),
                "wall_seconds": round(elapsed, 6),
                "peak_active_lane_tasks": self._peak_active,
                "worker_utilization": round(min(1.0, utilization), 6),
            }
