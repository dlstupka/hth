from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

SUCCESS_STATUSES = {"ok", "success"}


def aggregate_page_metrics(pages: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Canonical Golden Set metrics. Failed pages contribute IoU=0."""
    page_list = list(pages)
    successful = [
        page for page in page_list
        if str(page.get("status") or "").strip().lower() in SUCCESS_STATUSES
    ]
    all_ious = [float(page.get("iou") or 0.0) for page in page_list]
    successful_ious = [float(page.get("iou") or 0.0) for page in successful]
    page_count = len(page_list)
    success_count = len(successful)
    failure_count = page_count - success_count
    if not page_count:
        return {
            "page_count": 0, "success_count": 0, "failure_count": 0,
            "mean_iou": 0.0, "mean_iou_success": 0.0,
            "minimum_iou": 0.0, "stddev_iou": 0.0,
        }
    return {
        "page_count": page_count,
        "success_count": success_count,
        "failure_count": failure_count,
        "mean_iou": round(sum(all_ious) / page_count, 8),
        "mean_iou_success": round(sum(successful_ious) / success_count, 8) if success_count else 0.0,
        "minimum_iou": round(min(all_ious), 8),
        "stddev_iou": round(statistics.pstdev(all_ious), 8),
    }


def normalize_result_metrics(stats: dict[str, Any]) -> dict[str, Any]:
    """Adapt legacy success-only metrics to canonical all-page metrics once."""
    if not isinstance(stats, dict) or "mean_iou_success" in stats:
        return stats
    try:
        page_count = int(stats.get("page_count") or 0)
        success_count = int(stats.get("success_count") or 0)
        failure_count = int(stats.get("failure_count") or 0)
        success_mean = float(stats.get("mean_iou") or 0.0)
        success_stddev = float(stats.get("stddev_iou") or 0.0)
    except (TypeError, ValueError):
        return stats
    if page_count <= 0 or success_count + failure_count != page_count:
        return stats

    stats["mean_iou_success"] = success_mean
    if not failure_count:
        return stats

    full_mean = success_mean * success_count / page_count
    success_second_moment = success_stddev ** 2 + success_mean ** 2
    full_second_moment = success_second_moment * success_count / page_count
    full_variance = max(0.0, full_second_moment - full_mean ** 2)
    stats["mean_iou"] = round(full_mean, 8)
    stats["minimum_iou"] = 0.0
    stats["stddev_iou"] = round(math.sqrt(full_variance), 8)
    return stats


def normalize_summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return summary
    for key in ("winner", "baseline"):
        result = summary.get(key)
        if isinstance(result, dict) and isinstance(result.get("summary"), dict):
            normalize_result_metrics(result["summary"])
    return summary


def result_metric_view(stats: dict[str, Any] | None) -> dict[str, Any]:
    """Return the one report-facing representation of result metrics."""
    normalized = normalize_result_metrics(dict(stats or {}))
    mean = normalized.get("mean_iou")
    return {
        "mean_iou": mean,
        "mean_iou_success": normalized.get("mean_iou_success", mean),
        "minimum_iou": normalized.get("minimum_iou"),
        "stddev_iou": normalized.get("stddev_iou"),
        "failure_count": normalized.get("failure_count", "unknown"),
    }


def calibration_metric_view(
    payload: dict[str, Any],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical metrics for a persisted calibration, including legacy records.

    A summary winner is page-derived evidence and therefore wins over older
    duplicated selection fields. New records should agree; legacy records are
    normalized here rather than in report code.
    """
    summary = normalize_summary_metrics(dict(summary or {}))
    winner = summary.get("winner") if isinstance(summary.get("winner"), dict) else {}
    winner_stats = result_metric_view(winner.get("summary") if isinstance(winner, dict) else {})
    if winner_stats.get("mean_iou") is not None:
        return winner_stats

    selection = payload.get("detector_selection_intelligence")
    selection = selection if isinstance(selection, dict) else {}
    landscape = payload.get("landscape")
    landscape = landscape if isinstance(landscape, dict) else {}
    mean = selection.get("best_avg_iou", landscape.get("best_mean_iou"))
    return {
        "mean_iou": mean,
        "mean_iou_success": selection.get("avg_iou_success", mean),
        "minimum_iou": selection.get("minimum_iou"),
        "stddev_iou": selection.get("stddev_iou"),
        "failure_count": selection.get("failure_count", "unknown"),
    }
