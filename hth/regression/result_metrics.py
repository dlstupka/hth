from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

SUCCESS_STATUSES = {"ok", "success"}


def aggregate_page_metrics(pages: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Canonical Golden Set IoU metrics for a parameter set."""
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
            "page_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "mean_iou": 0.0,
            "mean_iou_success": 0.0,
            "minimum_iou": 0.0,
            "stddev_iou": 0.0,
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
    """Normalize legacy persisted metrics to the canonical definitions."""
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
    """Normalize winner/baseline metrics in a regression summary."""
    if not isinstance(summary, dict):
        return summary
    for key in ("winner", "baseline"):
        result = summary.get(key)
        if isinstance(result, dict) and isinstance(result.get("summary"), dict):
            normalize_result_metrics(result["summary"])
    return summary
