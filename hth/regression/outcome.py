"""Canonical validity and winner-selection policy for detector regressions."""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterable

from hth.domain.result_metrics import SUCCESS_STATUSES


NO_VALID_MEASUREMENTS = "no_valid_measurements"


def _pages(result: dict[str, Any]) -> list[dict[str, Any]]:
    pages = result.get("pages")
    return [page for page in pages if isinstance(page, dict)] if isinstance(pages, list) else []


def result_success_count(result: dict[str, Any]) -> int:
    """Return the number of valid page measurements represented by a result."""
    pages = _pages(result)
    if pages:
        return sum(
            str(page.get("status") or "").strip().lower() in SUCCESS_STATUSES
            for page in pages
        )
    summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
    try:
        return max(0, int(summary.get("success_count") or 0))
    except (TypeError, ValueError):
        return 0


def is_winner_eligible(result: dict[str, Any]) -> bool:
    """A parameter set can win only after producing a valid page measurement."""
    if result_success_count(result) > 0:
        return True
    # Historical intelligence fixtures may retain aggregate metrics without
    # page rows. A positive aggregate is durable evidence of a past valid
    # measurement; zero-only aggregates are deliberately not inferred valid.
    if not _pages(result):
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        try:
            return float(summary.get("mean_iou") or 0.0) > 0.0
        except (TypeError, ValueError):
            return False
    return False


def _failure_reason(page: dict[str, Any]) -> str:
    candidate = page.get("candidate") if isinstance(page.get("candidate"), dict) else {}
    diagnostics = candidate.get("diagnostics") if isinstance(candidate.get("diagnostics"), dict) else {}
    error = page.get("error") if isinstance(page.get("error"), dict) else {}
    return str(diagnostics.get("reason") or error.get("type") or page.get("status") or "unknown")


def classify_measurements(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Reduce all parameter/page evidence to one detector-agnostic run outcome."""
    result_list = list(results)
    pages = [page for result in result_list for page in _pages(result)]
    successful = [
        page for page in pages
        if str(page.get("status") or "").strip().lower() in SUCCESS_STATUSES
    ]
    positive = [page for page in successful if float(page.get("iou") or 0.0) > 0.0]
    reasons = Counter(_failure_reason(page) for page in pages if page not in successful)
    eligible_count = sum(is_winner_eligible(result) for result in result_list)

    legacy_positive = not pages and any(is_winner_eligible(result) for result in result_list)
    if legacy_positive:
        status = "measured"
        reason = "Regression contains retained positive aggregate measurements."
        informative = True
        terminal_success = True
    elif not successful:
        status = NO_VALID_MEASUREMENTS
        reason = "No page evaluation produced a valid detector measurement."
        informative = False
        terminal_success = False
    elif not positive:
        status = "no_overlap_signal"
        reason = "Valid detector measurements were produced, but none overlapped an approved Golden Set bounding box."
        informative = False
        terminal_success = True
    else:
        status = "measured"
        reason = "Regression contains valid positive-overlap measurements."
        informative = True
        terminal_success = True

    return {
        "status": status,
        "reason": reason,
        "informative": informative,
        "terminal_success": terminal_success,
        "parameter_set_count": len(result_list),
        "eligible_parameter_set_count": eligible_count,
        "page_evaluation_count": len(pages),
        "successful_page_evaluation_count": len(successful),
        "positive_iou_page_evaluation_count": len(positive),
        # Compatibility aliases retained for calibration-intelligence readers.
        "page_evaluations": len(pages),
        "successful_page_evaluations": len(successful),
        "positive_iou_page_evaluations": len(positive),
        "failure_reason_counts": dict(reasons.most_common()),
    }


def reduce_regression_outcome(
    results: Iterable[dict[str, Any]],
    *,
    ranking_key: Callable[[dict[str, Any]], Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    """Return ordered evidence, eligible rankings, nullable winner, and state."""
    ordered = sorted(results, key=ranking_key)
    ranked = [result for result in ordered if is_winner_eligible(result)]
    for result in ordered:
        result["rank"] = None
    for rank, result in enumerate(ranked, 1):
        result["rank"] = rank
    state = classify_measurements(ordered)
    winner = ranked[0] if ranked else None
    return ordered, ranked, winner, state


def unavailable_winner_page_report(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": False,
        "reason": state.get("status", NO_VALID_MEASUREMENTS),
        "message": state.get("reason"),
        "pages": [],
    }
