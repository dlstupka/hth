from hth.regression.outcome import classify_measurements, reduce_regression_outcome
from hth.regression.reports import ranking_key


def _result(parameter_id, pages):
    successes = sum(page["status"] == "ok" for page in pages)
    ious = [float(page.get("iou", 0.0)) for page in pages]
    return {
        "parameter_set_id": parameter_id,
        "pages": pages,
        "summary": {
            "page_count": len(pages),
            "success_count": successes,
            "failure_count": len(pages) - successes,
            "mean_iou": sum(ious) / len(ious) if ious else 0.0,
            "minimum_iou": min(ious) if ious else 0.0,
            "stddev_iou": 0.0,
            "mean_edge_error_px": None,
        },
    }


def test_zero_valid_measurements_has_no_winner_or_numeric_ranks():
    results = [
        _result("a", [{"status": "no_candidate", "candidate": {"diagnostics": {"reason": "empty_mask"}}}]),
        _result("b", [{"status": "error", "error": {"type": "RuntimeError"}}]),
    ]

    ordered, ranked, winner, state = reduce_regression_outcome(results, ranking_key=ranking_key)

    assert len(ordered) == 2
    assert ranked == []
    assert winner is None
    assert all(result["rank"] is None for result in ordered)
    assert state["status"] == "no_valid_measurements"
    assert state["terminal_success"] is False
    assert state["failure_reason_counts"] == {"empty_mask": 1, "RuntimeError": 1}


def test_partial_valid_run_selects_only_from_eligible_parameter_sets():
    invalid = _result("invalid", [{"status": "error", "error": {"type": "RuntimeError"}}])
    valid = _result("valid", [{"status": "ok", "iou": 0.0}])

    ordered, ranked, winner, state = reduce_regression_outcome([invalid, valid], ranking_key=ranking_key)

    assert len(ordered) == 2
    assert ranked == [valid]
    assert winner is valid
    assert valid["rank"] == 1
    assert invalid["rank"] is None
    assert state["status"] == "no_overlap_signal"
    assert state["terminal_success"] is True


def test_measurement_state_counts_all_pages_across_parameter_sets():
    state = classify_measurements([
        _result("a", [{"status": "ok", "iou": 0.5}, {"status": "no_candidate"}]),
        _result("b", [{"status": "ok", "iou": 0.0}, {"status": "error", "error": {"type": "ValueError"}}]),
    ])

    assert state["status"] == "measured"
    assert state["eligible_parameter_set_count"] == 2
    assert state["successful_page_evaluation_count"] == 2
    assert state["positive_iou_page_evaluation_count"] == 1
