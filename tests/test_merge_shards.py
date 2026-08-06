from __future__ import annotations

import csv
import json

from hth.regression.merge_shards import _results_from_raw
from hth.regression.sharding import plan_shards


def _write_raw_row(path, *, status: str = "ok", iou: float = 0.9) -> None:
    fields = [
        "run_id", "parameter_set_id", "profile", "rank", "completion_index", "completion_elapsed_seconds", "search_fraction", "global_ordinal", "label",
        "layout_type", "status", "iou", "left_error_px", "top_error_px",
        "right_error_px", "bottom_error_px", "edge_error_mean_px",
        "edge_error_maximum_px", "elapsed_ms", "approved_bbox_json",
        "predicted_bbox_json", "parameters_json", "error_type", "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "run_id": "run-1",
            "parameter_set_id": "abc",
            "profile": "baseline",
            "rank": 1,
            "completion_index": 3,
            "completion_elapsed_seconds": 12.0,
            "search_fraction": 0.3,
            "global_ordinal": 1,
            "label": "page-1",
            "layout_type": "single",
            "status": status,
            "iou": iou,
            "elapsed_ms": 12.5,
            "approved_bbox_json": json.dumps([0, 0, 10, 10]),
            "predicted_bbox_json": json.dumps([0, 0, 10, 10]),
            "parameters_json": json.dumps({"x": 1}),
        })


def test_shard_planner_caps_runner_threads() -> None:
    assert plan_shards(4 * 3600, runner_label="e7k", requested_threads="auto").threads == 48
    assert plan_shards(4 * 3600, runner_label="e9k", requested_threads="auto").threads == 32


def test_reconstructed_success_rows_have_canonical_optional_fields(tmp_path) -> None:
    raw = tmp_path / "results.csv"
    _write_raw_row(raw)
    result = _results_from_raw(raw)[0]
    page = result["pages"][0]
    assert page["error"] == {}
    assert page["warnings"] == []
    assert page["metadata"] == {}


def test_reconstructed_ok_rows_are_counted_as_success(tmp_path) -> None:
    raw = tmp_path / "results.csv"
    _write_raw_row(raw, status="ok", iou=0.9)
    result = _results_from_raw(raw)[0]
    assert result["summary"]["success_count"] == 1
    assert result["summary"]["failure_count"] == 0
    assert result["summary"]["mean_iou"] == 0.9


def test_reconstructed_result_preserves_completion_observation(tmp_path) -> None:
    raw = tmp_path / "results.csv"
    _write_raw_row(raw)
    result = _results_from_raw(raw)[0]
    observation = result["search_observation"]
    assert observation["completion_index"] == 3
    assert observation["parameter_set_number"] == 3
    assert observation["elapsed_seconds"] == 12.0
    assert observation["search_fraction"] == 0.3
