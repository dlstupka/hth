from __future__ import annotations

import csv
import gzip
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
    assert plan_shards(4 * 3600, runner_label="e7k", requested_threads="auto").threads == 64
    assert plan_shards(4 * 3600, runner_label="e9k", requested_threads="auto").threads == 64


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


def test_reconstructed_rows_round_trip_through_persisted_gzip(tmp_path) -> None:
    raw = tmp_path / "results.csv"
    _write_raw_row(raw, status="ok", iou=0.9)
    compressed = raw.with_name("results.csv.gz")
    with raw.open("rb") as source, gzip.open(compressed, "wb") as target:
        target.write(source.read())

    result = _results_from_raw(compressed)[0]

    assert result["parameter_set_id"] == "abc"
    assert result["pages"][0]["status"] == "ok"
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


def test_reconstructed_result_preserves_local_completion_without_elapsed(tmp_path) -> None:
    raw = tmp_path / "results.csv"
    _write_raw_row(raw)
    rows = raw.read_text(encoding="utf-8").splitlines()
    header = rows[0].split(",")
    values = rows[1].split(",")
    elapsed_index = header.index("completion_elapsed_seconds")
    values[elapsed_index] = ""
    raw.write_text(rows[0] + "\n" + ",".join(values) + "\n", encoding="utf-8")
    observation = _results_from_raw(raw)[0]["search_observation"]
    assert observation["completion_index"] == 3
    assert observation["elapsed_seconds"] is None


def test_reconstructed_mean_includes_failed_pages_as_zero(tmp_path) -> None:
    raw = tmp_path / "results.csv"
    _write_raw_row(raw, status="ok", iou=0.9638)
    rows = list(csv.DictReader(raw.open(encoding="utf-8")))
    fields = list(rows[0].keys())
    with raw.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(rows[0])
        for ordinal in range(2, 6):
            failed = dict(rows[0])
            failed["global_ordinal"] = ordinal
            failed["label"] = f"page-{ordinal}"
            failed["status"] = "no_candidate"
            failed["iou"] = 0.0
            failed["edge_error_mean_px"] = ""
            failed["edge_error_maximum_px"] = ""
            failed["predicted_bbox_json"] = "null"
            writer.writerow(failed)
    summary = _results_from_raw(raw)[0]["summary"]
    assert summary["success_count"] == 1
    assert summary["failure_count"] == 4
    assert summary["mean_iou"] == 0.19276
    assert summary["mean_iou_success"] == 0.9638
    assert summary["minimum_iou"] == 0.0
