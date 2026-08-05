from __future__ import annotations

from hth.regression.merge_shards import _results_from_raw
from hth.regression.sharding import plan_shards


def test_shard_planner_caps_runner_threads() -> None:
    assert plan_shards(4 * 3600, runner_label="e7k", requested_threads="auto").threads == 48
    assert plan_shards(4 * 3600, runner_label="e9k", requested_threads="auto").threads == 32


def test_reconstructed_success_rows_have_canonical_optional_fields(tmp_path) -> None:
    raw = tmp_path / "results.csv"
    raw.write_text(
        "run_id,parameter_set_id,profile,rank,global_ordinal,label,layout_type,status,iou,"
        "left_error_px,top_error_px,right_error_px,bottom_error_px,edge_error_mean_px,"
        "edge_error_maximum_px,elapsed_ms,approved_bbox_json,predicted_bbox_json,"
        "parameters_json,error_type,error_message\n"
        'run-1,abc,baseline,1,1,page-1,single,success,0.9,,,,,,,'
        '12.5,"[0, 0, 10, 10]","[0, 0, 10, 10]","{\"x\": 1}",,\n',
        encoding="utf-8",
    )
    result = _results_from_raw(raw)[0]
    page = result["pages"][0]
    assert page["error"] == {}
    assert page["warnings"] == []
    assert page["metadata"] == {}
