from __future__ import annotations

import json
from pathlib import Path

from hth.runtime_store import observation_from_run, order_configs, update_runtime_index


def _config(path: Path, detector: str) -> Path:
    path.write_text(json.dumps({"detector": detector}), encoding="utf-8")
    return path


def test_lpt_orders_longest_first_and_unknown_conservatively(tmp_path: Path) -> None:
    configs = [_config(tmp_path / "fast.json", "fast"), _config(tmp_path / "slow.json", "slow"), _config(tmp_path / "new.json", "new")]
    runtime = tmp_path / "runtime-index.json"
    runtime.write_text(json.dumps({"observations": [
        {"detector_id": "fast", "wall_clock_seconds": 2, "mode": "smoke", "resolved_strategy": "exhaustive", "configured_threads": 4, "max_dimension": 1800, "golden_set_sha256": "gold", "observed_at_utc": "2026-08-01T00:00:00Z"},
        {"detector_id": "slow", "wall_clock_seconds": 20, "mode": "smoke", "resolved_strategy": "exhaustive", "configured_threads": 4, "max_dimension": 1800, "golden_set_sha256": "gold", "observed_at_utc": "2026-08-01T00:00:00Z"},
    ]}), encoding="utf-8")
    rows = order_configs(configs, loading_strategy="lpt", runtime_index_path=runtime, calibration_index_path=None, mode="smoke", search_strategy="exhaustive", threads=4, max_dimension=1800, golden_set_sha256="gold", runner_label="")
    assert [row[0].stem for row in rows][:2] == ["slow", "new"]
    assert rows[-1][0].stem == "fast"


def test_ranked_orders_best_detector_first(tmp_path: Path) -> None:
    configs = [_config(tmp_path / "a.json", "a"), _config(tmp_path / "b.json", "b")]
    calibration = tmp_path / "calibration-index.json"
    calibration.write_text(json.dumps({"entries": [
        {"detector_id": "a", "golden_set_sha256": "gold", "created_at_utc": "1", "selection": {"best_avg_iou": 0.8}},
        {"detector_id": "b", "golden_set_sha256": "gold", "created_at_utc": "1", "selection": {"best_avg_iou": 0.9}},
    ]}), encoding="utf-8")
    rows = order_configs(configs, loading_strategy="ranked", runtime_index_path=None, calibration_index_path=calibration, mode="smoke", search_strategy="exhaustive", threads=1, max_dimension=1800, golden_set_sha256="gold", runner_label="")
    assert [row[0].stem for row in rows] == ["b", "a"]


def test_runtime_index_keeps_latest_summary(tmp_path: Path) -> None:
    update_runtime_index(tmp_path, [
        {"observation_id": "1", "detector_id": "d", "wall_clock_seconds": 10, "mode": "smoke", "resolved_strategy": "exhaustive", "configured_threads": 1, "observed_at_utc": "2026-08-01T00:00:00Z"},
        {"observation_id": "2", "detector_id": "d", "wall_clock_seconds": 8, "mode": "smoke", "resolved_strategy": "exhaustive", "configured_threads": 4, "observed_at_utc": "2026-08-01T01:00:00Z"},
    ])
    payload = json.loads((tmp_path / "indexes" / "runtime-index.json").read_text(encoding="utf-8"))
    assert payload["latest"]["d"]["wall_clock_seconds"] == 8
    assert len(payload["observations"]) == 2


def _runtime_run(path: Path, detector: str, run_id: str) -> Path:
    (path / "reports").mkdir(parents=True)
    config = path / f"{detector}.json"
    config.write_text(json.dumps({"detector": detector}), encoding="utf-8")
    (path / "RUN-INFO.json").write_text(json.dumps({
        "run_id": run_id,
        "detector": detector,
        "detector_config": str(config),
        "elapsed_seconds": 10.0,
        "threads": 2,
        "golden_set_sha256": "gold",
        "strategy": "exhaustive",
    }), encoding="utf-8")
    (path / "parameters.json").write_text(json.dumps({
        "detector": detector,
        "strategy": "exhaustive",
        "threads": 2,
        "max_dimension": 1800,
    }), encoding="utf-8")
    (path / "reports" / "summary.json").write_text(json.dumps({
        "detector": detector,
        "parameter_set_count": 10,
        "page_ordinals": [1],
        "parameter_space": {"actual_parameter_sets": 10, "golden_set_pages": 1},
        "progress": {"average_eval_rate": 1.0},
    }), encoding="utf-8")
    return path


def test_concurrent_detector_run_ids_do_not_collide(tmp_path: Path) -> None:
    first = _runtime_run(tmp_path / "grabcut" / "run-same", "grabcut", "run-20260807-120000")
    second = _runtime_run(tmp_path / "contour" / "run-same", "contour", "run-20260807-120000")
    build = {"github_run_id": "243", "mode": "smoke"}
    observations = [observation_from_run(first, build=build), observation_from_run(second, build=build)]

    assert observations[0]["observation_id"] != observations[1]["observation_id"]
    update_runtime_index(tmp_path / "results", observations)
    payload = json.loads((tmp_path / "results" / "indexes" / "runtime-index.json").read_text(encoding="utf-8"))
    assert {row["detector_id"] for row in payload["observations"]} == {"grabcut", "contour"}


def test_order_supplements_missing_runtime_detector_from_persisted_calibration(tmp_path: Path) -> None:
    configs = [_config(tmp_path / "fast.json", "fast"), _config(tmp_path / "missing.json", "missing")]
    runtime = tmp_path / "runtime-index.json"
    runtime.write_text(json.dumps({"observations": [
        {"detector_id": "fast", "wall_clock_seconds": 2, "mode": "smoke", "resolved_strategy": "exhaustive", "configured_threads": 2, "max_dimension": 1800, "golden_set_sha256": "gold", "observed_at_utc": "2026-08-01T00:00:00Z"}
    ]}), encoding="utf-8")

    record = tmp_path / "records" / "missing"
    record.mkdir(parents=True)
    (record / "RUN-INFO.json").write_text(json.dumps({
        "run_id": "run-missing",
        "detector": "missing",
        "elapsed_seconds": 20,
        "threads": 2,
        "golden_set_sha256": "gold",
        "strategy": "exhaustive",
        "github_runner_labels": ["github-hosted"],
    }), encoding="utf-8")
    calibration = tmp_path / "calibration-index.json"
    calibration.write_text(json.dumps({"entries": [{
        "calibration_id": "missing-1",
        "detector_id": "missing",
        "golden_set_sha256": "gold",
        "record_path": "records/missing",
        "created_at_utc": "2026-08-01T01:00:00Z",
        "build": {"mode": "smoke"},
    }]}), encoding="utf-8")

    rows = order_configs(
        configs, loading_strategy="lpt", runtime_index_path=runtime, calibration_index_path=calibration,
        mode="smoke", search_strategy="exhaustive", threads=2, max_dimension=1800,
        golden_set_sha256="gold", runner_label="github-hosted",
    )
    by_detector = {row[0].stem: row for row in rows}
    assert by_detector["missing"][1] == 20
    assert by_detector["missing"][2].startswith("runtime-index:")
