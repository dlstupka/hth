from __future__ import annotations

import json
from pathlib import Path

from hth.runtime_store import order_configs, update_runtime_index


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
    payload = json.loads((tmp_path / "runtime-index.json").read_text(encoding="utf-8"))
    assert payload["latest"]["d"]["wall_clock_seconds"] == 8
    assert len(payload["observations"]) == 2
