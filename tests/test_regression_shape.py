from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hth.regression_shape import RunnerProfile, parse_manual_shape, resolve_preferred_shape


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _row(*, detector: str, detector_sha: str, golden_sha: str, runner_name: str, cpu_model: str, logical: int, physical: int, pipelines: int, threads: int, rate: float) -> dict:
    return {
        "source": "execution-optimizer",
        "detector_id": detector,
        "mode": "full",
        "strategy": "exhaustive",
        "detector_config_sha256": detector_sha,
        "golden_set_sha256": golden_sha,
        "possible_parameter_sets": 100,
        "actual_parameter_sets": 100,
        "max_dimension": 1800,
        "wall_clock_seconds": 10.0,
        "parameter_sets_per_second": rate,
        "active_pipelines": pipelines,
        "shards": pipelines,
        "threads_per_pipeline": threads,
        "allocated_threads": pipelines * threads,
        "runner": {
            "runner_label": "e9k",
            "runner_name": runner_name,
            "cpu_model": cpu_model,
            "physical_core_count": physical,
            "logical_cpu_count": logical,
        },
    }


def test_manual_shape_parser_accepts_human_forms() -> None:
    assert parse_manual_shape("8p/48t") == (8, 48)
    assert parse_manual_shape("8/48") == (8, 48)
    assert parse_manual_shape("8p x 48t") == (8, 48)


def test_preferred_shape_prefers_exact_runner_before_hardware_peer(tmp_path: Path) -> None:
    detector = tmp_path / "detector.json"
    golden = tmp_path / "golden.json"
    _write_json(detector, {"detector": "components"})
    _write_json(golden, {"pages": []})
    detector_sha, golden_sha = _sha(detector), _sha(golden)
    model = "AMD EPYC 9655 96-Core Processor"
    rows = [
        _row(detector="components", detector_sha=detector_sha, golden_sha=golden_sha, runner_name="rh8-al321", cpu_model=model, logical=192, physical=192, pipelines=8, threads=48, rate=596.45),
        _row(detector="components", detector_sha=detector_sha, golden_sha=golden_sha, runner_name="rh8-al324", cpu_model=model, logical=192, physical=192, pipelines=9, threads=42, rate=700.00),
    ]
    index = tmp_path / "parallelism-index.json"
    _write_json(index, {"observations": rows})
    result = resolve_preferred_shape(
        parallelism_index=index,
        detector_config=detector,
        golden_set=golden,
        max_dimension=1800,
        profile=RunnerProfile("rh8-al321", "e9k", model, 192, 192),
    )
    assert result is not None
    assert (result["pipelines"], result["threads_per_pipeline"]) == (8, 48)
    assert result["source"] == "exact-runner"


def test_preferred_shape_reuses_hardware_equivalent_runner_when_exact_missing(tmp_path: Path) -> None:
    detector = tmp_path / "detector.json"
    golden = tmp_path / "golden.json"
    _write_json(detector, {"detector": "ransac"})
    _write_json(golden, {"pages": []})
    detector_sha, golden_sha = _sha(detector), _sha(golden)
    model = "AMD EPYC 9655 96-Core Processor"
    rows = [
        _row(detector="ransac", detector_sha=detector_sha, golden_sha=golden_sha, runner_name="rh8-al321", cpu_model=model, logical=192, physical=192, pipelines=23, threads=16, rate=145.80),
    ]
    index = tmp_path / "parallelism-index.json"
    _write_json(index, {"observations": rows})
    result = resolve_preferred_shape(
        parallelism_index=index,
        detector_config=detector,
        golden_set=golden,
        max_dimension=1800,
        profile=RunnerProfile("rh8-new", "e9k", model, 192, 192),
    )
    assert result is not None
    assert (result["pipelines"], result["threads_per_pipeline"]) == (23, 16)
    assert result["source"] == "hardware-profile"


def test_preferred_shape_rejects_incompatible_workload(tmp_path: Path) -> None:
    detector = tmp_path / "detector.json"
    golden = tmp_path / "golden.json"
    _write_json(detector, {"detector": "ransac"})
    _write_json(golden, {"pages": []})
    index = tmp_path / "parallelism-index.json"
    _write_json(index, {"observations": [{
        **_row(detector="ransac", detector_sha="wrong", golden_sha="wrong", runner_name="rh8-al321", cpu_model="AMD", logical=192, physical=192, pipelines=23, threads=16, rate=145.80),
    }]})
    result = resolve_preferred_shape(
        parallelism_index=index,
        detector_config=detector,
        golden_set=golden,
        max_dimension=1800,
        profile=RunnerProfile("rh8-al321", "e9k", "AMD", 192, 192),
    )
    assert result is None
