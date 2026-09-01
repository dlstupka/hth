from __future__ import annotations

import json
from pathlib import Path

from hth.regression.finalize_run import finalize_run


def _staged_run(root: Path, detector: str, run_id: str) -> Path:
    run = root / detector / run_id
    (run / "reports").mkdir(parents=True)
    (run / "raw").mkdir()
    (run / "logs").mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "run_id": run_id,
        "detector": detector,
        "status": "complete",
    }), encoding="utf-8")
    (run / "reports" / "summary.json").write_text("{}", encoding="utf-8")
    (run.parent / f"{detector}-regression-results.csv").write_text(
        "rank,parameter_set_id\n1,winner\n", encoding="utf-8"
    )
    return run


def _debug(root: Path, detector: str, run_id: str, filename: str = "03-evidence.png") -> None:
    page = root / "debug" / detector / run_id / "winner" / "page-0001"
    page.mkdir(parents=True)
    (page / filename).write_bytes(b"debug")


def test_single_and_multi_shard_sources_use_identical_finalization_contract(tmp_path: Path):
    output = tmp_path / "output"

    single_root = tmp_path / ".shards" / "distance_transform" / "shard-0000"
    single = _staged_run(single_root, "distance_transform", "run-single")
    _debug(single_root, "distance_transform", "run-single", "03-distance-transform.png")
    single_target = finalize_run(
        canonical_run=single,
        staging_root=single_root,
        output_root=output,
        detector="distance_transform",
    )

    merged_root = tmp_path / ".finalize" / "convex_hull"
    merged = _staged_run(merged_root, "convex_hull", "run-merged")
    _debug(merged_root, "convex_hull", "run-merged", "03-convex-hull.png")
    merged_target = finalize_run(
        canonical_run=merged,
        staging_root=merged_root,
        output_root=output,
        detector="convex_hull",
    )

    assert single_target == output / "distance_transform" / "run-single"
    assert merged_target == output / "convex_hull" / "run-merged"
    assert (output / "debug" / "distance_transform" / "run-single" /
            "winner" / "page-0001" / "03-distance-transform.png").is_file()
    assert (output / "debug" / "convex_hull" / "run-merged" /
            "winner" / "page-0001" / "03-convex-hull.png").is_file()
    assert (output / "distance_transform" / "distance_transform-regression-results.csv").is_file()
    assert (output / "convex_hull" / "convex_hull-regression-results.csv").is_file()


def test_finalizer_promotes_only_matching_detector_run_debug(tmp_path: Path):
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    run = _staged_run(staging, "distance_transform", "run-1")
    _debug(staging, "distance_transform", "run-1", "03-distance-transform.png")
    _debug(staging, "convex_hull", "old-run", "03-convex-hull.png")

    finalize_run(
        canonical_run=run,
        staging_root=staging,
        output_root=output,
        detector="distance_transform",
    )

    assert (output / "debug" / "distance_transform" / "run-1").is_dir()
    assert not (output / "debug" / "convex_hull").exists()


def test_finalizer_rejects_detector_identity_mismatch(tmp_path: Path):
    staging = tmp_path / "staging"
    run = _staged_run(staging, "convex_hull", "run-1")

    try:
        finalize_run(
            canonical_run=run,
            staging_root=staging,
            output_root=tmp_path / "output",
            detector="distance_transform",
        )
    except ValueError as exc:
        assert "Canonical detector mismatch" in str(exc)
    else:
        raise AssertionError("Expected detector identity mismatch")


def test_finalizer_marks_zero_valid_measurement_run_failed(tmp_path: Path):
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    run = _staged_run(staging, "ransac", "run-invalid")
    state = {
        "status": "no_valid_measurements",
        "reason": "No page evaluation produced a valid detector measurement.",
        "terminal_success": False,
    }
    (run / "reports" / "summary.json").write_text(
        json.dumps({"winner": None, "measurement_state": state}), encoding="utf-8"
    )
    (run / "RUN-INFO.json").write_text(json.dumps({"status": "invalid"}), encoding="utf-8")

    target = finalize_run(
        canonical_run=run,
        staging_root=staging,
        output_root=output,
        detector="ransac",
    )

    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    info = json.loads((target / "RUN-INFO.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert info["status"] == "failed"
    assert manifest["error"]["code"] == "no_valid_measurements"
