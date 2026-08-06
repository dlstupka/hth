from __future__ import annotations

import json
from pathlib import Path

from hth.optimizer_store import build_optimizer_index, render_heatmap_svg, render_markdown, update_optimizer_artifacts


def _row(identifier: str, runner: str, pipelines: int, threads: int, wall: float) -> dict:
    return {
        "observation_id": identifier,
        "detector_id": "adaptive_radial_edge",
        "mode": "full",
        "strategy": "exhaustive",
        "possible_parameter_sets": 6562,
        "actual_parameter_sets": 6562,
        "execution_shape": f"{pipelines}p/{pipelines}s/{threads}t",
        "shards": pipelines,
        "active_pipelines": pipelines,
        "threads_per_pipeline": threads,
        "allocated_threads": pipelines * threads,
        "wall_clock_seconds": wall,
        "parameter_sets_per_second": 6562 / wall,
        "effective_acceleration": 12.0,
        "parallel_efficiency": 12.0 / (pipelines * threads),
        "runner": {
            "runner_label": runner,
            "runner_name": runner,
            "runner_labels": ["self-hosted", "linux", runner],
            "cpu_model": f"CPU {runner}",
            "logical_cpu_count": 96 if runner == "e7k" else 32,
        },
    }


def test_optimizer_index_groups_shapes_by_runner_and_marks_best() -> None:
    parallelism = {"schema_version": "2.1", "observations": [
        _row("a", "e7k", 1, 64, 2600),
        _row("b", "e7k", 8, 8, 420),
        _row("c", "e9k", 1, 32, 1800),
        _row("d", "e9k", 8, 4, 500),
    ]}
    index = build_optimizer_index(parallelism, "adaptive_radial_edge")
    assert index["runner_count"] == 2
    assert index["observation_count"] == 4
    e7k = next(row for row in index["runners"] if row["runner_label"] == "e7k")
    assert e7k["best_shape"]["pipelines"] == 8
    assert e7k["best_shape"]["observed_speedup_vs_one_pipeline"] > 6


def test_optimizer_artifacts_include_table_and_multi_runner_heatmap(tmp_path: Path) -> None:
    (tmp_path / "parallelism-index.json").write_text(json.dumps({
        "schema_version": "2.1",
        "observations": [_row("a", "e7k", 1, 64, 2600), _row("b", "e9k", 8, 4, 500)],
    }), encoding="utf-8")
    paths = update_optimizer_artifacts(tmp_path, "adaptive_radial_edge")
    payload = json.loads(paths["index"].read_text(encoding="utf-8"))
    assert "adaptive_radial_edge" in payload["detectors"]
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert "| Runner | Pipelines |" in markdown
    assert "e7k" in markdown and "e9k" in markdown
    svg = paths["heatmap"].read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "e7k" in svg and "e9k" in svg
