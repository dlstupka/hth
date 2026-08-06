#!/usr/bin/env python3
"""Build execution-optimizer intelligence, Markdown tables, and SVG heat maps."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

OPTIMIZER_INDEX_SCHEMA_VERSION = "1.0"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _duration(seconds: Any) -> str:
    value = _as_float(seconds)
    if value is None:
        return "unknown"
    total = max(0, int(round(value)))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def _runner_key(row: dict[str, Any]) -> str:
    runner = row.get("runner") if isinstance(row.get("runner"), dict) else {}
    labels = runner.get("runner_labels")
    if isinstance(labels, list):
        label_text = ",".join(sorted(str(item) for item in labels))
    else:
        label_text = str(labels or "")
    identity = {
        "runner_label": runner.get("runner_label"),
        "runner_name": runner.get("runner_name"),
        "labels": label_text,
        "cpu_model": runner.get("cpu_model"),
        "logical_cpu_count": runner.get("logical_cpu_count"),
    }
    return _canonical_hash(identity)[:16]


def _runner_title(row: dict[str, Any]) -> str:
    runner = row.get("runner") if isinstance(row.get("runner"), dict) else {}
    label = str(runner.get("runner_label") or "unknown")
    name = str(runner.get("runner_name") or "").strip()
    logical = _as_int(runner.get("logical_cpu_count"))
    suffix = f" — {name}" if name and name.lower() != label.lower() else ""
    cpu_suffix = f" ({logical} vCPU)" if logical else ""
    return f"{label}{suffix}{cpu_suffix}"


def _runner_labels(row: dict[str, Any]) -> str:
    runner = row.get("runner") if isinstance(row.get("runner"), dict) else {}
    labels = runner.get("runner_labels")
    if isinstance(labels, list):
        return ", ".join(str(item) for item in labels)
    return str(labels or runner.get("runner_label") or "unknown")


def _comparable(rows: Iterable[dict[str, Any]], detector_id: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        if str(row.get("detector_id")) != detector_id:
            continue
        if row.get("mode") != "full" or row.get("strategy") != "exhaustive":
            continue
        if _as_int(row.get("actual_parameter_sets")) != _as_int(row.get("possible_parameter_sets")):
            continue
        if (_as_float(row.get("wall_clock_seconds")) or 0) <= 0:
            continue
        result.append(row)
    return result


def build_optimizer_index(parallelism_index: dict[str, Any], detector_id: str) -> dict[str, Any]:
    rows = _comparable(
        (row for row in parallelism_index.get("observations", []) if isinstance(row, dict)),
        detector_id,
    )
    runner_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        runner_groups.setdefault(_runner_key(row), []).append(row)

    runners: list[dict[str, Any]] = []
    for runner_key, runner_rows in runner_groups.items():
        shape_groups: dict[str, list[dict[str, Any]]] = {}
        for row in runner_rows:
            shape_groups.setdefault(str(row.get("execution_shape") or "unknown"), []).append(row)

        one_pipeline = [
            row for row in runner_rows
            if _as_int(row.get("active_pipelines")) == 1 and (_as_float(row.get("wall_clock_seconds")) or 0) > 0
        ]
        baseline_wall = min((_as_float(row.get("wall_clock_seconds")) for row in one_pipeline), default=None)

        shapes: list[dict[str, Any]] = []
        for execution_shape, shape_rows in shape_groups.items():
            walls = sorted(float(row["wall_clock_seconds"]) for row in shape_rows)
            fastest = min(shape_rows, key=lambda row: float(row["wall_clock_seconds"]))
            median_wall = statistics.median(walls)
            wall = walls[0]
            observed_speedup = (baseline_wall / wall) if baseline_wall and wall > 0 else None
            shapes.append({
                "execution_shape": execution_shape,
                "pipelines": _as_int(fastest.get("active_pipelines")),
                "shards": _as_int(fastest.get("shards")),
                "threads_per_pipeline": _as_int(fastest.get("threads_per_pipeline")),
                "allocated_threads": _as_int(fastest.get("allocated_threads")),
                "observation_count": len(shape_rows),
                "fastest_wall_clock_seconds": wall,
                "median_wall_clock_seconds": median_wall,
                "parameter_sets_per_second": _as_float(fastest.get("parameter_sets_per_second")),
                "page_evaluations_per_second": _as_float(fastest.get("page_evaluations_per_second")),
                "effective_acceleration": _as_float(fastest.get("effective_acceleration")),
                "parallel_efficiency": _as_float(fastest.get("parallel_efficiency")),
                "observed_speedup_vs_one_pipeline": observed_speedup,
                "fastest_observation_id": fastest.get("observation_id"),
                "build": fastest.get("build"),
            })
        shapes.sort(key=lambda item: (float(item.get("fastest_wall_clock_seconds") or math.inf), int(item.get("pipelines") or 0)))
        representative = min(runner_rows, key=lambda row: float(row["wall_clock_seconds"]))
        best = shapes[0] if shapes else None
        runners.append({
            "runner_key": runner_key,
            "runner_title": _runner_title(representative),
            "runner_label": (representative.get("runner") or {}).get("runner_label"),
            "runner_name": (representative.get("runner") or {}).get("runner_name"),
            "runner_labels": _runner_labels(representative),
            "cpu_model": (representative.get("runner") or {}).get("cpu_model"),
            "logical_cpu_count": (representative.get("runner") or {}).get("logical_cpu_count"),
            "physical_core_count": (representative.get("runner") or {}).get("physical_core_count"),
            "memory_gib": (representative.get("runner") or {}).get("memory_gib"),
            "baseline_one_pipeline_wall_clock_seconds": baseline_wall,
            "best_shape": best,
            "shapes": shapes,
        })

    runners.sort(key=lambda item: str(item.get("runner_title") or ""))
    all_best = [runner["best_shape"] | {"runner_key": runner["runner_key"], "runner_title": runner["runner_title"]}
                for runner in runners if runner.get("best_shape")]
    all_best.sort(key=lambda item: float(item.get("fastest_wall_clock_seconds") or math.inf))
    return {
        "schema_version": OPTIMIZER_INDEX_SCHEMA_VERSION,
        "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_parallelism_schema_version": parallelism_index.get("schema_version"),
        "detector_id": detector_id,
        "runner_count": len(runners),
        "observation_count": len(rows),
        "best_across_runners": all_best[0] if all_best else None,
        "runners": runners,
    }


def render_markdown(index: dict[str, Any]) -> str:
    lines = [
        "### Execution optimizer summary",
        "",
        f"Detector: `{index.get('detector_id')}`  ",
        f"Compatible observations: **{index.get('observation_count', 0)}** across **{index.get('runner_count', 0)}** runner profiles.",
        "",
        "| Runner | Pipelines | Shards | Threads / pipeline | Allocated threads | Fastest wall | Median wall | Sets/s | Speedup vs 1 pipeline | Efficiency | Runs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows: list[tuple[str, dict[str, Any], bool]] = []
    for runner in index.get("runners", []):
        best_shape = (runner.get("best_shape") or {}).get("execution_shape")
        for shape in runner.get("shapes", []):
            rows.append((str(runner.get("runner_title") or "unknown"), shape, shape.get("execution_shape") == best_shape))
    rows.sort(key=lambda item: (item[0], int(item[1].get("pipelines") or 0), int(item[1].get("threads_per_pipeline") or 0)))
    for runner_title, shape, best in rows:
        runner_cell = f"**{runner_title}**" if best else runner_title
        speedup = _as_float(shape.get("observed_speedup_vs_one_pipeline"))
        efficiency = _as_float(shape.get("parallel_efficiency"))
        rate = _as_float(shape.get("parameter_sets_per_second"))
        lines.append(
            "| {runner} | {pipelines} | {shards} | {threads} | {allocated} | {fastest} | {median} | {rate} | {speedup} | {efficiency} | {runs} |".format(
                runner=runner_cell,
                pipelines=shape.get("pipelines") or "?",
                shards=shape.get("shards") or "?",
                threads=shape.get("threads_per_pipeline") or "?",
                allocated=shape.get("allocated_threads") or "?",
                fastest=_duration(shape.get("fastest_wall_clock_seconds")),
                median=_duration(shape.get("median_wall_clock_seconds")),
                rate=f"{rate:.2f}" if rate is not None else "unknown",
                speedup=f"{speedup:.2f}×" if speedup is not None else "no 1-pipeline baseline",
                efficiency=f"{efficiency * 100:.1f}%" if efficiency is not None else "unknown",
                runs=shape.get("observation_count") or 0,
            )
        )
    lines.extend(["", "**Bold runner rows mark that runner profile's fastest measured execution shape.**", ""])
    return "\n".join(lines)


def _color(value: float, minimum: float, maximum: float) -> str:
    if maximum <= minimum:
        ratio = 0.0
    else:
        ratio = (math.log(max(value, 1e-9)) - math.log(max(minimum, 1e-9))) / (
            math.log(max(maximum, 1e-9)) - math.log(max(minimum, 1e-9))
        )
    ratio = min(1.0, max(0.0, ratio))
    # Green (fast) through amber to red (slow), expressed directly for portable SVG.
    if ratio < 0.5:
        local = ratio / 0.5
        r, g, b = int(40 + 190 * local), int(170 + 35 * local), int(90 - 45 * local)
    else:
        local = (ratio - 0.5) / 0.5
        r, g, b = int(230 + 15 * local), int(205 - 150 * local), int(45 - 5 * local)
    return f"rgb({r},{g},{b})"


def render_heatmap_svg(index: dict[str, Any]) -> str:
    runners = [runner for runner in index.get("runners", []) if runner.get("shapes")]
    all_shapes = [shape for runner in runners for shape in runner.get("shapes", [])]
    walls = [float(shape["fastest_wall_clock_seconds"]) for shape in all_shapes]
    min_wall = min(walls) if walls else 0.0
    max_wall = max(walls) if walls else 1.0
    panel_width = 480
    panel_height = 390
    columns = min(2, max(1, len(runners)))
    rows = max(1, math.ceil(len(runners) / columns))
    width = columns * panel_width
    height = 95 + rows * panel_height + 65
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        '<style>text{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;fill:#e6edf3}.muted{fill:#8b949e}.axis{stroke:#484f58;stroke-width:1}.cell{stroke:#e6edf3;stroke-width:.7}</style>',
        f'<text x="20" y="32" font-size="22" font-weight="700">Execution optimizer — {html.escape(str(index.get("detector_id")))}</text>',
        f'<text x="20" y="56" font-size="13" class="muted">Fastest measured wall time by pipeline/thread shape; shared logarithmic color scale across runner profiles.</text>',
    ]
    for panel_index, runner in enumerate(runners):
        col = panel_index % columns
        row = panel_index // columns
        ox = col * panel_width + 50
        oy = 90 + row * panel_height
        shapes = runner.get("shapes", [])
        pipelines = sorted({int(shape["pipelines"]) for shape in shapes if shape.get("pipelines")})
        threads = sorted({int(shape["threads_per_pipeline"]) for shape in shapes if shape.get("threads_per_pipeline")})
        plot_x, plot_y, plot_w, plot_h = ox + 70, oy + 55, 340, 255
        parts.append(f'<text x="{ox}" y="{oy + 20}" font-size="16" font-weight="700">{html.escape(str(runner.get("runner_title")))}</text>')
        parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" class="axis"/>')
        parts.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" class="axis"/>')
        x_positions = {value: plot_x + (i + 0.5) * plot_w / max(1, len(threads)) for i, value in enumerate(threads)}
        y_positions = {value: plot_y + plot_h - (i + 0.5) * plot_h / max(1, len(pipelines)) for i, value in enumerate(pipelines)}
        cell_w = min(58, plot_w / max(1, len(threads)) * 0.78)
        cell_h = min(42, plot_h / max(1, len(pipelines)) * 0.72)
        for value, x in x_positions.items():
            parts.append(f'<text x="{x}" y="{plot_y + plot_h + 22}" text-anchor="middle" font-size="11">{value}</text>')
        for value, y in y_positions.items():
            parts.append(f'<text x="{plot_x - 10}" y="{y + 4}" text-anchor="end" font-size="11">{value}</text>')
        parts.append(f'<text x="{plot_x + plot_w / 2}" y="{plot_y + plot_h + 43}" text-anchor="middle" font-size="12" class="muted">threads / pipeline</text>')
        parts.append(f'<text x="{plot_x - 48}" y="{plot_y + plot_h / 2}" text-anchor="middle" font-size="12" class="muted" transform="rotate(-90 {plot_x - 48} {plot_y + plot_h / 2})">pipelines</text>')
        best_shape = (runner.get("best_shape") or {}).get("execution_shape")
        for shape in shapes:
            x = x_positions.get(int(shape["threads_per_pipeline"]))
            y = y_positions.get(int(shape["pipelines"]))
            wall = float(shape["fastest_wall_clock_seconds"])
            if x is None or y is None:
                continue
            stroke_width = 3 if shape.get("execution_shape") == best_shape else 0.7
            parts.append(f'<rect x="{x - cell_w/2:.1f}" y="{y - cell_h/2:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" rx="5" fill="{_color(wall, min_wall, max_wall)}" class="cell" stroke-width="{stroke_width}"/>')
            parts.append(f'<text x="{x}" y="{y + 4}" text-anchor="middle" font-size="10" fill="#0d1117" style="fill:#0d1117">{html.escape(_duration(wall))}</text>')
    legend_y = height - 38
    legend_x = 20
    legend_w = min(500, width - 40)
    steps = 80
    for i in range(steps):
        value = min_wall + (max_wall - min_wall) * i / max(1, steps - 1)
        parts.append(f'<rect x="{legend_x + legend_w*i/steps:.1f}" y="{legend_y}" width="{legend_w/steps + 1:.1f}" height="12" fill="{_color(value, min_wall, max_wall)}"/>')
    parts.append(f'<text x="{legend_x}" y="{legend_y - 6}" font-size="11" class="muted">fast {_duration(min_wall)}</text>')
    parts.append(f'<text x="{legend_x + legend_w}" y="{legend_y - 6}" text-anchor="end" font-size="11" class="muted">slow {_duration(max_wall)}</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def update_optimizer_artifacts(results_root: Path, detector_id: str) -> dict[str, Path]:
    parallelism_path = results_root / "parallelism-index.json"
    if not parallelism_path.is_file():
        raise FileNotFoundError(f"Missing {parallelism_path}")
    index = build_optimizer_index(_read_json(parallelism_path), detector_id)
    index_path = results_root / "optimizer-index.json"
    existing: dict[str, Any]
    if index_path.is_file():
        existing = _read_json(index_path)
    else:
        existing = {"schema_version": OPTIMIZER_INDEX_SCHEMA_VERSION, "detectors": {}}
    detectors = existing.get("detectors") if isinstance(existing.get("detectors"), dict) else {}
    detectors[detector_id] = index
    existing.update({
        "schema_version": OPTIMIZER_INDEX_SCHEMA_VERSION,
        "updated_at_utc": index["updated_at_utc"],
        "detectors": detectors,
    })
    _write_json(index_path, existing)
    output_dir = results_root / "execution-optimizer" / detector_id
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "summary.md"
    svg_path = output_dir / "heatmap.svg"
    markdown_path.write_text(render_markdown(index), encoding="utf-8")
    svg_path.write_text(render_heatmap_svg(index), encoding="utf-8")
    return {"index": index_path, "markdown": markdown_path, "heatmap": svg_path}


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--detector", required=True)
    args = parser.parse_args()
    paths = update_optimizer_artifacts(args.results_root, args.detector)
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
