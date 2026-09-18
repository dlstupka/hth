#!/usr/bin/env python3
"""Render compact GitHub summaries from canonical tonal evidence."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _method_summary(comparison: dict[str, Any]) -> list[str]:
    lines = [
        "### Bounded tonal method comparison",
        "",
        f"- Development candidates: `{comparison['candidate_count']}`",
        f"- Globally safe methods: `{len(comparison['globally_safe_methods'])}`",
        f"- Recommended method: `{comparison.get('recommended_method_id') or 'none'}`",
        "",
        "| Method | Safe pages | Mean tonal gain | Mean detail correlation | Gate failures |",
        "|---|---:|---:|---:|---|",
    ]
    pages = comparison["pages"]
    for method in comparison["config"]["methods"]:
        variants = [
            variant
            for page in pages
            for variant in page["variants"]
            if variant["method_id"] == method["id"]
        ]
        safe_count = sum(variant["safe"] is True for variant in variants)
        failures = {
            gate: sum(variant["gates"][gate] is False for variant in variants)
            for gate in ("tonal_span_gain", "detail_correlation", "endpoint_clipping", "median_shift")
        }
        failure_text = ", ".join(
            f"{name.replace('_', ' ')}: {count}" for name, count in failures.items() if count
        ) or "none"
        mean_gain = statistics.fmean(variant["tonal_span_gain"] for variant in variants) if variants else 0.0
        mean_correlation = (
            statistics.fmean(variant["high_frequency_correlation"] for variant in variants)
            if variants else 0.0
        )
        lines.append(
            f"| `{method['id']}` | {safe_count}/{len(variants)} | {mean_gain:.3f} | "
            f"{mean_correlation:.3f} | {failure_text} |"
        )
    return lines


def summary_lines(stage: str, payload: dict[str, Any]) -> list[str]:
    if stage == "assess":
        aggregate = payload["aggregate"]
        return [
            "### Contrast and tonal assessment",
            "",
            f"- Pages evaluated: `{aggregate['page_count']}`",
            f"- Correction candidates: `{aggregate['correction-candidate']}`",
            f"- Preserve: `{aggregate['preserve']}`",
            f"- Review: `{aggregate['review']}`",
            "- Pipeline action: `all-pages-continue`",
        ]
    if stage == "compare":
        return _method_summary(payload)
    if stage == "validate":
        aggregate = payload["aggregate"]
        method = payload.get("method")
        return [
            "### Held-out tonal validation",
            "",
            f"- Method: `{method['id'] if method else 'none'}`",
            f"- Held-out candidates: `{aggregate['held_out_candidates']}`",
            f"- Safe candidates: `{aggregate['safe_candidates']}`",
            f"- Mean tonal-span gain: `{aggregate['mean_tonal_span_gain']:.3f}`",
            f"- Decision: `{payload['decision']}`",
        ]
    aggregate = payload["aggregate"]
    method = payload.get("method")
    return [
        "### Contrast and tonal integration",
        "",
        f"- Method: `{method['id'] if method else 'none'}`",
        f"- Input pages: `{aggregate['page_count']}`",
        f"- Corrected pages: `{aggregate['corrected_pages']}`",
        f"- Preserved pages: `{aggregate['preserved_pages']}`",
        f"- Tonal result: `{payload['tonal_result_identity']}`",
        "- Pipeline action: `all-pages-continue`",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("assess", "compare", "validate", "integrate"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--github-summary", type=Path, required=True)
    args = parser.parse_args(argv)
    args.github_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.github_summary.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "\n".join(summary_lines(args.stage, _read(args.input))) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
