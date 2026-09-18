"""Compact summaries for denoising and sharpening evidence."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any


def summary_lines(stage: str, payload: dict[str, Any]) -> list[str]:
    domain = str(payload.get("domain") or "restoration")
    title = "Denoising and artifact suppression" if domain == "denoising" else "Sharpening and detail enhancement"
    aggregate = payload.get("aggregate") or {}
    if stage == "assess":
        return [f"## {title} assessment", "", f"- Pages evaluated: `{aggregate.get('page_count', 0)}`", f"- Correction candidates: `{aggregate.get('correction-candidate', 0)}`", f"- Preserve: `{aggregate.get('preserve', 0)}`", f"- Review: `{aggregate.get('review', 0)}`", "- Pipeline action: `all-pages-continue`", ""]
    if stage == "compare":
        lines = [f"## Bounded {domain} method comparison", "", f"- Development candidates: `{payload.get('candidate_count', 0)}`", f"- Globally safe methods: `{len(payload.get('globally_safe_methods') or [])}`", f"- Recommended method: `{payload.get('recommended_method_id') or 'none'}`", "", "| Method | Safe pages | Mean noise reduction | Mean detail gain | Mean detail correlation | Gate failures |", "|---|---:|---:|---:|---:|---|"]
        for method in (payload.get("config") or {}).get("methods") or []:
            variants = [next((v for v in p.get("variants", []) if v.get("method_id") == method["id"]), None) for p in payload.get("pages") or []]
            variants = [v for v in variants if v]
            safe = sum(v.get("safe") is True for v in variants)
            mean = lambda key: sum(float(v.get(key, 0)) for v in variants) / len(variants) if variants else 0.0
            failures: dict[str, int] = {}
            for variant in variants:
                for gate, passed in (variant.get("gates") or {}).items():
                    if not passed: failures[gate.replace("_", " ")] = failures.get(gate.replace("_", " "), 0) + 1
            failure_text = ", ".join(f"{k}: {v}" for k, v in failures.items()) or "none"
            lines.append(f"| `{method['id']}` | {safe}/{len(variants)} | {mean('noise_reduction_fraction'):.3f} | {mean('detail_gain_fraction'):.3f} | {mean('detail_correlation'):.3f} | {failure_text} |")
        return lines + [""]
    if stage == "validate":
        return [f"## Held-out {domain} validation", "", f"- Method: `{(payload.get('method') or {}).get('id', 'none')}`", f"- Held-out candidates: `{aggregate.get('held_out_candidates', 0)}`", f"- Safe candidates: `{aggregate.get('safe_candidates', 0)}`", f"- Decision: `{payload.get('decision', 'unknown')}`", ""]
    return [f"## {title} integration", "", f"- Method: `{(payload.get('method') or {}).get('id', 'none')}`", f"- Input pages: `{aggregate.get('page_count', 0)}`", f"- Corrected pages: `{aggregate.get('corrected_pages', 0)}`", f"- Preserved pages: `{aggregate.get('preserved_pages', 0)}`", f"- Result: `{payload.get(domain + '_result_identity', 'unknown')}`", "- Pipeline action: `all-pages-continue`", ""]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("assess", "compare", "validate", "integrate"), required=True); parser.add_argument("--input", type=Path, required=True); parser.add_argument("--github-summary", type=Path, required=True)
    args = parser.parse_args(argv); payload = json.loads(args.input.read_text(encoding="utf-8"))
    with args.github_summary.open("a", encoding="utf-8") as handle: handle.write("\n".join(summary_lines(args.stage, payload)) + "\n")
    return 0


if __name__ == "__main__": raise SystemExit(main())
