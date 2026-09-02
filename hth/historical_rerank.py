#!/usr/bin/env python3
"""Rebuild derived historical regression results from immutable raw page evidence."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from hth.results_layout import canonical_index_path, readable_index_path
from typing import Any

from hth.calibration_store import publish_run, update_index
from hth.contracts import adapt_regression_summary
from hth.regression.materialization import (
    build_canonical_calibration,
    canonical_outcome_summary_fields,
    derive_canonical_outcome,
    write_canonical_reports,
)
from hth.regression.merge_shards import _results_from_raw
from hth.regression.reports import ranking_key
from hth.regression.runner import build_winner_page_report

class HistoricalRerankSkip(RuntimeError):
    """Historical artifact is incompatible with safe canonical reranking."""



def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _matching_index_entry(results_root: Path, run_id: str) -> dict[str, Any] | None:
    index_path = readable_index_path(results_root, "calibration-index.json")
    if not index_path.is_file():
        return None
    index = _read_json(index_path)
    matches = [
        entry for entry in index.get("entries", [])
        if isinstance(entry, dict) and str(entry.get("calibration_id") or "") == run_id
    ]
    if not matches:
        return None
    return max(matches, key=lambda entry: str(entry.get("created_at_utc") or ""))


def rerank_run(run_dir: Path, results_root: Path, *, top: int = 20) -> dict[str, Any]:
    """Rerank one completed historical run without changing raw evidence."""
    raw_path = run_dir / "raw" / "results.csv"
    summary_path = run_dir / "reports" / "summary.json"
    intelligence_path = run_dir / "reports" / "calibration-intelligence.json"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Historical raw evidence is unavailable: {raw_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Historical summary is unavailable: {summary_path}")
    if not intelligence_path.is_file():
        raise HistoricalRerankSkip(
            f"historical calibration intelligence is unavailable: {intelligence_path}"
        )

    summary = adapt_regression_summary(_read_json(summary_path))
    manifest = _read_json(run_dir / "manifest.json")
    info = _read_json(run_dir / "RUN-INFO.json")
    old_intelligence = _read_json(intelligence_path)

    if str(manifest.get("status") or "").lower() != "complete":
        raise ValueError(f"Historical run is not complete: {run_dir}")
    strategy = str(summary.get("strategy") or summary.get("requested_strategy") or manifest.get("strategy") or "")
    if strategy not in {"exhaustive", "exhaustive-with-zombies", "cartesian"}:
        raise ValueError(f"Historical reranking is restricted to exhaustive runs; got {strategy!r}")

    evidence = _results_from_raw(raw_path)
    if not evidence:
        raise ValueError(f"No parameter-set evidence found in {raw_path}")

    outcome = derive_canonical_outcome(
        evidence,
        ranking_key=ranking_key,
        winner_page_builder=build_winner_page_report,
    )
    ordered = outcome.ordered
    winner = outcome.winner
    measurement_state = outcome.measurement_state
    if winner is None:
        raise HistoricalRerankSkip(
            f"historical run has {measurement_state['status']}; refusing to manufacture a winner"
        )

    run_id = str(summary.get("run_id") or info.get("run_id") or manifest.get("run_id") or run_dir.name)
    for result in ordered:
        result["run_id"] = run_id

    old_winner_id = str(((summary.get("winner") or {}).get("parameter_set_id") or ""))
    page_ordinals = summary.get("page_ordinals", []) if isinstance(summary.get("page_ordinals"), list) else []
    pages = len(page_ordinals)
    possible = int(
        info.get("possible_parameter_sets")
        or (summary.get("parameter_space") or {}).get("possible_parameter_sets")
        or len(ordered)
    )

    # Preserve historical execution/provenance fields and replace only values
    # derived from parameter-set/page evidence.
    summary.update(canonical_outcome_summary_fields(outcome))
    progress = summary.setdefault("progress", {})
    progress["failures"] = sum(int(result["summary"].get("failure_count") or 0) for result in ordered)
    summary["historical_rerank"] = {
        "schema_version": "1.0",
        "reranked_from_raw": True,
        "reranked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "reason": "canonical-result-metrics",
        "original_run_id": run_id,
        "previous_winner_parameter_set_id": old_winner_id or None,
        "winner_changed": bool(old_winner_id and old_winner_id != str(winner.get("parameter_set_id") or "")),
    }

    calibration = build_canonical_calibration(
        outcome,
        detector=str(summary.get("detector") or manifest.get("detector") or old_intelligence.get("detector") or "unknown"),
        strategy=strategy,
        possible_parameter_sets=possible,
        calibration_identity=old_intelligence.get("calibration_identity")
        if isinstance(old_intelligence.get("calibration_identity"), dict) else {},
        regression_metadata=old_intelligence.get("regression_metadata")
        if isinstance(old_intelligence.get("regression_metadata"), dict) else {},
    )
    # Preserve persistence/status metadata; this is a reinterpretation of the
    # original evidence, not a new detector execution.
    for key in ("calibration_status", "persistence"):
        if key in old_intelligence:
            calibration[key] = old_intelligence[key]
    write_canonical_reports(
        run_dir,
        outcome,
        summary=summary,
        calibration=calibration,
        top=top,
        write_raw=False,
    )

    existing_entry = _matching_index_entry(results_root, run_id)
    if existing_entry is None:
        raise HistoricalRerankSkip(
            f"no persisted calibration-index entry matches historical run {run_id}; "
            "refusing to create ambiguous provenance"
        )
    build = dict(existing_entry.get("build")) if isinstance(existing_entry.get("build"), dict) else {}
    mode = "full" if str(existing_entry.get("calibration_status") or "") == "authoritative" else "smoke"
    entry = publish_run(
        run_dir,
        results_root,
        mode=mode,
        source_fallback=str(existing_entry.get("source_document_id") or "results-repository"),
        build=build,
    )
    update_index(results_root, [entry])

    return {
        "run_id": run_id,
        "detector": str(summary.get("detector") or manifest.get("detector") or "unknown"),
        "parameter_sets": len(ordered),
        "previous_winner": old_winner_id or None,
        "winner": str(winner.get("parameter_set_id") or ""),
        "winner_changed": bool(old_winner_id and old_winner_id != str(winner.get("parameter_set_id") or "")),
        "avg_iou": winner["summary"].get("mean_iou"),
        "avg_iou_success": winner["summary"].get("mean_iou_success"),
        "failures": winner["summary"].get("failure_count"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--summary")
    args = parser.parse_args(argv)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in args.run_dir:
        try:
            row = rerank_run(path, args.results_root, top=args.top)
        except HistoricalRerankSkip as exc:
            skipped.append({"run_dir": str(path), "reason": str(exc)})
            print(f"WARNING: Skipping incompatible historical artifact: {path}: {exc}")
            continue
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    if args.summary:
        out = Path(args.summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "## Historical regression rerank", "",
            f"- Reranked: **{len(rows)}**",
            f"- Skipped incompatible: **{len(skipped)}**", "",
        ]
        for row in rows:
            changed = "changed" if row["winner_changed"] else "unchanged"
            lines.append(
                f"- `{row['detector']}` / `{row['run_id']}`: winner {changed}; "
                f"`{row['winner']}` — Avg IoU `{float(row['avg_iou'] or 0):.4f}`, "
                f"Avg IoU Success `{float(row['avg_iou_success'] or 0):.4f}`, "
                f"failures `{row['failures']}`."
            )
        if skipped:
            lines += ["", "### Skipped incompatible historical artifacts", ""]
            for item in skipped:
                lines.append(f"- `{item['run_dir']}` — {item['reason']}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if rows or skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
