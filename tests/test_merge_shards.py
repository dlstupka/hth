from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from hth.regression.merge_shards import _results_from_raw
from hth.regression.sharding import plan_shards


def _write_raw_row(path, *, status: str = "ok", iou: float = 0.9) -> None:
    fields = [
        "run_id", "parameter_set_id", "profile", "rank", "completion_index", "completion_elapsed_seconds", "search_fraction", "global_ordinal", "label",
        "layout_type", "status", "iou", "left_error_px", "top_error_px",
        "right_error_px", "bottom_error_px", "edge_error_mean_px",
        "edge_error_maximum_px", "elapsed_ms", "approved_bbox_json",
        "predicted_bbox_json", "parameters_json", "error_type", "error_message",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "run_id": "run-1",
            "parameter_set_id": "abc",
            "profile": "baseline",
            "rank": 1,
            "completion_index": 3,
            "completion_elapsed_seconds": 12.0,
            "search_fraction": 0.3,
            "global_ordinal": 1,
            "label": "page-1",
            "layout_type": "single",
            "status": status,
            "iou": iou,
            "elapsed_ms": 12.5,
            "approved_bbox_json": json.dumps([0, 0, 10, 10]),
            "predicted_bbox_json": json.dumps([0, 0, 10, 10]),
            "parameters_json": json.dumps({"x": 1}),
        })


class MergeShardReconstructionTests(unittest.TestCase):
    def test_shard_planner_caps_runner_threads(self) -> None:
        for runner_label in ("e7k", "e9k"):
            with self.subTest(runner_label=runner_label):
                self.assertEqual(
                    plan_shards(
                        4 * 3600,
                        runner_label=runner_label,
                        requested_threads="auto",
                    ).threads,
                    64,
                )

    def test_success_row_reconstructs_canonical_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "results.csv"
            _write_raw_row(raw, status="ok", iou=0.9)

            result = _results_from_raw(raw)[0]

        page = result["pages"][0]
        self.assertEqual(page["error"], {})
        self.assertEqual(page["warnings"], [])
        self.assertEqual(page["metadata"], {})
        self.assertEqual(result["summary"]["success_count"], 1)
        self.assertEqual(result["summary"]["failure_count"], 0)
        self.assertEqual(result["summary"]["mean_iou"], 0.9)
        self.assertEqual(result["search_observation"], {
            "completion_index": 3,
            "parameter_set_number": 3,
            "elapsed_seconds": 12.0,
            "search_fraction": 0.3,
        })

    def test_persisted_gzip_reconstructs_the_same_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "results.csv"
            _write_raw_row(raw, status="ok", iou=0.9)
            compressed = raw.with_name("results.csv.gz")
            with raw.open("rb") as source, gzip.open(compressed, "wb") as target:
                target.write(source.read())

            plain = _results_from_raw(raw)
            restored = _results_from_raw(compressed)

        self.assertEqual(restored, plain)

    def test_reconstruction_preserves_missing_elapsed_and_failed_page_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "results.csv"
            _write_raw_row(raw, status="ok", iou=0.9638)
            with raw.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            fields = list(rows[0])
            rows[0]["completion_elapsed_seconds"] = ""
            with raw.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(rows[0])
                for ordinal in range(2, 6):
                    failed = dict(rows[0])
                    failed.update({
                        "global_ordinal": ordinal,
                        "label": f"page-{ordinal}",
                        "status": "no_candidate",
                        "iou": 0.0,
                        "edge_error_mean_px": "",
                        "edge_error_maximum_px": "",
                        "predicted_bbox_json": "null",
                    })
                    writer.writerow(failed)

            result = _results_from_raw(raw)[0]

        self.assertEqual(result["search_observation"]["completion_index"], 3)
        self.assertIsNone(result["search_observation"]["elapsed_seconds"])
        self.assertEqual(result["summary"]["success_count"], 1)
        self.assertEqual(result["summary"]["failure_count"], 4)
        self.assertEqual(result["summary"]["mean_iou"], 0.19276)
        self.assertEqual(result["summary"]["mean_iou_success"], 0.9638)
        self.assertEqual(result["summary"]["minimum_iou"], 0.0)


if __name__ == "__main__":
    unittest.main()
