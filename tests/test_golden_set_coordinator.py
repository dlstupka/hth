import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from hth.geometry.model import Candidate
from hth.regression.golden_set_coordinator import GoldenSetCoordinator, evaluate_set


def _pages(count=6):
    return [
        {
            "global_ordinal": ordinal,
            "label": f"page-{ordinal}",
            "layout_type": "page",
            "image": np.full((8, 8, 3), ordinal, dtype=np.uint8),
            "mask": np.ones((8, 8), dtype=np.uint8),
            "scale": 1.0,
            "original_width": 8,
            "original_height": 8,
            "approved_bbox": [1, 1, 7, 7],
        }
        for ordinal in range(1, count + 1)
    ]


def _detector(*, image_bgr, mask, parameters):
    del mask
    if parameters.get("delay"):
        time.sleep(float(parameters["delay"]))
    if parameters.get("fail") and int(image_bgr[0, 0, 0]) == 3:
        raise RuntimeError("page failure")
    return Candidate("test", [1, 1, 7, 7], None, 1.0, 1.0, {})


class GoldenSetCoordinatorTests(unittest.TestCase):
    def test_parallel_lanes_preserve_serial_results_and_ordinal_order(self):
        pages = _pages()
        parameters = {"value": 1}
        serial = evaluate_set(_detector, parameters, pages)
        coordinated = GoldenSetCoordinator(
            _detector, pages, lanes=3, total_threads=6, threads_per_lane=2,
        ).evaluate(parameters)
        self.assertEqual(
            [row["global_ordinal"] for row in coordinated["pages"]],
            list(range(1, 7)),
        )
        self.assertEqual(
            [row["iou"] for row in coordinated["pages"]],
            [row["iou"] for row in serial["pages"]],
        )

    def test_batch_has_full_results_and_stable_candidate_order(self):
        coordinator = GoldenSetCoordinator(
            _detector, _pages(4), lanes=2, total_threads=4, threads_per_lane=2,
        )
        parameters = [{"value": 2}, {"value": 1}]
        results = coordinator.evaluate_batch(parameters)
        self.assertEqual([row["parameters"] for row in results], parameters)
        self.assertTrue(all(len(row["pages"]) == 4 for row in results))
        snapshot = coordinator.snapshot()
        self.assertEqual(snapshot["rounds"], 1)
        self.assertEqual(snapshot["candidate_parameter_sets"], 2)
        self.assertEqual(snapshot["page_evaluations"], 8)
        self.assertLessEqual(snapshot["peak_active_lane_tasks"], 4)

    def test_page_failure_remains_page_evidence(self):
        result = GoldenSetCoordinator(
            _detector, _pages(4), lanes=2, total_threads=2, threads_per_lane=1,
        ).evaluate({"fail": True})
        failed = [row for row in result["pages"] if row["status"] == "error"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["global_ordinal"], 3)
        self.assertEqual(failed[0]["error"]["type"], "RuntimeError")

    def test_verbose_diagnostics_are_optional_and_bounded_to_page_events(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "coordinator.jsonl"
            quiet = GoldenSetCoordinator(
                _detector, _pages(2), lanes=1, total_threads=1,
                threads_per_lane=1, diagnostics_path=path, verbose=False,
            )
            quiet.evaluate({"value": 1})
            self.assertEqual(path.read_text(encoding="utf-8"), "")
            verbose = GoldenSetCoordinator(
                _detector, _pages(2), lanes=1, total_threads=1,
                threads_per_lane=1, diagnostics_path=path, verbose=True,
            )
            verbose.evaluate({"value": 1})
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["event"] == "page-finish" for row in rows))

    def test_invalid_capacity_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "reservation"):
            GoldenSetCoordinator(
                _detector, _pages(2), lanes=2, total_threads=3, threads_per_lane=2,
            )
        with self.assertRaisesRegex(ValueError, "at least one page"):
            GoldenSetCoordinator(
                _detector, [], lanes=1, total_threads=1, threads_per_lane=1,
            )


if __name__ == "__main__":
    unittest.main()
