from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TOOL = Path(__file__).resolve().parents[1] / "tools" / "layout-smoke-report.py"
SPEC = importlib.util.spec_from_file_location("layout_smoke_report", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LayoutSmokeReportTests(unittest.TestCase):
    def test_source_only_report_does_not_claim_a_normalized_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory) / "source"
            source_dir.mkdir()
            (source_dir / "fs_0001.json").write_text(
                json.dumps({"type": "baselines", "lines": [], "regions": {}, "line_orders": []}),
                encoding="utf-8",
            )
            inputs = {
                "views": ["source"],
                "evaluation_mode": "full",
                "golden_set_id": "HTH-0001",
                "golden_set_sha256": "a" * 64,
                "source_release": {},
                "results_commit": None,
                "base_normalization_result_identity": None,
                "final_normalization_result_identity": None,
                "pages": [{"global_ordinal": 1, "source_size": [10, 10], "source_pixel_sha256": "b" * 64}],
            }
            report = MODULE.summarize(inputs, source_dir, Path(directory) / "missing", "c" * 64, "7.0.2", "cpu", 4)
            self.assertEqual(report["views"], ["source"])
            self.assertNotIn("normalized", report["summary"])
            self.assertIsNone(report["paired_pages_with_line_count_change"])

    def test_valid_geometry_and_missing_reading_order(self) -> None:
        payload = {
            "type": "baselines",
            "lines": [{
                "id": "l1",
                "baseline": [[1, 2], [8, 2]],
                "boundary": [[1, 1], [8, 1], [8, 3], [1, 3]],
                "regions": ["r1"],
            }],
            "regions": {"text": [{"id": "r1", "boundary": [[0, 0], [9, 0], [9, 9]]}]},
            "line_orders": [],
        }
        result = MODULE.inspect_segmentation(payload, [10, 10])
        self.assertEqual(result["lines"], 1)
        self.assertEqual(result["regions_by_type"], {"text": 1})
        self.assertEqual(result["invalid_line_geometry"], 0)
        self.assertEqual(result["invalid_region_geometry"], 0)
        self.assertEqual(result["orphan_lines"], 0)
        self.assertFalse(result["reading_order_present"])
        self.assertFalse(result["reading_order_complete"])

    def test_flags_out_of_bounds_and_orphaned_line(self) -> None:
        payload = {
            "type": "baselines",
            "lines": [{
                "id": "l1",
                "baseline": [[1, 2], [10, 2]],
                "boundary": [[1, 1], [8, 1], [8, 3]],
                "regions": ["missing"],
            }],
            "regions": {"text": [{"id": "r1", "boundary": [[0, 0], [9, 0]]}]},
            "line_orders": [],
        }
        result = MODULE.inspect_segmentation(payload, [10, 10])
        self.assertEqual(result["invalid_line_geometry"], 1)
        self.assertEqual(result["invalid_region_geometry"], 1)
        self.assertEqual(result["orphan_lines"], 1)

    def test_complete_reading_order(self) -> None:
        payload = {
            "type": "baselines",
            "lines": [{"id": "l1", "baseline": [[1, 1], [3, 1]], "boundary": [[1, 0], [3, 0], [3, 2]], "regions": ["r1"]}],
            "regions": {"text": [{"id": "r1", "boundary": [[0, 0], [4, 0], [4, 4]]}]},
            "line_orders": [["l1"]],
        }
        self.assertTrue(MODULE.inspect_segmentation(payload, [5, 5])["reading_order_complete"])

    def test_empty_page_is_not_a_complete_reading_order(self) -> None:
        payload = {"type": "baselines", "lines": [], "regions": {}, "line_orders": [[]]}
        self.assertFalse(MODULE.inspect_segmentation(payload, [5, 5])["reading_order_complete"])


if __name__ == "__main__":
    unittest.main()
