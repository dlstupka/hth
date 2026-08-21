from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hth.write_action_summary import append_bounded_summary, compact_manifest


class WriteActionSummaryTests(unittest.TestCase):
    def test_compact_manifest_omits_nested_per_detector_detail_sections(self) -> None:
        text = """# Manifest\n\n## Ranked Detector Smoke Test Results\n\n| Rank | Detector |\n|---:|---|\n| 1 | A |\n\n<details open>\n<summary><h3>Per-Detector Calibration Reports</h3></summary>\n\n<details>\n<summary>A</summary>\nvery large calibration body\n</details>\n\n</details>\n\n## Keep Me\n\nimportant execution summary\n\n<details open>\n<summary><h3>Per-Detector Regression Reports</h3></summary>\n\n<details>\n<summary>A</summary>\nvery large regression body\n</details>\n\n</details>\n\n## Tail\n\nkept\n"""
        compacted, removed = compact_manifest(text)
        self.assertEqual(2, len(removed))
        self.assertIn("Ranked Detector Smoke Test Results", compacted)
        self.assertIn("important execution summary", compacted)
        self.assertIn("## Tail", compacted)
        self.assertNotIn("very large calibration body", compacted)
        self.assertNotIn("very large regression body", compacted)
        self.assertIn("complete manifest is preserved", compacted)

    def test_append_bounded_summary_respects_total_destination_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "manifest.md"
            destination = root / "summary.md"
            destination.write_text("existing\n", encoding="utf-8")
            source.write_text("# Manifest\n\n" + "paragraph\n\n" * 200, encoding="utf-8")
            result = append_bounded_summary(source, destination, max_bytes=512)
            self.assertLessEqual(destination.stat().st_size, 512)
            self.assertTrue(result["truncated"])


if __name__ == "__main__":
    unittest.main()
