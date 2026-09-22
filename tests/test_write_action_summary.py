from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hth.write_action_summary import append_bounded_summary, compact_manifest


ROOT = Path(__file__).resolve().parents[1]


class WriteActionSummaryTests(unittest.TestCase):
    def test_summary_writer_runs_as_module_from_workflow_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "report.md"
            destination = root / "summary.md"
            source.write_text("# Report\n", encoding="utf-8")
            environment = {**os.environ, "PYTHONPATH": str(ROOT)}
            subprocess.run(
                [sys.executable, "-m", "hth.write_action_summary", str(source), str(destination)],
                cwd=root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual("# Report\n", destination.read_text(encoding="utf-8"))

    def test_all_workflows_use_package_aware_summary_writer(self) -> None:
        for name in ("_core-report.yml", "regress-detector.yml", "normalize.yml"):
            with self.subTest(workflow=name):
                text = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
                self.assertIn("python -m hth.write_action_summary", text)
                self.assertNotIn("python hth-pipeline/hth/write_action_summary.py", text)

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

    def test_compaction_removes_links_to_omitted_detector_details(self) -> None:
        text = (
            '# Manifest\n\n<a id="table-of-contents"></a>\n<details open>\n'
            '<summary><strong>Navigation</strong></summary>\n\n'
            '- [Calibration](#detector-calibration-report)\n'
            '  - [Per-Detector Calibration Reports](#per-detector-calibration-reports)\n'
            '    - [A](#calibration-a)\n'
            '- [Regression](#detector-regression-reports)\n'
            '  - [Per-Detector Regression Reports](#per-detector-regression-reports)\n'
            '    - [A](#regression-a)\n\n</details>\n\n'
            '<a id="detector-calibration-report"></a>\n<details open>\n'
            '<summary><h2>Detector Calibration Report</h2></summary>\n'
            '<details open>\n<summary><h3>Per-Detector Calibration Reports</h3></summary>\n'
            '<a id="calibration-a"></a>\n<details><summary>A</summary>body</details>\n'
            '</details>\n</details>\n'
            '<a id="detector-regression-reports"></a>\n<details open>\n'
            '<summary><h2>Detector Regression Reports</h2></summary>\n'
            '<details open>\n<summary><h3>Per-Detector Regression Reports</h3></summary>\n'
            '<a id="regression-a"></a>\n<details><summary>A</summary>body</details>\n'
            '</details>\n</details>\n'
        )
        compacted, removed = compact_manifest(text)
        self.assertEqual(2, len(removed))
        self.assertIn('- [Calibration](#detector-calibration-report)', compacted)
        self.assertIn('- [Regression](#detector-regression-reports)', compacted)
        self.assertNotIn('](#per-detector-calibration-reports)', compacted)
        self.assertNotIn('](#per-detector-regression-reports)', compacted)
        self.assertNotIn('](#calibration-a)', compacted)
        self.assertNotIn('](#regression-a)', compacted)

    def test_truncation_prunes_links_to_later_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "report.md"
            destination = root / "summary.md"
            source.write_text(
                '# Report\n\n<a id="table-of-contents"></a>\n<details open>\n'
                '<summary><strong>Navigation</strong></summary>\n\n'
                '- [First](#first)\n- [Later](#later)\n\n</details>\n\n'
                '<a id="first"></a>\n## First\n\n' + ('filler ' * 60) +
                '\n\n<a id="later"></a>\n## Later\n',
                encoding="utf-8",
            )
            result = append_bounded_summary(source, destination, max_bytes=550)
            summary = destination.read_text(encoding="utf-8")
            self.assertTrue(result["truncated"])
            self.assertIn('](#first)', summary)
            self.assertNotIn('](#later)', summary)
            short_destination = root / "short-summary.md"
            append_bounded_summary(source, short_destination, max_bytes=350)
            short_summary = short_destination.read_text(encoding="utf-8")
            self.assertIn('<a id="table-of-contents"></a>', short_summary)
            self.assertNotIn('<details open>', short_summary)


if __name__ == "__main__":
    unittest.main()
