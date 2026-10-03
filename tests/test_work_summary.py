from __future__ import annotations

import json
import io
import tempfile
import unittest
import urllib.error
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from hth import work_summary


ROOT = Path(__file__).parents[1]


class WorkSummaryTests(unittest.TestCase):
    def test_github_collector_retries_a_timed_out_page(self) -> None:
        payload = io.BytesIO(b'{"workflow_runs": []}')
        with patch.object(work_summary.urllib.request, "urlopen",
                          side_effect=[TimeoutError("read timed out"), payload]) as opener, \
             patch.object(work_summary.time, "sleep") as sleep:
            result = work_summary._github_get("https://api.github.com/repos/o/r/actions/runs?page=2", "token")
        self.assertEqual(result, {"workflow_runs": []})
        self.assertEqual(opener.call_count, 2)
        self.assertEqual(sleep.call_args.args, (2,))

    def test_github_collector_does_not_retry_authorization_failure(self) -> None:
        error = urllib.error.HTTPError("https://api.github.com/repos/o/r", 401, "Unauthorized", {}, None)
        with patch.object(work_summary.urllib.request, "urlopen", side_effect=error) as opener, \
             patch.object(work_summary.time, "sleep") as sleep:
            with self.assertRaises(urllib.error.HTTPError):
                work_summary._github_get("https://api.github.com/repos/o/r", "token")
        self.assertEqual(opener.call_count, 1)
        sleep.assert_not_called()

    def test_chicago_fallback_respects_dst_transition(self) -> None:
        before = datetime(2026, 3, 8, 7, 30, tzinfo=timezone.utc).astimezone(work_summary.ZONE)
        after = datetime(2026, 3, 8, 8, 30, tzinfo=timezone.utc).astimezone(work_summary.ZONE)
        self.assertEqual((before.hour, before.utcoffset().total_seconds()), (1, -21600))
        self.assertEqual((after.hour, after.utcoffset().total_seconds()), (3, -18000))
        fall_before = datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc).astimezone(work_summary.ZONE)
        fall_after = datetime(2026, 11, 1, 7, 30, tzinfo=timezone.utc).astimezone(work_summary.ZONE)
        self.assertEqual((fall_before.hour, fall_before.utcoffset().total_seconds()), (1, -18000))
        self.assertEqual((fall_after.hour, fall_after.utcoffset().total_seconds()), (1, -21600))

    def test_historical_ranges_and_new_day_proxies_are_labeled(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "reports" / "hth-work-summary"
            github = root / "github.json"
            github.write_text(json.dumps({
                "schema": work_summary.SCHEMA,
                "runs": [{"id": 1, "name": "HTH report", "status": "completed", "conclusion": "success",
                          "created_at": "2026-10-02T15:00:00Z", "run_started_at": "2026-10-02T15:00:00Z",
                          "updated_at": "2026-10-02T16:00:00Z", "html_url": "https://github.com/o/r/actions/runs/1"}],
                "releases": [{"repository": "o/r", "tag": "v1", "name": "First", "published_at": "2026-10-02T16:00:00Z",
                              "html_url": "https://github.com/o/r/releases/tag/v1"}],
            }), encoding="utf-8")
            commits = {
                "2026-07-10": [{"sha": "a" * 40, "author": "Dan Stupka", "subject": "Initial capture"}],
                "2026-07-11": [{"sha": "b" * 40, "author": "Dan Stupka", "subject": "Improve capture"},
                               {"sha": "c" * 40, "author": "Dan Stupka", "subject": "Add pipeline"}],
                "2026-10-02": [{"sha": "d" * 40, "author": "Dan Stupka", "subject": "Add work report"}],
            }
            with patch.object(work_summary, "_git_commits", return_value=commits):
                summary = work_summary.generate(ROOT, root, output, date(2026, 10, 3),
                    ROOT / "config" / "work-summary-estimates.json", github)
            july = json.loads((output / "months" / "2026-07.json").read_text(encoding="utf-8"))
            self.assertEqual(july["estimate_ranges"]["human_hours"], [90.0, 140.0])
            july_10 = json.loads((output / "days" / "2026-07-10.json").read_text(encoding="utf-8"))
            self.assertEqual(july_10["estimate_ranges"]["human_hours"], [30.0, 46.667])
            june_10 = json.loads((output / "days" / "2026-06-10.json").read_text(encoding="utf-8"))
            self.assertIsNone(june_10["estimate_ranges"]["human_hours"])
            october = json.loads((output / "days" / "2026-10-02.json").read_text(encoding="utf-8"))
            self.assertEqual(october["estimate_ranges"]["human_hours"], [0.4, 2.0])
            self.assertEqual(october["estimate_ranges"]["compute_core_hours"], [1.0, 192.0])
            self.assertEqual(summary["workflow_runs"], 1)
            self.assertEqual(summary["releases"], 1)
            self.assertEqual(summary["commits"], 4)
            self.assertEqual(summary["contributors"], {"Dan Stupka": 4})
            annual = json.loads((output / "years" / "2026.json").read_text(encoding="utf-8"))
            self.assertEqual(annual["estimate_ranges"]["human_hours"], summary["estimate_ranges"]["human_hours"])
            self.assertEqual(annual["workflow_runs"], 1)
            self.assertEqual(annual["releases"], 1)
            self.assertFalse(annual["closed"])
            report = (output / "summary.md").read_text(encoding="utf-8")
            self.assertIn("## Annual", report)
            self.assertIn("2026 YTD", report)
            self.assertIn("API-equivalent", report)
            self.assertIn("not billed subscription", report)
            self.assertIn("days reused", report)

    def test_closed_day_and_month_reuse_avoids_history_rescan(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "reports" / "hth-work-summary"
            commits = {"2026-07-10": [{"sha": "a" * 40, "author": "Dan", "subject": "Start"}]}
            with patch.object(work_summary, "_git_commits", return_value=commits):
                work_summary.generate(ROOT, root, output, date(2026, 10, 3),
                    ROOT / "config" / "work-summary-estimates.json")
            with patch.object(work_summary, "_git_commits", return_value={}) as git_log:
                summary = work_summary.generate(ROOT, root, output, date(2026, 10, 4),
                    ROOT / "config" / "work-summary-estimates.json")
            self.assertEqual(git_log.call_args.args[1], date(2026, 10, 3))
            self.assertGreater(summary["cache"]["days_reused"], 100)
            self.assertEqual(summary["cache"]["days_built"], 2)
            self.assertGreaterEqual(summary["cache"]["months_reused"], 4)
            self.assertEqual(work_summary.pending_since(root, date(2026, 10, 5)), date(2026, 10, 4))

    def test_closed_year_is_persisted_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "reports" / "hth-work-summary"
            ledger = ROOT / "config" / "work-summary-estimates.json"
            with patch.object(work_summary, "_git_commits", return_value={}):
                work_summary.generate(ROOT, root, output, date(2027, 1, 1), ledger)
                summary = work_summary.generate(ROOT, root, output, date(2027, 1, 2), ledger)
            self.assertEqual(summary["cache"]["years_reused"], 1)
            self.assertEqual(summary["cache"]["years_built"], 1)
            annual = json.loads((output / "years" / "2026.json").read_text(encoding="utf-8"))
            self.assertTrue(annual["closed"])
            report = (output / "summary.md").read_text(encoding="utf-8")
            self.assertIn("| 2026 |", report)
            self.assertIn("| 2027 YTD |", report)

    def test_github_collector_records_facts_without_inventing_cpu_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "github.json"
            def response(url, token):
                if "/actions/runs?" in url:
                    return {"workflow_runs": [{"id": 7, "name": "Build", "status": "completed",
                                              "conclusion": "success", "created_at": "2026-10-02T12:00:00Z"}]}
                return [{"tag_name": "v1", "name": "Release", "published_at": "2026-10-02T12:00:00Z"}]
            with patch.object(work_summary, "_github_get", side_effect=response):
                work_summary.collect_github("o/r", ["o/r"], date(2026, 10, 2), date(2026, 10, 3), output)
            snapshot = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(snapshot["runs"][0]["id"], 7)
            self.assertEqual(snapshot["releases"][0]["tag"], "v1")
            self.assertNotIn("cpu_hours", snapshot["runs"][0])


if __name__ == "__main__":
    unittest.main()
