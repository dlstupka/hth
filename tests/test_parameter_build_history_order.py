import unittest

from hth.write_regression_summary import _known_builds_for_parameter


class ParameterBuildHistoryOrderTests(unittest.TestCase):
    def test_current_run_is_pinned_before_newest_to_oldest_prior_history(self):
        index = {
            "kraken_page_mask": [
                {
                    "explicit_shas": ["same-sha"],
                    "build_number": "488",
                    "build_url": "https://example.invalid/488",
                    "date": "2026-08-17",
                    "evidence": "authoritative",
                },
                {
                    "explicit_shas": ["same-sha"],
                    "build_number": "474",
                    "build_url": "https://example.invalid/474",
                    "date": "2026-08-17",
                    "evidence": "authoritative",
                },
                {
                    "explicit_shas": ["same-sha"],
                    "build_number": "445",
                    "build_url": "https://example.invalid/445",
                    "date": "2026-08-16",
                    "evidence": "authoritative",
                },
            ]
        }
        rows = _known_builds_for_parameter(
            detector="kraken_page_mask",
            full_sha="same-sha",
            parameters=None,
            info={
                "github_run_url": "https://example.invalid/current",
                "started_at_utc": "2026-08-17T19:00:00Z",
            },
            run_url="https://example.invalid/current",
            parameter_build_index=index,
        )

        self.assertEqual(rows[0][3], "current run")
        self.assertEqual([row[0] for row in rows[1:]], ["488", "474", "445"])


if __name__ == "__main__":
    unittest.main()
