from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hth.regression.runner import load_or_evaluate_shared_baseline


class SharedBaselineTests(unittest.TestCase):
    def test_shared_baseline_is_evaluated_once_then_reused(self) -> None:
        calls = 0
        payload = {
            "parameter_set_id": "baseline",
            "parameters": {"threshold": 0.5},
            "summary": {"mean_iou": 0.8, "failure_count": 0},
            "pages": [],
        }

        def evaluate():
            nonlocal calls
            calls += 1
            return dict(payload)

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "baseline.json"
            first, first_reused = load_or_evaluate_shared_baseline(cache, evaluate)
            second, second_reused = load_or_evaluate_shared_baseline(cache, evaluate)

        self.assertEqual(calls, 1)
        self.assertFalse(first_reused)
        self.assertTrue(second_reused)
        self.assertEqual(first, payload)
        self.assertEqual(second, payload)

    def test_no_shared_path_preserves_direct_evaluation(self) -> None:
        calls = 0

        def evaluate():
            nonlocal calls
            calls += 1
            return {"parameters": {}}

        _, reused = load_or_evaluate_shared_baseline(None, evaluate)
        self.assertEqual(calls, 1)
        self.assertFalse(reused)


if __name__ == "__main__":
    unittest.main()
