from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hth.optimizer_dispatch import resolve_targets


class OptimizerDispatchTests(unittest.TestCase):
    def test_all_returns_every_configured_detector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector_dir = root / "detectors"
            detector_dir.mkdir()
            for name in ("b", "a", "c"):
                (detector_dir / f"{name}.json").write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_targets(detector_dir, root / "missing.json", "all"), ["a", "b", "c"])

    def test_all_without_preference_excludes_detectors_with_persisted_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector_dir = root / "detectors"
            detector_dir.mkdir()
            for name in ("a", "b", "c"):
                (detector_dir / f"{name}.json").write_text("{}", encoding="utf-8")
            index = root / "optimizer-index.json"
            index.write_text(json.dumps({
                "preferred_executor_configurations": [
                    {"detector_id": "a"},
                    {"detector_id": "c"},
                ]
            }), encoding="utf-8")
            self.assertEqual(resolve_targets(detector_dir, index, "all-without-preference"), ["b"])

    def test_missing_index_treats_every_detector_as_missing_preference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector_dir = root / "detectors"
            detector_dir.mkdir()
            (detector_dir / "a.json").write_text("{}", encoding="utf-8")
            self.assertEqual(resolve_targets(detector_dir, root / "missing.json", "all-without-preference"), ["a"])


if __name__ == "__main__":
    unittest.main()
