from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from hth.optimizer_dispatch import _dispatch, resolve_targets


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

    def test_dispatch_accepts_any_2xx_response(self) -> None:
        class FakeResponse:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return None

        with mock.patch("hth.optimizer_dispatch.urllib.request.urlopen", return_value=FakeResponse()):
            _dispatch(
                "https://api.github.test/dispatches",
                "token",
                "main",
                "adaptive_radial_edge",
                {"pipeline_enumeration": "adaptive"},
            )

    def test_dispatch_rejects_non_2xx_response(self) -> None:
        class FakeResponse:
            status = 300
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return None

        with mock.patch("hth.optimizer_dispatch.urllib.request.urlopen", return_value=FakeResponse()):
            with self.assertRaisesRegex(RuntimeError, "Unexpected dispatch status 300"):
                _dispatch(
                    "https://api.github.test/dispatches",
                    "token",
                    "main",
                    "adaptive_radial_edge",
                    {"pipeline_enumeration": "adaptive"},
                )


if __name__ == "__main__":
    unittest.main()
