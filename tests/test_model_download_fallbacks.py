from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hth.detector_lifecycle import _download_from_sources
from hth.model_variants import ModelSource


class ModelDownloadFallbackTests(unittest.TestCase):
    def test_fallback_logs_failed_site_and_successful_site(self):
        sources = (
            ModelSource("primary.example", "https://primary.example/model", "primary-ref"),
            ModelSource("mirror.example", "https://mirror.example/model", "mirror-ref"),
        )
        calls = []

        def fake_download(url, target):
            calls.append(url)
            if "primary.example" in url:
                raise OSError("synthetic primary outage")
            Path(target).write_bytes(b"model")

        with tempfile.TemporaryDirectory() as temp, patch(
            "hth.detector_lifecycle._download", side_effect=fake_download
        ):
            target = Path(temp) / "model.pth"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                selected = _download_from_sources(
                    sources, target, artifact="model", variant="test_variant"
                )

        log = output.getvalue()
        self.assertEqual(calls, [sources[0].url, sources[1].url])
        self.assertEqual(selected["site"], "mirror.example")
        self.assertIn("site=primary.example", log)
        self.assertIn("synthetic primary outage", log)
        self.assertIn("site=mirror.example", log)
        self.assertIn("Model download succeeded", log)

    def test_all_sources_reported_when_every_download_fails(self):
        sources = (
            ModelSource("one", "https://one.invalid/model"),
            ModelSource("two", "https://two.invalid/model"),
            ModelSource("three", "https://three.invalid/model"),
        )
        with tempfile.TemporaryDirectory() as temp, patch(
            "hth.detector_lifecycle._download", side_effect=OSError("offline")
        ):
            with self.assertRaisesRegex(RuntimeError, "one: OSError: offline.*two: OSError: offline.*three: OSError: offline"):
                _download_from_sources(
                    sources, Path(temp) / "model.pth", artifact="model", variant="test_variant"
                )


if __name__ == "__main__":
    unittest.main()
