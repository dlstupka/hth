from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from hth.detector_lifecycle import MODEL_DOWNLOAD_SOURCE_LIMIT, _download_from_sources
from hth.model_variants import ModelSource


class ModelDownloadFallbackTests(unittest.TestCase):
    def test_validation_failure_retries_the_same_source_before_fallback(self):
        source = ModelSource("primary.example", "https://primary.example/model")
        calls = []

        def fake_download(url, target):
            calls.append(url)
            if len(calls) == 1:
                raise RuntimeError("truncated validated artifact")
            Path(target).write_bytes(b"complete")

        with tempfile.TemporaryDirectory() as temp, patch(
            "hth.detector_lifecycle._download", side_effect=fake_download
        ), patch("hth.detector_lifecycle.time.sleep") as sleep:
            selected = _download_from_sources(
                (source,), Path(temp) / "model.pth",
                artifact="model", variant="test_variant",
            )

        self.assertEqual(calls, [source.url, source.url])
        sleep.assert_called_once()
        self.assertEqual(selected["site"], source.site)

    def test_permanent_http_error_falls_back_without_retrying_source(self):
        sources = (
            ModelSource("missing.example", "https://missing.example/model"),
            ModelSource("mirror.example", "https://mirror.example/model"),
        )
        calls = []

        def fake_download(url, target):
            calls.append(url)
            if "missing.example" in url:
                raise urllib.error.HTTPError(url, 404, "not found", {}, None)
            Path(target).write_bytes(b"model")

        with tempfile.TemporaryDirectory() as temp, patch(
            "hth.detector_lifecycle._download", side_effect=fake_download
        ), patch("hth.detector_lifecycle.time.sleep") as sleep:
            selected = _download_from_sources(
                sources, Path(temp) / "model.pth",
                artifact="model", variant="test_variant",
            )

        self.assertEqual(calls, [sources[0].url, sources[1].url])
        sleep.assert_not_called()
        self.assertEqual(selected["site"], sources[1].site)

    def test_transient_http_error_retries_the_same_source(self):
        source = ModelSource("busy.example", "https://busy.example/model")
        calls = []

        def fake_download(url, target):
            calls.append(url)
            if len(calls) == 1:
                raise urllib.error.HTTPError(url, 503, "service unavailable", {}, None)
            Path(target).write_bytes(b"model")

        with tempfile.TemporaryDirectory() as temp, patch(
            "hth.detector_lifecycle._download", side_effect=fake_download
        ), patch("hth.detector_lifecycle.time.sleep") as sleep:
            selected = _download_from_sources(
                (source,), Path(temp) / "model.pth",
                artifact="model", variant="test_variant",
            )

        self.assertEqual(calls, [source.url, source.url])
        sleep.assert_called_once()
        self.assertEqual(selected["site"], source.site)

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
        ), patch("hth.detector_lifecycle.time.sleep"):
            target = Path(temp) / "model.pth"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                selected = _download_from_sources(
                    sources, target, artifact="model", variant="test_variant"
                )

        log = output.getvalue()
        self.assertEqual(calls, [sources[0].url, sources[0].url, sources[1].url])
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
        ), patch("hth.detector_lifecycle.time.sleep"):
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "one: OSError: offline.*two: OSError: offline.*three: OSError: offline",
                ):
                    _download_from_sources(
                        sources,
                        Path(temp) / "model.pth",
                        artifact="model",
                        variant="test_variant",
                    )

        self.assertIn("site=one", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")

    def test_source_count_is_bounded_to_three(self):
        sources = tuple(
            ModelSource(f"site-{index}", f"https://site-{index}.invalid/model")
            for index in range(MODEL_DOWNLOAD_SOURCE_LIMIT + 1)
        )
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "maximum is 3"):
                _download_from_sources(
                    sources, Path(temp) / "model.pth", artifact="model", variant="too_many"
                )


if __name__ == "__main__":
    unittest.main()
