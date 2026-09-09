from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

from hth.artifact_mirror import MirrorArtifact
from hth.detector_lifecycle import (
    MODEL_DOWNLOAD_SOURCE_LIMIT,
    MODEL_PROVENANCE_FALLBACKS,
    _download_from_sources,
    _provenance_source,
    _publish_model_bundle_to_mirror,
    _publish_cached_model_bundle_if_missing,
    _restore_model_bundle_from_mirror,
    _write_model_provenance,
)
from hth.model_variants import ModelSource


class ModelDownloadFallbackTests(unittest.TestCase):
    mirror = MirrorArtifact("owner/mirror", "HTH-MIRROR-TEST", "model.pth", "model", "upstream", "ref", "MIT")

    def test_legacy_provenance_uses_recorded_model_release_as_reference(self):
        source = _provenance_source({
            "model_url": "https://example.invalid/model.pth",
            "model_release_version": "0.0.2",
        })
        self.assertEqual(source["site"], "recorded-provenance")
        self.assertEqual(source["reference"], "0.0.2")

    def test_legacy_pagenet_provenance_uses_sole_registered_source(self):
        source = _provenance_source({
            "weights_sha256": "abc",
            "registered_weights_sources": [{
                "site": "GitHub / PageNet",
                "url": "https://raw.githubusercontent.com/ctensmeyer/pagenet/master/model",
                "reference": "master",
            }],
        })
        self.assertEqual(source["site"], "GitHub / PageNet")
        self.assertEqual(source["reference"], "master")

    def test_legacy_kraken_provenance_identifies_bundled_package_resource(self):
        source = _provenance_source({
            "model_id": "kraken-blla-default-7.0.2",
            "kraken_version": "7.0.2",
            "upstream_repository": "https://github.com/mittagessen/kraken",
        })
        self.assertEqual(source["site"], "installed Kraken package resource")
        self.assertEqual(source["url"], "https://pypi.org/project/kraken/7.0.2/")
        self.assertEqual(source["reference"], "7.0.2")

    def test_multiple_registered_sources_are_not_reported_as_selected(self):
        source = _provenance_source({
            "model_release_version": "0.0.2",
            "registered_model_sources": [
                {"site": "primary", "reference": "main"},
                {"site": "fallback", "reference": "commit"},
            ],
        })
        self.assertEqual(source["site"], "recorded-provenance")
        self.assertEqual(source["reference"], "0.0.2")

    def test_every_registered_model_has_complete_legacy_fallback_provenance(self):
        self.assertEqual(len(MODEL_PROVENANCE_FALLBACKS), 8)
        for model_id, expected in MODEL_PROVENANCE_FALLBACKS.items():
            source = _provenance_source({"model_id": model_id})
            self.assertEqual(source["site"], expected["site"])
            self.assertTrue(source["url"], model_id)
            self.assertTrue(source["reference"], model_id)
            self.assertEqual(source["provenance_status"], "known-authoritative-source")

    def test_new_provenance_persists_selected_source_url_and_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model-provenance.json"
            _write_model_provenance(path, {
                "model_id": "test-model",
                "model_source": {
                    "site": "authoritative",
                    "url": "https://example.invalid/exact-model.bin",
                    "reference": "commit-sha",
                },
            })
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["artifact_source"]["url"], "https://example.invalid/exact-model.bin")
        self.assertEqual(payload["artifact_source"]["reference"], "commit-sha")
        self.assertEqual(payload["artifact_source"]["provenance_status"], "selected-download")

    def test_model_bundle_restore_replaces_partial_cache(self):
        bundle = MirrorArtifact("owner/mirror", "TAG", "model.zip", "model", "upstream", "ref", "MIT")

        def restore(_spec, target, *, fetch, validator):
            with zipfile.ZipFile(target, "w") as archive:
                archive.writestr("model-provenance.json", "{}")
                archive.writestr("model.bin", b"verified")
            validator(target)

        with tempfile.TemporaryDirectory() as temp, patch(
            "hth.detector_lifecycle.download_mirror", side_effect=restore
        ):
            root = Path(temp) / "model"
            root.mkdir()
            (root / "partial.bin").write_bytes(b"partial")
            self.assertTrue(_restore_model_bundle_from_mirror(bundle, root))
            self.assertFalse((root / "partial.bin").exists())
            self.assertEqual((root / "model.bin").read_bytes(), b"verified")

    def test_model_bundle_restore_validates_extracted_contents_before_install(self):
        bundle = MirrorArtifact("owner/mirror", "TAG", "model.zip", "model", "upstream", "ref", "MIT")

        def restore(_spec, target, *, fetch, validator):
            with zipfile.ZipFile(target, "w") as archive:
                archive.writestr("model-provenance.json", "{}")
                archive.writestr("saved_model.pb", b"corrupt")
            validator(target)

        with tempfile.TemporaryDirectory() as temp, patch(
            "hth.detector_lifecycle.download_mirror", side_effect=restore
        ):
            root = Path(temp) / "model"
            accepted = _restore_model_bundle_from_mirror(
                bundle, root, root_validator=lambda _root: (_ for _ in ()).throw(RuntimeError("bad model"))
            )
        self.assertFalse(accepted)
        self.assertFalse(root.exists())

    def test_fresh_model_bundle_is_published_without_runtime_bytecode(self):
        bundle = MirrorArtifact("owner/mirror", "TAG", "model.zip", "model", "upstream", "ref", "MIT")
        members = []

        def publish(_spec, artifact, *, authoritative_source):
            with zipfile.ZipFile(artifact) as archive:
                members.extend(archive.namelist())
            return "published"

        with tempfile.TemporaryDirectory() as temp, patch(
            "hth.detector_lifecycle.publish_mirror", side_effect=publish
        ):
            root = Path(temp) / "model"
            root.mkdir()
            (root / "model-provenance.json").write_text("{}", encoding="utf-8")
            (root / "model.bin").write_bytes(b"verified")
            bytecode = root / "__pycache__"
            bytecode.mkdir()
            (bytecode / "generated.pyc").write_bytes(b"ignore")
            _publish_model_bundle_to_mirror(bundle, root, {"site": "upstream"})

        self.assertEqual(members, ["model-provenance.json", "model.bin"])

    def test_valid_cache_backfills_a_missing_mirror(self):
        bundle = MirrorArtifact("owner/mirror", "TAG", "model.zip", "model", "upstream", "ref", "MIT")
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            "os.environ",
            {"HTH_ENABLE_MIRROR_PUBLICATION": "1", "HTH_RESULTS_TOKEN": "token"},
            clear=True,
        ), patch(
            "hth.detector_lifecycle.mirror_exists", return_value=False
        ), patch(
            "hth.detector_lifecycle._publish_model_bundle_to_mirror", return_value="published"
        ) as publication:
            status = _publish_cached_model_bundle_if_missing(
                bundle, Path(temp), {"model_id": "model", "model_source": {"site": "upstream"}}
            )
        self.assertEqual(status, "published")
        publication.assert_called_once()

    def test_valid_cache_does_not_republish_an_existing_mirror(self):
        bundle = MirrorArtifact("owner/mirror", "TAG", "model.zip", "model", "upstream", "ref", "MIT")
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            "os.environ",
            {"HTH_ENABLE_MIRROR_PUBLICATION": "1", "HTH_RESULTS_TOKEN": "token"},
            clear=True,
        ), patch(
            "hth.detector_lifecycle.mirror_exists", return_value=True
        ), patch("hth.detector_lifecycle._publish_model_bundle_to_mirror") as publication:
            status = _publish_cached_model_bundle_if_missing(bundle, Path(temp), {"model_id": "model"})
        self.assertEqual(status, "already-present")
        publication.assert_not_called()

    def test_verified_mirror_precedes_authoritative_sources(self):
        source = ModelSource("upstream", "https://upstream.example/model")
        with tempfile.TemporaryDirectory() as temp, patch(
            "hth.detector_lifecycle.download_mirror",
            return_value={"site": "HTH non-authoritative mirror", "tier": "mirror"},
        ) as mirror_download, patch("hth.detector_lifecycle._download_model_source") as upstream:
            selected = _download_from_sources(
                (source,), Path(temp) / "model.pth", artifact="model",
                variant="test_variant", mirror=self.mirror,
            )
        mirror_download.assert_called_once()
        upstream.assert_not_called()
        self.assertEqual(selected["tier"], "mirror")

    def test_mirror_miss_falls_back_to_authoritative_and_publishes(self):
        source = ModelSource("upstream", "https://upstream.example/model", "source-ref")

        def authoritative(_source, target, *, validator=None):
            Path(target).write_bytes(b"model")

        with tempfile.TemporaryDirectory() as temp, patch(
            "hth.detector_lifecycle.download_mirror", side_effect=OSError("mirror unavailable")
        ), patch(
            "hth.detector_lifecycle._download_model_source", side_effect=authoritative
        ), patch("hth.detector_lifecycle.publish_mirror", return_value="published") as publication:
            selected = _download_from_sources(
                (source,), Path(temp) / "model.pth", artifact="model",
                variant="test_variant", mirror=self.mirror,
            )
        self.assertEqual(selected["tier"], "authoritative")
        publication.assert_called_once()

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
