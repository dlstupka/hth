import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from hth.source_release import _download_verified_asset, _validate_manifest

ROOT = Path(__file__).resolve().parents[1]


class SourceReleaseContractTests(unittest.TestCase):
    def test_source_workflows_use_verified_release_assets_not_source_checkout(self):
        core = (ROOT / ".github/workflows/_core-hth.yml").read_text(encoding="utf-8")
        self.assertIn("Download immutable source release", core)
        self.assertIn("PYTHONPATH: hth-pipeline", core)
        self.assertIn("python -m hth.source_release", core)
        self.assertNotIn("python hth-pipeline/hth/source_release.py", core)
        self.assertIn("source_release_tag", core)
        self.assertNotIn("Checkout source repository", core)
        self.assertNotIn("lfs: true", core.lower())

    def test_source_release_module_bootstraps_from_pipeline_checkout_parent(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT)
        completed = subprocess.run(
            [sys.executable, "-m", "hth.source_release", "--help"],
            cwd=ROOT.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Download and verify immutable HTH source DOCX masters", completed.stdout)

    def test_preprocess_test_dispatch_can_select_source_release_and_limit(self):
        workflow = (ROOT / ".github/workflows/preprocess-test.yml").read_text(encoding="utf-8")
        dispatch = workflow.split("workflow_dispatch:", 1)[1].split("permissions:", 1)[0]
        self.assertIn("source_repository:", dispatch)
        self.assertIn("source_release_tag:", dispatch)
        self.assertIn('default: "HTH-SOURCE-0001"', dispatch)
        self.assertIn("image_limit:", dispatch)
        self.assertIn("default: 10", dispatch)

        core = (ROOT / ".github/workflows/_core-hth.yml").read_text(encoding="utf-8")
        self.assertIn('LIMIT_ARGS+=(--limit "$IMAGE_LIMIT")', core)

    def test_pipeline_has_no_binary_tracking_rule(self):
        attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8").lower()
        self.assertNotIn("filter=lfs", attrs)
        self.assertNotIn("diff=lfs", attrs)

    def test_manifest_requires_matching_repository_and_tag(self):
        manifest = {
            "schema_version": 1,
            "source_repository": "owner/source",
            "release_tag": "HTH-SOURCE-0001",
            "assets": [
                {
                    "name": "master.docx",
                    "size": 123,
                    "sha256": "a" * 64,
                }
            ],
        }
        assets = _validate_manifest(
            manifest,
            repository="owner/source",
            tag="HTH-SOURCE-0001",
        )
        self.assertEqual(assets[0]["name"], "master.docx")
        with self.assertRaises(RuntimeError):
            _validate_manifest(manifest, repository="other/source", tag="HTH-SOURCE-0001")
        with self.assertRaises(RuntimeError):
            _validate_manifest(manifest, repository="owner/source", tag="HTH-SOURCE-0002")

    def test_verified_asset_retries_a_clean_eof_partial_download(self):
        content = b"complete release asset"
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / "master.docx"
            attempts = 0

            def download(_asset, path, *, token=""):
                nonlocal attempts
                attempts += 1
                path.write_bytes(content[:5] if attempts == 1 else content)

            with mock.patch("hth.source_release._download_asset", side_effect=download), mock.patch(
                "hth.source_release.time.sleep"
            ):
                _download_verified_asset(
                    {"name": "master.docx"},
                    destination,
                    expected_size=len(content),
                    expected_sha256=digest,
                )

            self.assertEqual(attempts, 2)
            self.assertEqual(destination.read_bytes(), content)
            self.assertFalse(destination.with_name("master.docx.part").exists())

    def test_verified_asset_exhaustion_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / "master.docx"

            def download(_asset, path, *, token=""):
                path.write_bytes(b"partial")

            with mock.patch("hth.source_release._download_asset", side_effect=download), mock.patch(
                "hth.source_release.time.sleep"
            ):
                with self.assertRaisesRegex(RuntimeError, "after 3 attempts.*size mismatch"):
                    _download_verified_asset(
                        {"name": "master.docx"},
                        destination,
                        expected_size=100,
                        expected_sha256="a" * 64,
                        attempts=3,
                    )

            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name("master.docx.part").exists())

    def test_source_release_documentation_exists(self):
        text = (ROOT / "docs/source-releases.md").read_text(encoding="utf-8")
        self.assertIn("HTH-SOURCE-0001", text)
        self.assertIn("source-release-manifest.json", text)
        self.assertIn("publish-source-release.py", text)


if __name__ == "__main__":
    unittest.main()
