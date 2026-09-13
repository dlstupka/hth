from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error

from hth.collection_cache import (
    EvidenceCacheArtifact,
    canonical_evidence_id,
    deterministic_evidence_bundle,
    download,
    materialize,
    publish,
    sha256_file,
    validate_evidence_bundle,
    validate_evidence_manifest,
)


class CollectionCacheTests(unittest.TestCase):
    def _manifest(self, root: Path) -> tuple[Path, dict]:
        identity = {
            "schema_version": "1.0",
            "detector": "orli_page_mask",
            "model_id": "orli-base-2026",
            "model_sha256": "a" * 64,
            "golden_set_sha256": "b" * 64,
            "maximum_dimension": 1800,
            "image_keys": ["page-a", "page-b"],
            "evidence_representation": "immutable-json",
        }
        evidence_id = canonical_evidence_id(identity)
        payload = {
            "schema_version": "0.1",
            "detector": "orli_page_mask",
            "page_count": 2,
            "representation": "immutable-json",
            "records": [
                {"image_key": "page-a", "evidence": {"regions": []}},
                {"image_key": "page-b", "evidence": {"regions": []}},
            ],
            "persistence": {
                "schema_version": "1.0",
                "evidence_id": evidence_id,
                "identity": identity,
            },
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path, payload

    def test_manifest_and_index_identity_are_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            path, payload = self._manifest(Path(temp))
            evidence_id = payload["persistence"]["evidence_id"]
            validated = validate_evidence_manifest(path, index_entry={
                "evidence_id": evidence_id,
                "manifest_sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "page_count": 2,
                "image_keys": ["page-a", "page-b"],
            })
            self.assertEqual(validated["evidence_id"], evidence_id)

    def test_manifest_rejects_record_order_that_differs_from_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path, payload = self._manifest(Path(temp))
            payload["records"].reverse()
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "record order"):
                validate_evidence_manifest(path)

    def test_evidence_bundle_is_reproducible_and_round_trips(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            root = workspace / "evidence"
            root.mkdir()
            manifest, payload = self._manifest(root)
            first, second = root / "first.zip", root / "second.zip"
            deterministic_evidence_bundle(manifest, first)
            deterministic_evidence_bundle(manifest, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            validated = validate_evidence_bundle(first)
            self.assertEqual(validated["evidence_id"], payload["persistence"]["evidence_id"])

    def test_release_names_are_deterministic_and_identity_scoped(self):
        identity = {"detector": "orli_page_mask"}
        evidence_id = canonical_evidence_id(identity)
        spec = EvidenceCacheArtifact("owner/cache", "orli_page_mask", evidence_id, identity)
        self.assertEqual(spec.asset_name, f"orli_page_mask-{evidence_id}.zip")
        self.assertEqual(spec.tag, f"HTH-EVIDENCE-ORLI-PAGE-MASK-{evidence_id.upper()}")

    def test_bundle_includes_and_materializes_referenced_sidecars(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            root = workspace / "evidence"
            root.mkdir()
            manifest, payload = self._manifest(root)
            payload["records"][0]["file"] = "page-a.npy"
            payload["records"][1]["file"] = "page-b.npy"
            (root / "page-a.npy").write_bytes(b"array-a")
            (root / "page-b.npy").write_bytes(b"array-b")
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            bundle = workspace / "evidence.zip"
            deterministic_evidence_bundle(manifest, bundle)
            validate_evidence_bundle(bundle)
            output = workspace / "hydrated"
            hydrated = materialize(bundle, output)
            self.assertTrue(hydrated.is_file())
            self.assertEqual((output / "page-a.npy").read_bytes(), b"array-a")
            self.assertEqual((output / "page-b.npy").read_bytes(), b"array-b")

    def test_download_validates_release_and_bundled_evidence_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, payload = self._manifest(root)
            identity = payload["persistence"]["identity"]
            evidence_id = payload["persistence"]["evidence_id"]
            spec = EvidenceCacheArtifact("owner/cache", "orli_page_mask", evidence_id, identity)
            bundle = root / spec.asset_name
            deterministic_evidence_bundle(manifest, bundle)
            release_manifest = {
                "schema_version": "1",
                "artifact_kind": "learned-evidence",
                "cache_repository": spec.repository,
                "cache_tag": spec.tag,
                "asset_name": spec.asset_name,
                "detector": spec.detector,
                "evidence_id": spec.evidence_id,
                "evidence_identity": spec.identity,
                "trust_role": "non-authoritative collection cache",
                "size_bytes": bundle.stat().st_size,
                "sha256": sha256_file(bundle),
                "evidence_manifest_sha256": sha256_file(manifest),
            }

            def fetch(url: str, target: Path) -> None:
                if url.endswith(".manifest.json"):
                    target.write_text(json.dumps(release_manifest), encoding="utf-8")
                else:
                    target.write_bytes(bundle.read_bytes())

            target = root / "downloaded.zip"
            returned = download(spec, target, fetch=fetch)
            self.assertEqual(returned["evidence_id"], evidence_id)
            self.assertEqual(target.read_bytes(), bundle.read_bytes())

            wrong = dict(release_manifest)
            wrong["evidence_id"] = "0" * 64
            with mock.patch.dict(release_manifest, wrong, clear=True):
                with self.assertRaisesRegex(RuntimeError, "evidence_id mismatch"):
                    download(spec, root / "rejected.zip", fetch=fetch)

    def test_publish_reuses_release_created_by_a_competing_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, payload = self._manifest(root)
            identity = payload["persistence"]["identity"]
            evidence_id = payload["persistence"]["evidence_id"]
            spec = EvidenceCacheArtifact("owner/cache", "orli_page_mask", evidence_id, identity)
            bundle = root / spec.asset_name
            deterministic_evidence_bundle(manifest, bundle)
            not_found = urllib.error.HTTPError("tag", 404, "missing", {}, None)
            create_race = urllib.error.HTTPError("release", 422, "already exists", {}, None)
            winner = {"assets": [], "upload_url": "https://uploads.example/release{?name,label}"}
            with mock.patch("hth.collection_cache._github_json", side_effect=[not_found, create_race, winner]), \
                 mock.patch("hth.collection_cache._upload") as upload:
                status = publish(
                    spec,
                    bundle,
                    evidence_manifest_sha256=sha256_file(manifest),
                    token="test-token",
                )
            self.assertEqual(status, "published")
            self.assertEqual(upload.call_count, 2)
