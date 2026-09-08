from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hth.artifact_mirror import MirrorArtifact, download, publish


SPEC = MirrorArtifact(
    repository="example/hth-mirror",
    tag="HTH-MIRROR-TEST-1",
    asset_name="model.bin",
    artifact_id="test-model",
    authoritative_repository="https://authoritative.example/models",
    authoritative_reference="doi:test",
    license="MIT",
)


def _manifest(payload: bytes) -> dict[str, object]:
    return {
        "schema_version": "1",
        "artifact_id": SPEC.artifact_id,
        "mirror_repository": SPEC.repository,
        "mirror_tag": SPEC.tag,
        "asset_name": SPEC.asset_name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "license": SPEC.license,
        "authoritative_repository": SPEC.authoritative_repository,
        "authoritative_reference": SPEC.authoritative_reference,
        "trust_role": "non-authoritative redundancy mirror",
    }


class ArtifactMirrorTests(unittest.TestCase):
    def test_download_requires_matching_manifest_and_sha(self):
        payload = b"verified model"

        def fetch(url: str, target: Path) -> None:
            if url.endswith(".manifest.json"):
                target.write_text(json.dumps(_manifest(payload)), encoding="utf-8")
            else:
                target.write_bytes(payload)

        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "model.bin"
            selected = download(SPEC, target, fetch=fetch)
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(selected["tier"], "mirror")

    def test_download_rejects_sha_mismatch_without_replacing_target(self):
        def fetch(url: str, target: Path) -> None:
            if url.endswith(".manifest.json"):
                target.write_text(json.dumps(_manifest(b"expected")), encoding="utf-8")
            else:
                target.write_bytes(b"tampered")

        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "model.bin"
            target.write_bytes(b"existing")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                download(SPEC, target, fetch=fetch)
            self.assertEqual(target.read_bytes(), b"existing")

    def test_download_rejects_provenance_mismatch(self):
        manifest = _manifest(b"model")
        manifest["authoritative_reference"] = "wrong"

        def fetch(url: str, target: Path) -> None:
            target.write_text(json.dumps(manifest), encoding="utf-8")

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "authoritative_reference mismatch"):
                download(SPEC, Path(temp) / "model.bin", fetch=fetch)

    def test_publish_without_token_is_an_explicit_noop(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", {}, clear=True):
            artifact = Path(temp) / "model.bin"
            artifact.write_bytes(b"model")
            self.assertEqual(
                publish(SPEC, artifact, authoritative_source={"site": "upstream"}),
                "skipped-no-token",
            )


if __name__ == "__main__":
    unittest.main()
