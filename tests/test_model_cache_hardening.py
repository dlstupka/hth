from __future__ import annotations

import contextlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hth.detector_lifecycle import (
    ORLI_MODEL_ID,
    ORLI_MODEL_SOURCES,
    ORLI_PACKAGE_VERSION,
    _download,
    _prepare_orli_page_mask_hook,
    _validate_safetensors_file,
)


def _write_safetensors(path: Path, payload: bytes = b"abcd") -> None:
    header = {
        "tensor": {
            "dtype": "U8",
            "shape": [len(payload)],
            "data_offsets": [0, len(payload)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # safetensors headers are padded to an 8-byte boundary.
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class ModelCacheHardeningTests(unittest.TestCase):
    def test_safetensors_validator_rejects_truncated_tensor_payload(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.safetensors"
            _write_safetensors(path)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(RuntimeError, "exceed data size|covers"):
                _validate_safetensors_file(path)

    def test_download_validates_before_atomic_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "model.safetensors"
            _write_safetensors(target, b"good")
            original = target.read_bytes()

            class Response:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def read(self, size=-1):
                    if getattr(self, "done", False):
                        return b""
                    self.done = True
                    return b"not-a-safetensors-file"

            with patch("hth.detector_lifecycle.urllib.request.urlopen", return_value=Response()):
                with self.assertRaises(RuntimeError):
                    _download("https://example.invalid/model", target, validator=_validate_safetensors_file)

            self.assertEqual(target.read_bytes(), original)

    def test_orli_reuse_repairs_only_corrupt_model_cache_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            results_root = Path(temp)
            root = results_root / "models" / ORLI_MODEL_ID
            model = root / "orli_base.safetensors"
            provenance = root / "model-provenance.json"
            unrelated = results_root / "models" / "other-model" / "keep.bin"
            unrelated.parent.mkdir(parents=True)
            unrelated.write_bytes(b"keep-me")

            root.mkdir(parents=True)
            model.write_bytes(b"truncated")
            provenance.write_text(
                json.dumps({"model_sha256": "deadbeef", "model_id": ORLI_MODEL_ID}) + "\n",
                encoding="utf-8",
            )

            downloads = []

            def fake_download(url, target, *, validator=None):
                downloads.append(url)
                _write_safetensors(Path(target), b"replacement")
                if validator is not None:
                    validator(target)

            output = io.StringIO()
            with (
                patch("hth.detector_lifecycle.importlib.util.find_spec", return_value=object()),
                patch("hth.detector_lifecycle.importlib.metadata.version", return_value=ORLI_PACKAGE_VERSION),
                patch("hth.detector_lifecycle._download", side_effect=fake_download),
                contextlib.redirect_stdout(output),
            ):
                payload = _prepare_orli_page_mask_hook(
                    results_root=results_root, policy="reuse", env_file=None
                )

            self.assertEqual(len(downloads), 1)
            self.assertTrue(model.is_file())
            _validate_safetensors_file(model)
            self.assertEqual(payload["model_sha256"], __import__("hashlib").sha256(model.read_bytes()).hexdigest())
            self.assertEqual(unrelated.read_bytes(), b"keep-me")
            log = output.getvalue()
            self.assertIn("Model cache invalid", log)
            self.assertIn("action=refetch-artifact", log)
            self.assertIn("orli_page_mask", log)

    def test_orli_download_falls_back_and_records_selected_source(self):
        with tempfile.TemporaryDirectory() as temp:
            results_root = Path(temp) / "results"
            calls = []

            def fake_download(url, target, *, validator=None):
                calls.append(url)
                if url == ORLI_MODEL_SOURCES[0].url:
                    raise OSError("synthetic primary outage")
                _write_safetensors(Path(target), b"fallback")
                if validator is not None:
                    validator(target)

            with (
                patch("hth.detector_lifecycle.importlib.util.find_spec", return_value=object()),
                patch("hth.detector_lifecycle.importlib.metadata.version", return_value=ORLI_PACKAGE_VERSION),
                patch("hth.detector_lifecycle._download", side_effect=fake_download),
                patch("hth.detector_lifecycle.time.sleep"),
            ):
                payload = _prepare_orli_page_mask_hook(
                    results_root=results_root, policy="refresh", env_file=None
                )

            self.assertEqual(calls, [
                ORLI_MODEL_SOURCES[0].url,
                ORLI_MODEL_SOURCES[0].url,
                ORLI_MODEL_SOURCES[1].url,
            ])
            self.assertEqual(payload["model_source_site"], ORLI_MODEL_SOURCES[1].site)
            self.assertEqual(payload["model_url"], ORLI_MODEL_SOURCES[1].url)
            self.assertEqual(len(payload["registered_model_sources"]), 3)


if __name__ == "__main__":
    unittest.main()
