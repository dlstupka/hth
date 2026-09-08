from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "tools" / "migrate-results-models.py"
SPEC = importlib.util.spec_from_file_location("migrate_results_models", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MigrateResultsModelsTests(unittest.TestCase):
    def test_selected_missing_model_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "models").mkdir()
            with self.assertRaisesRegex(RuntimeError, "Requested model cache"):
                MODULE.seed(
                    Path(temp), token=None, dry_run=True,
                    selected_models=["orli-base-2026"],
                )

    def test_validation_accepts_matching_provenance_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp) / "example-model"
            model_dir.mkdir()
            model = model_dir / "model.bin"
            model.write_bytes(b"model")
            (model_dir / "model-provenance.json").write_text(json.dumps({
                "model_id": model_dir.name,
                "model_filename": model.name,
                "model_sha256": MODULE.sha256(model),
                "license": "MIT",
                "upstream_repository": "https://example.invalid/model",
            }), encoding="utf-8")
            result = MODULE.validate_model_dir(model_dir)
            self.assertEqual(result["verified_files"], ["model.bin"])

    def test_validation_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp) / "example-model"
            model_dir.mkdir()
            (model_dir / "model.bin").write_bytes(b"model")
            (model_dir / "model-provenance.json").write_text(json.dumps({
                "model_id": model_dir.name,
                "model_filename": "model.bin",
                "model_sha256": "0" * 64,
                "license": "MIT",
                "upstream_repository": "https://example.invalid/model",
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                MODULE.validate_model_dir(model_dir)

    def test_bundle_is_reproducible_and_excludes_runtime_bytecode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "a.bin").write_bytes(b"a")
            cache = model_dir / "__pycache__"
            cache.mkdir()
            (cache / "generated.pyc").write_bytes(b"bytecode")
            first, second = root / "first.zip", root / "second.zip"
            MODULE.deterministic_bundle(model_dir, first)
            MODULE.deterministic_bundle(model_dir, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotIn(b"generated.pyc", first.read_bytes())


if __name__ == "__main__":
    unittest.main()
