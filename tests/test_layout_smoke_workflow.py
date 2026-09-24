from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "layout-smoke-run.py"
SPEC = importlib.util.spec_from_file_location("layout_smoke_run", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
INPUTS_SPEC = importlib.util.spec_from_file_location("layout_smoke_inputs", ROOT / "tools" / "layout-smoke-inputs.py")
assert INPUTS_SPEC is not None and INPUTS_SPEC.loader is not None
INPUTS_MODULE = importlib.util.module_from_spec(INPUTS_SPEC)
INPUTS_SPEC.loader.exec_module(INPUTS_MODULE)


class LayoutSmokeWorkflowTests(unittest.TestCase):
    def test_older_golden_set_materializes_source_without_normalization_claim(self) -> None:
        freeze_path = ROOT / "config/golden_sets/HTH-0001.freeze.json"
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            images = Path(directory) / "images"
            for row in freeze["image_bundle"]["images"]:
                target = images / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"image fixture")
            expected_sha = {str(images / row["path"]): row["sha256"] for row in freeze["image_bundle"]["images"]}
            expected_sha[str(ROOT / freeze["golden_set_path"])] = freeze["golden_set_sha256"]
            with patch.object(INPUTS_MODULE, "_sha256", side_effect=lambda path: expected_sha[str(path)]), patch.object(
                INPUTS_MODULE.cv2, "imread", return_value=np.zeros((2, 2, 3), dtype=np.uint8)
            ):
                result = INPUTS_MODULE.materialize(
                    freeze_path, images, Path("unused"), Path("unused"), Path("unused"), Path("unused"),
                    Path(directory) / "out", "", source_only=True,
                )
            self.assertEqual(result["views"], ["source"])
            self.assertEqual(len(result["pages"]), 5)
            self.assertIsNone(result["final_normalization_result_identity"])
            self.assertNotIn("normalized_file", result["pages"][0])

    def test_workflow_is_manual_read_only_and_uses_managed_runtime(self) -> None:
        workflow = (ROOT / ".github/workflows/layout-smoke.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("setup-hth-python", workflow)
        self.assertIn("setup-hth-managed-runtime", workflow)
        self.assertIn('need-kraken: "true"', workflow)
        self.assertIn("golden_set_release", workflow)
        self.assertIn("layout-smoke-inputs.py", workflow)
        self.assertIn("layout-smoke-run.py", workflow)
        self.assertIn('default: smoke', workflow)
        self.assertIn('          - full', workflow)
        self.assertIn('--mode "${{ inputs.mode }}"', workflow)
        self.assertIn("Full collection layout is not wired yet", workflow)
        self.assertIn("/normalization/binarization-integration/binarization-normalization-manifest.json", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertNotIn("hth_hardened_persist", workflow)
        self.assertIn("$RUNNER_TEMP/layout-pairs", workflow)
        self.assertNotIn("$RUNNER_TEMP/layout-smoke-pairs", workflow)

    def test_progress_log_uses_neutral_layout_label(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertIn('print(f"Layout: {results.name}', runner)
        self.assertNotIn("Layout smoke:", runner)

    def test_batch_command_is_argv_and_contains_both_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").mkdir()
            source = root / "source" / "fs_0003.png"
            source.write_bytes(b"fixture")
            results = root / "results"
            results.mkdir()
            command = MODULE._command(
                "kraken", [{"global_ordinal": 3, "source_file": "source/fs_0003.png"}],
                root, results, "source", 4,
            )
            self.assertEqual(command[:6], ["kraken", "-d", "cpu", "--threads", "4", "--raise-on-error"])
            self.assertEqual(command[-2:], ["segment", "-bl"])
            self.assertIn(str(source.resolve()), command)
            self.assertIn(str((results / "fs_0003.json").resolve()), command)

    def test_smoke_preserves_complete_frozen_membership_and_full_rejects_it(self) -> None:
        freeze = json.loads((ROOT / "config/golden_sets/HTH-GOLDEN-0002.freeze.json").read_text(encoding="utf-8"))
        ordinals = freeze["membership"]["global_ordinals"]
        self.assertEqual(freeze["membership"]["page_count"], 18)
        pages = [{"global_ordinal": ordinal} for ordinal in ordinals]
        self.assertEqual(MODULE._select_pages(pages, "smoke"), pages)
        self.assertEqual(MODULE._select_pages(pages[:5], "smoke"), pages[:5])
        with self.assertRaisesRegex(ValueError, "929 collection pages"):
            MODULE._select_pages(pages, "full")
        collection = [{"global_ordinal": ordinal} for ordinal in range(1, 930)]
        self.assertEqual(MODULE._select_pages(collection, "full", "layout-collection-inputs"), collection)
        with self.assertRaises(ValueError):
            MODULE._select_pages(pages, "invalid")

    def test_batch_runner_records_bounded_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "source"
            results.mkdir()
            fake = root / "fake.py"
            fake.write_text(
                "import pathlib,sys\n"
                "pathlib.Path(sys.argv[1]).write_text('{}', encoding='utf-8')\n"
                "print('Polygonizer failed on line 0')\n",
                encoding="utf-8",
            )
            diagnostic = MODULE._run_view(
                [sys.executable, str(fake), str(results / "fs_0003.json")],
                results, root / "source.log", 1,
            )
            self.assertEqual(diagnostic["pages"], 1)
            self.assertEqual(diagnostic["polygonizer_warnings"], 1)
            self.assertGreaterEqual(diagnostic["batch_wall_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
