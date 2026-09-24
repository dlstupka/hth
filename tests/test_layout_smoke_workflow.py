from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "layout-smoke-run.py"
SPEC = importlib.util.spec_from_file_location("layout_smoke_run", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LayoutSmokeWorkflowTests(unittest.TestCase):
    def test_workflow_is_manual_read_only_and_uses_managed_runtime(self) -> None:
        workflow = (ROOT / ".github/workflows/layout-smoke.yml").read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("setup-hth-python", workflow)
        self.assertIn("setup-hth-managed-runtime", workflow)
        self.assertIn('need-kraken: "true"', workflow)
        self.assertIn("golden_set_release", workflow)
        self.assertIn("layout-smoke-inputs.py", workflow)
        self.assertIn("layout-smoke-run.py", workflow)
        self.assertIn("/normalization/binarization-integration/binarization-normalization-manifest.json", workflow)
        self.assertIn("persist-credentials: false", workflow)
        self.assertNotIn("hth_hardened_persist", workflow)

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
