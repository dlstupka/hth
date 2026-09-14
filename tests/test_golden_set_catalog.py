from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hth.golden_set_catalog import main, resolve_golden_set_release


ROOT = Path(__file__).resolve().parents[1]


class GoldenSetCatalogTests(unittest.TestCase):
    def test_resolves_legacy_and_current_release_tags_to_canonical_files(self) -> None:
        freeze_root = ROOT / "config/golden_sets"

        legacy = resolve_golden_set_release(
            freeze_root,
            repository_root=ROOT,
            release_tag="HTH-GOLDEN-0001",
        )
        current = resolve_golden_set_release(
            freeze_root,
            repository_root=ROOT,
            release_tag="HTH-GOLDEN-0002",
        )

        self.assertEqual(legacy["golden_set_id"], "HTH-0001")
        self.assertEqual(Path(legacy["golden_set_path"]), (ROOT / "config/golden_set.json").resolve())
        self.assertEqual(current["golden_set_id"], "HTH-GOLDEN-0002")
        self.assertEqual(
            Path(current["golden_set_path"]),
            (ROOT / "config/golden_sets/HTH-GOLDEN-0002.golden-set.json").resolve(),
        )

    def test_unknown_release_tag_fails_closed(self) -> None:
        with self.assertRaisesRegex(SystemExit, "No frozen Golden Set matches release tag"):
            resolve_golden_set_release(
                ROOT / "config/golden_sets",
                repository_root=ROOT,
                release_tag="HTH-GOLDEN-9999",
            )

    def test_cli_emits_resolved_identity_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "github-output"
            rc = main([
                "--freeze-root", str(ROOT / "config/golden_sets"),
                "--repository-root", str(ROOT),
                "--release-tag", "HTH-GOLDEN-0002",
                "--github-output", str(output),
            ])
            text = output.read_text(encoding="utf-8")
            self.assertEqual(rc, 0)
            self.assertIn("golden_set_id=HTH-GOLDEN-0002", text)
            self.assertIn("golden_set_path=", text)
            self.assertIn("freeze_path=", text)


if __name__ == "__main__":
    unittest.main()
