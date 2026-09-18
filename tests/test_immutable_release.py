from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from hth.immutable_release import extract_collection


class ImmutableReleaseTests(unittest.TestCase):
    @staticmethod
    def _archive(path: Path, prefix: str = "photometric-integrated-collection") -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{prefix}/photometric-normalization-manifest.json", "{}")
            archive.writestr(f"{prefix}/photometric-normalized/fs_0001.png", b"pixels")

    def test_extracts_and_resolves_nested_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "release.zip"
            self._archive(asset)
            collection, marker = extract_collection(
                asset,
                root / "extracted",
                "photometric-normalization-manifest.json",
                "photometric-normalized",
            )
            self.assertEqual(collection.name, "photometric-integrated-collection")
            self.assertEqual(marker.parent, collection)

    def test_rejects_ambiguous_collection_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "release.zip"
            with zipfile.ZipFile(asset, "w") as archive:
                for prefix in ("first", "second"):
                    archive.writestr(f"{prefix}/manifest.json", "{}")
                    archive.writestr(f"{prefix}/images/page.png", b"pixels")
            with self.assertRaisesRegex(ValueError, "exactly one manifest.json"):
                extract_collection(asset, root / "extracted", "manifest.json", "images")

    def test_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = root / "release.zip"
            with zipfile.ZipFile(asset, "w") as archive:
                archive.writestr("../escape/manifest.json", "{}")
            with self.assertRaisesRegex(ValueError, "Unsafe immutable release member"):
                extract_collection(asset, root / "extracted", "manifest.json")

    def test_composite_action_owns_restore_and_extraction_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        action = (root / ".github/actions/restore-immutable-release/action.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("extract-destination:", action)
        self.assertIn("collection-marker:", action)
        self.assertIn("required-subdirectory:", action)
        self.assertIn("python -m hth.immutable_release extract-collection", action)


if __name__ == "__main__":
    unittest.main()
