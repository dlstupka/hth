import importlib.util
import json
import tempfile
import unittest
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "acquire-familysearch-images.py"

spec = importlib.util.spec_from_file_location("familysearch_acquisition", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)


class FamilySearchAcquisitionTests(unittest.TestCase):
    def test_extract_seed_accepts_image_id_and_viewer_context(self):
        iid, context = mod.extract_seed(
            "https://www.familysearch.org/ark:/61903/3:1:3Q9M-TEST-123?"
            "view=explore&cc=12345&wc=ABC-DEF&groupId=GROUP"
        )
        self.assertEqual(iid, "3:1:3Q9M-TEST-123")
        self.assertEqual(
            context,
            {"cc": "12345", "wc": "ABC-DEF", "groupId": "GROUP"},
        )

    def test_media_candidates_prefer_explicit_original_relation(self):
        document = {
            "links": {
                "thumbnail": {"href": "https://example.test/thumb.jpg"},
                "image": {"href": "https://example.test/image.jpg"},
                "original": {"href": "https://example.test/original.tif?signature=secret"},
            }
        }
        candidates = mod.media_candidates(document)
        self.assertEqual(candidates[0], ("original", "https://example.test/original.tif?signature=secret"))
        self.assertIn(("image", "https://example.test/image.jpg"), candidates)

    def test_sanitize_url_never_persists_signed_query(self):
        self.assertEqual(
            mod.sanitize_url("https://cdn.example/image.tif?token=SECRET#fragment"),
            "https://cdn.example/image.tif",
        )

    def test_source_expectations_accept_common_image_metadata_names(self):
        metadata = {
            "imageWidth": 6000,
            "imageHeight": "4000",
            "fileSize": 123456,
            "checksum": "0123456789abcdef0123456789abcdef",
        }
        self.assertEqual(
            mod.source_expectations(metadata),
            {
                "width": 6000,
                "height": 4000,
                "size": 123456,
                "md5": "0123456789abcdef0123456789abcdef",
            },
        )

    def test_existing_image_is_verified_without_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image_path = root / "fs_0001.png"
            Image.new("L", (20, 10), 128).save(image_path)
            before = image_path.read_bytes()
            facts, checks = mod.verify_existing(
                image_path,
                {
                    "familysearch_image_id": "3:1:TEST",
                    "sha256": mod.sha256_file(image_path),
                },
                iid="3:1:TEST",
                expectations={"width": 20, "height": 10},
            )
            self.assertEqual((facts.width, facts.height), (20, 10))
            self.assertTrue(checks["manifest_sha256"])
            self.assertEqual(image_path.read_bytes(), before)

    def test_existing_corrupt_image_is_rejected_not_replaced(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fs_0001.jpg"
            path.write_bytes(b"not an image")
            with self.assertRaises(mod.AcquisitionError):
                mod.verify_existing(path, None, iid="3:1:TEST", expectations={})
            self.assertEqual(path.read_bytes(), b"not an image")

    def test_image_resource_url_uses_documented_seek_context_only(self):
        url = mod.image_resource_url(
            "3:1:ABC",
            {"cc": "COLL", "wc": "WAYPOINT", "groupId": "VIEWER-GROUP"},
            seek="next",
        )
        self.assertIn("seek=next", url)
        self.assertIn("cc=COLL", url)
        self.assertIn("wc=WAYPOINT", url)
        self.assertNotIn("groupId", url)


if __name__ == "__main__":
    unittest.main()
