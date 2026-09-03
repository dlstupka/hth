import importlib.util
import tempfile
import unittest
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "publish-golden-set-release.py"
SPEC = importlib.util.spec_from_file_location("publish_golden_set_release", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GoldenSetReleaseTests(unittest.TestCase):
    def test_regression_workflow_can_materialize_release_bundle(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "regress-detector.yml").read_text(encoding="utf-8")
        self.assertIn("golden_release_repository:", workflow)
        self.assertIn("golden_release_tag:", workflow)
        self.assertIn("golden_release_freeze:", workflow)
        self.assertIn("python -m hth.golden_set_release", workflow)
        self.assertIn('IMAGE_ROOT=$RUNNER_TEMP/golden-set-images/raw', workflow)

    def test_hth_0001_release_assets_preserve_frozen_identity(self) -> None:
        tag, golden_name, freeze_name = MODULE.release_assets(
            repository="dlstupka/hth-baptisms-san-antonio-1788-1824--1858-1898",
            golden_set=ROOT / "config" / "golden_set.json",
            freeze=ROOT / "config" / "golden_sets" / "HTH-0001.freeze.json",
        )
        self.assertEqual(tag, "HTH-GOLDEN-0001")
        self.assertEqual(golden_name, "HTH-0001.golden-set.json")
        self.assertEqual(freeze_name, "HTH-0001.freeze.json")

    def test_wrong_repository_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "canonical_release.repository"):
            MODULE.release_assets(
                repository="other/source",
                golden_set=ROOT / "config" / "golden_set.json",
                freeze=ROOT / "config" / "golden_sets" / "HTH-0001.freeze.json",
            )

    def test_bundle_is_deterministic_and_verifies_exact_images(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            golden = root / "golden.json"
            golden.write_text(json.dumps({"pages": [{"global_ordinal": 3}, {"global_ordinal": 64}]}), encoding="utf-8")
            images = root / "images"; images.mkdir()
            (images / "fs_0003.png").write_bytes(b"page-three")
            (images / "fs_0064.jpg").write_bytes(b"page-sixty-four")
            first, second = root / "first.zip", root / "second.zip"
            one = MODULE.build_image_bundle(golden, images, first)
            two = MODULE.build_image_bundle(golden, images, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(one["sha256"], two["sha256"])
            extracted = root / "extracted"
            MODULE.verify_and_extract(first, one, extracted)
            self.assertEqual((extracted / "raw" / "fs_0003.png").read_bytes(), b"page-three")
            self.assertEqual((extracted / "raw" / "fs_0064.jpg").read_bytes(), b"page-sixty-four")


if __name__ == "__main__":
    unittest.main()
