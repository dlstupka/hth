import json
import tempfile
import unittest
from pathlib import Path

from hth.validate_golden_set_freeze import validate_freeze


ROOT = Path(__file__).parents[1]


class GoldenSetFreezeTests(unittest.TestCase):
    def test_repository_hth_0001_matches_freeze_manifest(self) -> None:
        validate_freeze(
            freeze_path=ROOT / "config" / "golden_sets" / "HTH-0001.freeze.json",
            repository_root=ROOT,
        )

    def test_modified_frozen_set_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / "config"
            config.mkdir()
            golden = {"collection_id": "HTH-0001", "pages": [{"global_ordinal": 1}]}
            (config / "golden_set.json").write_text(json.dumps(golden) + "\n", encoding="utf-8")
            freeze = {
                "state": "frozen",
                "golden_set_id": "HTH-0001",
                "canonical_release": {
                    "repository": "owner/source",
                    "tag": "HTH-GOLDEN-0001",
                    "golden_set_asset": "HTH-0001.golden-set.json",
                    "freeze_asset": "HTH-0001.freeze.json",
                },
                "golden_set_path": "config/golden_set.json",
                "golden_set_sha256": "0" * 64,
                "membership": {"page_count": 1, "global_ordinals": [1]},
            }
            freeze_path = config / "freeze.json"
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "create a new Golden Set ID"):
                validate_freeze(freeze_path=freeze_path, repository_root=root)

    def test_missing_canonical_release_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            golden = root / "golden.json"
            golden.write_text('{"collection_id":"HTH-0002","pages":[]}\n', encoding="utf-8")
            freeze = root / "freeze.json"
            freeze.write_text(json.dumps({
                "state": "frozen", "golden_set_id": "HTH-0002",
                "golden_set_path": "golden.json", "golden_set_sha256": "unused",
                "membership": {"page_count": 0, "global_ordinals": []},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "canonical_release.repository"):
                validate_freeze(freeze_path=freeze, repository_root=root)


if __name__ == "__main__":
    unittest.main()
