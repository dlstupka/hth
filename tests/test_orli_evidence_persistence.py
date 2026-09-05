import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hth.regression import learned_evidence
from hth.geometry import detector_orli_page_mask as detector


class OrliEvidencePersistenceTests(unittest.TestCase):
    def _provenance(self, root: Path) -> Path:
        path = root / "model-provenance.json"
        path.write_text(json.dumps({
            "model_id": "orli-base-2026",
            "model_sha256": "a" * 64,
            "orli_version": "0.0.2",
            "inference_backend": "orli.pred.segment",
            "serving_contract": "PIL image -> ordered baseline segmentation",
        }), encoding="utf-8")
        return path

    def test_persisted_orli_evidence_is_reused_without_exporter(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        page = {"image": image}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            golden = root / "golden.json"
            golden.write_text('{"pages": []}\n', encoding="utf-8")
            provenance = self._provenance(root)
            results = root / "results"
            first_output = root / "first"
            second_output = root / "second"
            calls = []

            def fake_exporter(images, output, *, progress=None):
                calls.append(len(images))
                output.mkdir(parents=True, exist_ok=True)
                target = output / "manifest.json"
                target.write_text(json.dumps({
                    "schema_version": "0.1",
                    "detector": "orli_page_mask",
                    "representation": "immutable-json",
                    "page_count": 1,
                    "records": [{"image_key": learned_evidence.detector_orli_page_mask._image_key(image), "evidence": {}}],
                }), encoding="utf-8")
                return target

            env = {"HTH_ORLI_PAGE_PROVENANCE": str(provenance)}
            with patch.dict(os.environ, env, clear=False), \
                 patch.object(learned_evidence, "load_pages", return_value=[page]), \
                 patch.dict(learned_evidence.EXPORTERS, {"orli_page_mask": fake_exporter}):
                learned_evidence.prepare(
                    detector="orli_page_mask",
                    golden_set=golden,
                    image_root=root,
                    maximum_dimension=2048,
                    output=first_output,
                    results_root=results,
                )
                learned_evidence.prepare(
                    detector="orli_page_mask",
                    golden_set=golden,
                    image_root=root,
                    maximum_dimension=2048,
                    output=second_output,
                    results_root=results,
                )

            self.assertEqual(calls, [1])
            self.assertTrue((second_output / "manifest.json").is_file())
            index = json.loads((results / learned_evidence.ORLI_EVIDENCE_INDEX).read_text(encoding="utf-8"))
            self.assertEqual(index["entry_count"], 1)
            entry = index["entries"][0]
            self.assertEqual(entry["model_sha256"], "a" * 64)
            self.assertEqual(entry["page_count"], 1)
            self.assertGreater(entry["size_bytes"], 0)
            self.assertTrue((results / entry["path"]).is_file())

    def test_export_supports_more_pages_than_the_process_cache_limit(self):
        images = [np.full((3, 3, 3), value, dtype=np.uint8) for value in range(detector._EVIDENCE_CACHE_LIMIT + 2)]
        frozen = detector._freeze_evidence({
            "regions": [], "lines": [], "baselines": [], "text_direction": "horizontal-lr",
        })
        with tempfile.TemporaryDirectory() as td, patch.object(detector, "_infer_evidence", return_value=frozen):
            manifest = detector.export_precomputed_golden_set_evidence(images, Path(td))
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["page_count"], detector._EVIDENCE_CACHE_LIMIT + 2)
        self.assertEqual(len(payload["records"]), detector._EVIDENCE_CACHE_LIMIT + 2)
        self.assertEqual(len({record["image_key"] for record in payload["records"]}), detector._EVIDENCE_CACHE_LIMIT + 2)


    def test_regression_driver_always_uses_parent_persistence_for_orli(self):
        text = Path("tools/run-detector-regressions.sh").read_text(encoding="utf-8")
        self.assertIn('[[ "$learned_detector" == "orli_page_mask" && "$learned_count" -gt 0 ]]', text)
        self.assertIn("--results-root results-repo", text)

    def test_results_workflows_publish_orli_evidence_and_index(self):
        for workflow in ("regress-detector.yml", "execution-optimizer.yml"):
            text = (Path(".github/workflows") / workflow).read_text(encoding="utf-8")
            self.assertIn("rebuild-orli-index --results-root results-repo", text)
            self.assertIn("hth_results_stage results-repo learned-evidence/orli_page_mask indexes/orli-evidence-index.json", text)

    def test_identity_changes_when_model_or_page_changes(self):
        base = {
            "schema_version": "1.0",
            "detector": "orli_page_mask",
            "model_sha256": "a" * 64,
            "image_keys": ["page-a"],
        }
        changed_model = dict(base, model_sha256="b" * 64)
        changed_page = dict(base, image_keys=["page-b"])
        self.assertNotEqual(learned_evidence._orli_evidence_id(base), learned_evidence._orli_evidence_id(changed_model))
        self.assertNotEqual(learned_evidence._orli_evidence_id(base), learned_evidence._orli_evidence_id(changed_page))

    def test_rebuild_index_is_stable_when_artifacts_do_not_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            artifact = root / learned_evidence.ORLI_EVIDENCE_ROOT / ("c" * 64) / "manifest.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text(json.dumps({
                "detector": "orli_page_mask",
                "persistence": {
                    "evidence_id": "c" * 64,
                    "created_at_utc": "2026-08-18T12:00:00Z",
                    "identity": {"model_sha256": "a" * 64, "image_keys": ["x"]},
                },
            }), encoding="utf-8")
            index = learned_evidence.rebuild_orli_index(results_root=root)
            first = index.read_bytes()
            learned_evidence.rebuild_orli_index(results_root=root)
            self.assertEqual(first, index.read_bytes())


if __name__ == "__main__":
    unittest.main()
