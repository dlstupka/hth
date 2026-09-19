import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from hth.regression.parameter_space import parameter_set_id
from hth.regression.parameter_provenance import parameter_identity_sha256
from hth.golden_set_catalog import resolve_golden_set_for_source
from hth.resolve_document_detector import resolve_rank_one, render_summary


class PreferredDocumentDetectorTests(unittest.TestCase):
    def _write_source_freeze(
        self,
        root: Path,
        *,
        golden_set_id: str,
        source_tag: str,
        manifest_sha: str,
        source_repository: str = "dlstupka/source",
        filename: str | None = None,
    ) -> Path:
        golden_name = f"{golden_set_id}.golden-set.json"
        golden = root / golden_name
        data = json.dumps({"collection_id": golden_set_id, "pages": []}) + "\n"
        golden.write_bytes(data.encode("utf-8"))
        freeze = root / (filename or f"{golden_set_id}.freeze.json")
        release_tag = (
            golden_set_id
            if golden_set_id.startswith("HTH-GOLDEN-")
            else f"HTH-GOLDEN-{golden_set_id.removeprefix('HTH-')}"
        )
        freeze.write_text(json.dumps({
            "state": "frozen",
            "golden_set_id": golden_set_id,
            "golden_set_path": golden_name,
            "golden_set_sha256": hashlib.sha256(data.encode("utf-8")).hexdigest(),
            "canonical_release": {
                "repository": source_repository,
                "tag": release_tag,
                "golden_set_asset": golden_name,
                "freeze_asset": f"{golden_set_id}.freeze.json",
            },
            "membership": {"page_count": 0, "global_ordinals": []},
            "source_release": {
                "repository": source_repository,
                "tag": source_tag,
                "manifest_sha256": manifest_sha,
            },
        }), encoding="utf-8")
        return freeze

    def _entry(self, detector, avg, minimum, stddev, evidence, record_path, parameter_id):
        return {
            "calibration_id": f"cal-{detector}",
            "calibration_status": "authoritative",
            "record_path": record_path,
            "parameter_provenance_path": f"{record_path}/parameter-provenance.json",
            "golden_set_id": "hth-0001",
            "golden_set_sha256": "gold123",
            "detector_id": detector,
            "created_at_utc": "2026-08-22T00:00:00Z",
            "build": {"github_run_number": "732", "run_url": "https://example.invalid/732"},
            "search": {"strategy": "exhaustive", "parameter_sets": 29, "exhaustive_complete": True},
            "selection": {
                "recommended_parameter_set_id": parameter_id,
                "best_avg_iou": avg,
                "minimum_iou": minimum,
                "stddev_iou": stddev,
                "failure_count": 0,
                "calibration_evidence": evidence,
            },
        }

    def test_resolves_highest_scoring_approved_calibration_and_exact_parameters(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            params = {
                "amsre_rescue_score_ceiling": 0.95,
                "doc_ufcn_minimum_confidence": 0.9,
                "maximum_amsre_refined_support_fraction": 0.65,
                "minimum_corner_disagreement_fraction": 0.0075,
            }
            detector = "amsre_doc_ufcn_fusion"
            legacy = parameter_set_id(params)
            full = parameter_identity_sha256(detector, params, schema_version="1")
            rec = Path("source-documents/source/golden-sets/hth-0001/gold123/calibrations") / detector / "cal"
            rec_dir = root / rec
            rec_dir.mkdir(parents=True)
            provenance = {
                "identity": {"detector": detector, "parameter_schema_version": "1"},
                "explicit_parameter_sets": {
                    full: {"sha256": full, "legacy_parameter_set_id": legacy, "parameters": params}
                },
            }
            (rec_dir / "parameter-provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
            index = {
                "entries": [
                    self._entry("other", 0.9800, 0.9700, 0.01, "High", "missing", "missing"),
                    self._entry(detector, 0.9897, 0.9814, 0.0063, "High", rec.as_posix(), legacy),
                    self._entry("unapproved", 0.9999, 0.9990, 0.001, "Medium", "missing", "missing"),
                ]
            }
            index_path = root / "calibration-index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")

            resolved = resolve_rank_one(index_path, golden_set_id="HTH-0001")
            self.assertEqual(resolved["detector"], detector)
            self.assertEqual(resolved["parameters"], params)
            self.assertEqual(resolved["parameter_set_id"], legacy)
            self.assertEqual(resolved["approval_level"], "Approved")
            compact_resolved = resolve_rank_one(
                index_path,
                golden_set_id="HTH-0001",
                include_persisted_backfill=False,
            )
            self.assertEqual(compact_resolved["parameter_identity_sha256"], full)
            resolved["golden_set_repository"] = "dlstupka/source"
            resolved["golden_set_release_tag"] = "HTH-GOLDEN-0001"
            summary = render_summary(
                resolved,
                display_name="Fusion Gen3",
                pipeline_repository="dlstupka/hth",
                pipeline_commit="abc123",
                results_repository="dlstupka/results",
                results_ref="results456",
            )
            self.assertIn("0.9897", summary)
            self.assertIn("0.9814", summary)
            self.assertIn("maximum_amsre_refined_support_fraction", summary)
            self.assertIn(
                "[`HTH-0001`](https://github.com/dlstupka/source/releases/tag/HTH-GOLDEN-0001)",
                summary,
            )
            self.assertIn("[`#732`](https://example.invalid/732)", summary)
            self.assertIn(
                "[Fusion Gen3](https://github.com/dlstupka/hth/blob/abc123/docs/detector-amsre-doc-ufcn-fusion.md)",
                summary,
            )
            self.assertIn(
                "**Rank:** [#1](https://github.com/dlstupka/results/blob/results456/indexes/calibration-index.json)",
                summary,
            )
            self.assertIn(
                "[`cal-amsre_doc_ufcn_fusion`](https://github.com/dlstupka/results/tree/results456/source-documents/source/golden-sets/hth-0001/gold123/calibrations/amsre_doc_ufcn_fusion/cal)",
                summary,
            )
            self.assertIn(
                "[`parameter-provenance.json`](https://github.com/dlstupka/results/blob/results456/source-documents/source/golden-sets/hth-0001/gold123/calibrations/amsre_doc_ufcn_fusion/cal/parameter-provenance.json)",
                summary,
            )

    def test_approved_accepts_persisted_authoritative_index_semantics(self):
        from hth.resolve_document_detector import _approved

        # Best Known treats authoritative exhaustive-family records as complete;
        # historic index rows need not redundantly carry exhaustive_complete.
        entry = self._entry("detector", 0.9897, 0.9814, 0.0063, {"rating": "High"}, "record", "abc")
        entry["search"].pop("exhaustive_complete")
        entry["search"]["strategy"] = "exhaustive-with-zombies"
        self.assertTrue(_approved(entry))

    def test_resolves_golden_set_from_exact_immutable_source_release(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_source_freeze(
                root,
                golden_set_id="HTH-0001",
                source_tag="HTH-SOURCE-0001",
                manifest_sha="a" * 64,
            )
            self._write_source_freeze(
                root,
                golden_set_id="HTH-GOLDEN-0002",
                source_tag="HTH-SOURCE-0002",
                manifest_sha="b" * 64,
                source_repository="https://github.com/dlstupka/source.git",
            )

            resolved = resolve_golden_set_for_source(
                root,
                repository_root=root,
                source_repository="dlstupka/source",
                source_release_tag="HTH-SOURCE-0002",
                source_release_manifest_sha256="b" * 64,
            )

            self.assertEqual(resolved, "HTH-GOLDEN-0002")

    def test_source_release_resolution_fails_closed_on_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_source_freeze(
                root,
                golden_set_id="HTH-GOLDEN-0002",
                source_tag="HTH-SOURCE-0002",
                manifest_sha="b" * 64,
            )

            with self.assertRaisesRegex(SystemExit, "No frozen Golden Set matches"):
                resolve_golden_set_for_source(
                    root,
                    repository_root=root,
                    source_repository="dlstupka/source",
                    source_release_tag="HTH-SOURCE-0002",
                    source_release_manifest_sha256="c" * 64,
                )

    def test_source_release_resolution_rejects_malformed_manifest_sha(self):
        with self.assertRaisesRegex(SystemExit, "64 hexadecimal"):
            resolve_golden_set_for_source(
                Path("unused"),
                repository_root=Path("."),
                source_repository="dlstupka/source",
                source_release_tag="HTH-SOURCE-0002",
                source_release_manifest_sha256="not-a-sha",
            )

    def test_source_release_resolution_rejects_ambiguous_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_sha = "d" * 64
            self._write_source_freeze(
                root,
                golden_set_id="HTH-GOLDEN-0002",
                source_tag="HTH-SOURCE-0002",
                manifest_sha=manifest_sha,
            )
            self._write_source_freeze(
                root,
                golden_set_id="HTH-GOLDEN-0003",
                source_tag="HTH-SOURCE-0002",
                manifest_sha=manifest_sha,
            )
            with self.assertRaisesRegex(SystemExit, "multiple frozen Golden Sets"):
                resolve_golden_set_for_source(
                    root,
                    repository_root=root,
                    source_repository="dlstupka/source",
                    source_release_tag="HTH-SOURCE-0002",
                    source_release_manifest_sha256=manifest_sha,
                )

    def test_source_release_resolution_validates_matching_freeze_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest_sha = "e" * 64
            self._write_source_freeze(
                root,
                golden_set_id="HTH-GOLDEN-0002",
                source_tag="HTH-SOURCE-0002",
                manifest_sha=manifest_sha,
            )
            (root / "HTH-GOLDEN-0002.golden-set.json").write_text(
                '{"collection_id":"HTH-GOLDEN-0002","pages":[{"global_ordinal":3}]}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "mapping is invalid"):
                resolve_golden_set_for_source(
                    root,
                    repository_root=root,
                    source_repository="dlstupka/source",
                    source_release_tag="HTH-SOURCE-0002",
                    source_release_manifest_sha256=manifest_sha,
                )

    def test_repository_freeze_maps_source_0002_to_golden_0002(self):
        root = Path(__file__).resolve().parents[1]
        freeze_root = root / "config/golden_sets"
        resolved = resolve_golden_set_for_source(
            freeze_root,
            repository_root=root,
            source_repository="dlstupka/hth-baptisms-san-antonio-1788-1824--1858-1898",
            source_release_tag="HTH-SOURCE-0002",
            source_release_manifest_sha256=(
                "871caffe1b6b09a14d3db130be220db6537cbd3746faeecb9f2efd3bfbcd7123"
            ),
        )
        self.assertEqual(resolved, "HTH-GOLDEN-0002")

    def test_workflows_use_preferred_without_detector_research_dropdown(self):
        root = Path(__file__).resolve().parents[1]
        preprocess = (root / ".github/workflows/preprocess.yml").read_text(encoding="utf-8")
        test = (root / ".github/workflows/preprocess-test.yml").read_text(encoding="utf-8")
        report = (root / ".github/workflows/generate-report.yml").read_text(encoding="utf-8")
        core = (root / ".github/workflows/_core-hth.yml").read_text(encoding="utf-8")
        self.assertNotIn("Run approved detector over every page for corpus review", preprocess)
        self.assertIn("document_detector: preferred", preprocess)
        self.assertIn("document_detector: preferred", test)
        self.assertIn("Resolve Rank #1 approved document detector", core)
        self.assertIn('if [[ ! -f "$calibration_index" && -f results-repo/calibration-index.json ]]', core)
        self.assertIn('calibration_index="results-repo/calibration-index.json"', core)
        self.assertIn('--index "../$calibration_index"', core)
        self.assertNotIn("--golden-set-id HTH-0001", core)
        self.assertIn("--golden-set-freeze-root config/golden_sets", core)
        self.assertIn('--source-release-tag "$SOURCE_RELEASE_TAG"', core)
        self.assertIn("--source-release-manifest-sha256", core)
        self.assertIn('--results-repository "$RESULTS_REPOSITORY"', core)
        self.assertIn('--results-ref "$RESULTS_REF"', core)
        self.assertIn("HTH_GOLDEN_SET_REPOSITORY: ${{ steps.preferred_document_detector.outputs.golden_set_repository }}", core)
        self.assertIn("HTH_GOLDEN_SET_RELEASE: ${{ steps.preferred_document_detector.outputs.golden_set_release_tag }}", core)
        self.assertIn("releases/download/${SOURCE_RELEASE_TAG}/source-release-manifest.json", core)
        self.assertIn("GOLDEN_SET_ID: ${{ steps.preferred_document_detector.outputs.golden_set_id }}", core)
        self.assertIn("--selection \"$RUNNER_TEMP/preferred-document-detector.json\"", core)
        self.assertRegex(
            preprocess,
            r"(?s)source_release_tag:\s+description:.*?default: HTH-SOURCE-0002"
            r".*?type: choice\s+options:\s+- HTH-SOURCE-0001\s+- HTH-SOURCE-0002",
        )
        self.assertRegex(
            report,
            r"(?s)golden_release_tag:\s+description:.*?default: HTH-GOLDEN-0002"
            r".*?type: choice\s+options:\s+- HTH-GOLDEN-0001\s+- HTH-GOLDEN-0002",
        )


if __name__ == "__main__":
    unittest.main()
