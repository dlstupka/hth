from __future__ import annotations

import json
import argparse
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from hth.canonical_build_evidence import (
    SCOPE_ARTIFACT_PROFILES,
    SCOPE_EVIDENCE_PATHS,
    TONAL_INTEGRATION_SCOPE,
    artifact_profile,
    canonical_hash,
    evidence_relative_path,
    finalize,
)
from hth.normalize_document_images import _pixel_sha256
from hth.normalize_tonal import (
    assess,
    compare,
    integrate,
    package_release,
    validate,
)
from hth.tonal_summary import summary_lines


class TonalNormalizationTests(unittest.TestCase):
    @staticmethod
    def _config(name: str) -> dict:
        root = Path(__file__).resolve().parents[1]
        return json.loads((root / "config" / name).read_text(encoding="utf-8"))

    def _collection(self, root: Path, count: int = 10) -> tuple[Path, dict]:
        collection = root / "photometric"
        images = collection / "photometric-normalized"
        images.mkdir(parents=True)
        pages = []
        for ordinal in range(1, count + 1):
            gradient = np.linspace(85, 165, 320, dtype=np.uint8)
            image = cv2.cvtColor(np.tile(gradient, (420, 1)), cv2.COLOR_GRAY2BGR)
            for y in range(40, 400, 32):
                cv2.line(image, (20, y), (300, y), (65, 65, 65), 2)
            target = images / f"fs_{ordinal:04d}.png"
            self.assertTrue(cv2.imwrite(str(target), image))
            pages.append({"global_ordinal": ordinal, "output_pixel_sha256": _pixel_sha256(image)})
        manifest = {"photometric_result_identity": "a" * 64, "pages": pages}
        (collection / "photometric-normalization-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return collection, manifest

    def test_complete_tonal_evidence_and_integration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection, upstream = self._collection(root)
            assessment = assess(collection, upstream, self._config("tonal-assessment.json"))
            self.assertEqual(assessment["aggregate"]["page_count"], 10)
            self.assertEqual(assessment["aggregate"]["correction-candidate"], 10)
            comparison = compare(
                collection, upstream, assessment, self._config("tonal-method-assessment.json")
            )
            self.assertEqual(comparison["candidate_count"], 2)
            self.assertIsNotNone(comparison["recommended_method_id"])
            validation = validate(
                collection,
                upstream,
                assessment,
                comparison,
                self._config("tonal-method-validation.json"),
            )
            self.assertEqual(validation["aggregate"]["held_out_candidates"], 8)
            self.assertEqual(validation["decision"], "apply")
            output = root / "tonal"
            result = integrate(collection, upstream, assessment, comparison, validation, output)
            self.assertEqual(result["aggregate"]["page_count"], 10)
            self.assertEqual(result["aggregate"]["corrected_pages"], 10)
            self.assertTrue(all(page["pipeline_action"] == "corrected-and-continue" for page in result["pages"]))

            first = root / "first.zip"
            second = root / "second.zip"
            tag = f"HTH-TONAL-{result['tonal_result_identity']}"
            first_record = package_release(output, first, tag)
            (output / "release.json").write_text(json.dumps(first_record), encoding="utf-8")
            second_record = package_release(output, second, tag)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            for name, payload in (
                ("tonal-assessment.json", assessment),
                ("tonal-method-assessment.json", comparison),
                ("tonal-validation.json", validation),
                ("release.json", first_record),
            ):
                (output / name).write_text(json.dumps(payload), encoding="utf-8")
            effective_inputs = {
                "source": {"release_manifest_sha256": "b" * 64, "files": [{"sha256": "c" * 64}]},
                "contract": {}, "configuration": [], "implementation": [], "runtime_contract": [], "runtime": {},
            }
            plan = {
                "decision": "execute",
                "scope": TONAL_INTEGRATION_SCOPE,
                "policy": "auto",
                "effective_inputs": effective_inputs,
                "effective_build_identity": canonical_hash(effective_inputs),
                "comparison_required": False,
                "incumbent_result_identity": None,
                "execution": {},
            }
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            evidence = finalize(argparse.Namespace(
                plan=plan_path,
                output_root=output,
                evidence_store=root / "results/canonical-build-evidence.json",
                evidence_output=output / "canonical-build-evidence.json",
                github_output="",
                github_summary="",
            ))
            self.assertEqual(evidence["scope"], TONAL_INTEGRATION_SCOPE)
            self.assertEqual(len(evidence["canonical_result"]["pages"]), 10)

    def test_cbe_scope_has_complete_tonal_contract(self) -> None:
        specs = artifact_profile(TONAL_INTEGRATION_SCOPE)
        self.assertEqual(
            [spec.logical_name for spec in specs],
            ["tonal-normalization-manifest", "tonal-assessment", "tonal-method-assessment", "tonal-validation", "release-record"],
        )
        self.assertEqual(set(SCOPE_ARTIFACT_PROFILES), set(SCOPE_EVIDENCE_PATHS))
        self.assertEqual(
            evidence_relative_path(TONAL_INTEGRATION_SCOPE),
            "normalization/tonal-integration/canonical-build-evidence.json",
        )

    def test_summaries_expose_compact_scientific_decisions(self) -> None:
        assessment = {
            "aggregate": {
                "page_count": 929,
                "correction-candidate": 673,
                "preserve": 221,
                "review": 35,
            }
        }
        comparison = {
            "candidate_count": 1,
            "globally_safe_methods": [],
            "recommended_method_id": None,
            "config": {"methods": [{"id": "percentile-stretch-50"}]},
            "pages": [{
                "variants": [{
                    "method_id": "percentile-stretch-50",
                    "safe": False,
                    "tonal_span_gain": 0.25,
                    "high_frequency_correlation": 0.95,
                    "gates": {
                        "tonal_span_gain": True,
                        "detail_correlation": False,
                        "endpoint_clipping": True,
                        "median_shift": True,
                    },
                }],
            }],
        }
        validation = {
            "aggregate": {
                "held_out_candidates": 535,
                "safe_candidates": 0,
                "mean_tonal_span_gain": 0.0,
            },
            "method": None,
            "decision": "preserve",
        }
        integration = {
            "aggregate": {"page_count": 929, "corrected_pages": 0, "preserved_pages": 929},
            "method": None,
            "tonal_result_identity": "a" * 64,
        }

        self.assertIn("Correction candidates: `673`", "\n".join(summary_lines("assess", assessment)))
        method_summary = "\n".join(summary_lines("compare", comparison))
        self.assertIn("Mean detail correlation", method_summary)
        self.assertIn("detail correlation: 1", method_summary)
        self.assertIn("Decision: `preserve`", "\n".join(summary_lines("validate", validation)))
        self.assertIn("Corrected pages: `0`", "\n".join(summary_lines("integrate", integration)))

    def test_workflows_use_cached_release_and_cbe(self) -> None:
        root = Path(__file__).resolve().parents[1]
        core = (root / ".github/workflows/_core-tonal-evidence.yml").read_text(encoding="utf-8")
        integration = (root / ".github/workflows/integrate-tonal.yml").read_text(encoding="utf-8")
        orchestrator = (root / ".github/workflows/normalize.yml").read_text(encoding="utf-8")
        action = (root / ".github/actions/restore-immutable-release/action.yml").read_text(encoding="utf-8")
        for name in ("assess-tonal.yml", "assess-tonal-methods.yml", "validate-tonal-method.yml"):
            dispatcher = (root / ".github/workflows" / name).read_text(encoding="utf-8")
            self.assertIn("uses: ./.github/workflows/_core-tonal-evidence.yml", dispatcher)
        self.assertIn("uses: ./hth-pipeline/.github/actions/restore-immutable-release", core)
        self.assertIn("Evaluate tonal evidence Canonical Build Evidence", core)
        self.assertIn("steps.cbe_plan.outputs.decision == 'execute'", core)
        self.assertIn("--scope \"${{ steps.stage_contract.outputs.scope }}\"", core)
        self.assertIn("hth-tonal-assessment", core)
        self.assertIn("hth-tonal-method-assessment", core)
        self.assertIn("hth-tonal-validation", core)
        self.assertIn("photometric_result_identity", core)
        self.assertIn("Finalize tonal evidence Canonical Build Evidence", core)
        self.assertIn("uses: ./hth-pipeline/.github/actions/restore-immutable-release", integration)
        self.assertIn("/tmp/.ar/.hth-release-cache", action)
        self.assertIn("uses: actions/cache@v5", action)
        self.assertIn("sha256sum --check", action)
        self.assertIn("python -m hth.release_provenance_summary", action)
        self.assertIn('--input-asset "$HTH_RELEASE_ASSET"', action)
        self.assertIn('--cache-source "$HTH_RELEASE_CACHE_SOURCE"', action)
        self.assertIn("--scope hth-tonal-integration", integration)
        self.assertIn("decision != 'execute'", integration)
        self.assertIn("Existing tonal release does not match deterministic rebuild", integration)
        self.assertEqual(integration.count("python -m hth.release_provenance_summary"), 2)
        self.assertIn('--release-tag "${fields[0]}"', integration)
        self.assertIn('--release-tag "${{ steps.integration.outputs.release_tag }}"', integration)
        self.assertIn("needs: integrate-photometric", orchestrator)
        self.assertIn("uses: ./.github/workflows/assess-perspective.yml", orchestrator)
        self.assertIn("needs: assess-tonal", orchestrator)
        self.assertIn("needs: assess-tonal-methods", orchestrator)
        self.assertIn("needs: validate-tonal-method", orchestrator)
        self.assertIn("uses: ./.github/workflows/integrate-tonal.yml", orchestrator)
        self.assertIn("start_stage:", orchestrator)
        self.assertIn("- crop-framing", orchestrator)
        self.assertIn("- orientation-deskew", orchestrator)
        self.assertIn("- tonal-integration", orchestrator)
        self.assertIn("uses: ./.github/workflows/assess-crop-framing.yml", orchestrator)
        self.assertIn("uses: ./.github/workflows/assess-orientation-deskew.yml", orchestrator)
        self.assertIn("collection-marker: photometric-normalization-manifest.json", core)
        self.assertIn("steps.photometric_asset.outputs.collection-root", core)
        self.assertIn("collection-marker: photometric-normalization-manifest.json", integration)
        self.assertIn("steps.photometric_asset.outputs.collection-root", integration)
        self.assertEqual(core.count('--github-summary "$GITHUB_STEP_SUMMARY"'), 3)
        self.assertEqual(integration.count('--github-summary "$GITHUB_STEP_SUMMARY"'), 5)
        self.assertIn("Summarize tonal evidence", core)
        self.assertIn("python -m hth.tonal_summary", core)
        self.assertIn("--stage integrate", integration)
        self.assertIn('(\"photometric_result_identity\", \"source_identity\")', integration)
        self.assertIn('--source-commit "${{ steps.inputs.outputs.source_identity }}"', integration)
        self.assertNotIn('--source-commit "${{ steps.inputs.outputs.results_commit }}"', integration)

    def test_reusable_workflow_retention_inputs_are_explicitly_numeric(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cast = "${{ fromJSON(format('{0}', inputs.artifact_retention_days)) }}"
        callers = {
            "normalize.yml": 15,
            "assess-tonal.yml": 1,
            "assess-tonal-methods.yml": 1,
            "validate-tonal-method.yml": 1,
        }
        for name, expected_count in callers.items():
            with self.subTest(workflow=name):
                text = (root / ".github/workflows" / name).read_text(encoding="utf-8")
                self.assertEqual(text.count(cast), expected_count)


if __name__ == "__main__":
    unittest.main()
