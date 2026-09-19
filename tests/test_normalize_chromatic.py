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
    CHROMATIC_INTEGRATION_SCOPE,
    artifact_profile,
    canonical_hash,
    evidence_relative_path,
    finalize,
)
from hth.normalize_document_images import _pixel_sha256
from hth.normalize_chromatic import (
    apply_method,
    assess,
    compare,
    integrate,
    measure,
    package_release,
    validate,
)
from hth.chromatic_summary import summary_lines


class ChromaticNormalizationTests(unittest.TestCase):
    @staticmethod
    def _config(name: str) -> dict:
        root = Path(__file__).resolve().parents[1]
        return json.loads((root / "config" / name).read_text(encoding="utf-8"))

    def _collection(self, root: Path, count: int = 10) -> tuple[Path, dict]:
        collection = root / "tonal"
        images = collection / "tonal-normalized"
        images.mkdir(parents=True)
        pages = []
        for ordinal in range(1, count + 1):
            gradient = np.linspace(85, 165, 320, dtype=np.uint8)
            image = cv2.cvtColor(np.tile(gradient, (420, 1)), cv2.COLOR_GRAY2BGR)
            image = np.clip(
                image.astype(np.int16) + np.array([0, 14, 28], dtype=np.int16), 0, 255
            ).astype(np.uint8)
            for y in range(40, 400, 32):
                cv2.line(image, (20, y), (300, y), (65, 65, 65), 2)
            target = images / f"fs_{ordinal:04d}.png"
            self.assertTrue(cv2.imwrite(str(target), image))
            pages.append({"global_ordinal": ordinal, "output_pixel_sha256": _pixel_sha256(image)})
        manifest = {"tonal_result_identity": "a" * 64, "pages": pages}
        (collection / "tonal-normalization-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return collection, manifest

    def test_complete_chromatic_evidence_and_integration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection, upstream = self._collection(root)
            assessment = assess(collection, upstream, self._config("chromatic-assessment.json"))
            self.assertEqual(assessment["aggregate"]["page_count"], 10)
            self.assertEqual(assessment["aggregate"]["correction-candidate"], 10)
            comparison = compare(
                collection, upstream, assessment, self._config("chromatic-method-assessment.json")
            )
            self.assertEqual(comparison["candidate_count"], 2)
            self.assertIsNotNone(comparison["recommended_method_id"])
            validation = validate(
                collection,
                upstream,
                assessment,
                comparison,
                self._config("chromatic-method-validation.json"),
            )
            self.assertEqual(validation["aggregate"]["held_out_candidates"], 8)
            self.assertEqual(validation["decision"], "apply")
            output = root / "chromatic"
            result = integrate(collection, upstream, assessment, comparison, validation, output)
            self.assertEqual(result["aggregate"]["page_count"], 10)
            self.assertEqual(result["aggregate"]["corrected_pages"], 10)
            self.assertTrue(all(page["pipeline_action"] == "corrected-and-continue" for page in result["pages"]))

            first = root / "first.zip"
            second = root / "second.zip"
            tag = f"HTH-CHROMATIC-{result['chromatic_result_identity']}"
            first_record = package_release(output, first, tag)
            (output / "release.json").write_text(json.dumps(first_record), encoding="utf-8")
            second_record = package_release(output, second, tag)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            for name, payload in (
                ("chromatic-assessment.json", assessment),
                ("chromatic-method-assessment.json", comparison),
                ("chromatic-validation.json", validation),
                ("release.json", first_record),
            ):
                (output / name).write_text(json.dumps(payload), encoding="utf-8")
            effective_inputs = {
                "source": {"release_manifest_sha256": "b" * 64, "files": [{"sha256": "c" * 64}]},
                "contract": {}, "configuration": [], "implementation": [], "runtime_contract": [], "runtime": {},
            }
            plan = {
                "decision": "execute",
                "scope": CHROMATIC_INTEGRATION_SCOPE,
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
            self.assertEqual(evidence["scope"], CHROMATIC_INTEGRATION_SCOPE)
            self.assertEqual(len(evidence["canonical_result"]["pages"]), 10)

    def test_grayscale_and_color_rich_pages_are_protected(self) -> None:
        config = self._config("chromatic-assessment.json")
        method = self._config("chromatic-method-assessment.json")["methods"][0]
        grayscale = np.tile(np.linspace(20, 230, 128, dtype=np.uint8), (96, 1))
        self.assertEqual(measure(grayscale, config)["decision"], "preserve")
        self.assertTrue(np.array_equal(apply_method(grayscale, method), grayscale))

        color_rich = np.zeros((96, 128, 3), dtype=np.uint8)
        color_rich[:, :64] = (20, 20, 230)
        color_rich[:, 64:] = (230, 40, 20)
        result = measure(color_rich, config)
        self.assertEqual(result["decision"], "review")
        self.assertIn("potential-significant-color-content", result["decision_reasons"])

    def test_preserve_decision_keeps_every_page_pixel_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection, upstream = self._collection(root, count=3)
            assessment = assess(collection, upstream, self._config("chromatic-assessment.json"))
            comparison = compare(
                collection, upstream, assessment, self._config("chromatic-method-assessment.json")
            )
            validation = validate(
                collection,
                upstream,
                assessment,
                comparison,
                self._config("chromatic-method-validation.json"),
            )
            validation["decision"] = "preserve"
            validation["method"] = None
            validation.pop("validation_identity", None)
            validation["validation_identity"] = canonical_hash(validation)
            result = integrate(
                collection, upstream, assessment, comparison, validation, root / "preserved"
            )
            self.assertEqual(result["aggregate"]["corrected_pages"], 0)
            self.assertEqual(result["aggregate"]["preserved_pages"], 3)
            self.assertTrue(
                all(page["input_pixel_sha256"] == page["output_pixel_sha256"] for page in result["pages"])
            )

    def test_cbe_scope_has_complete_chromatic_contract(self) -> None:
        specs = artifact_profile(CHROMATIC_INTEGRATION_SCOPE)
        self.assertEqual(
            [spec.logical_name for spec in specs],
            ["chromatic-normalization-manifest", "chromatic-assessment", "chromatic-method-assessment", "chromatic-validation", "release-record"],
        )
        self.assertEqual(set(SCOPE_ARTIFACT_PROFILES), set(SCOPE_EVIDENCE_PATHS))
        self.assertEqual(
            evidence_relative_path(CHROMATIC_INTEGRATION_SCOPE),
            "normalization/chromatic-integration/canonical-build-evidence.json",
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
            "config": {"methods": [{"id": "background-neutralization-25"}]},
            "pages": [{
                "variants": [{
                    "method_id": "background-neutralization-25",
                    "safe": False,
                    "background_cast_reduction_fraction": 0.25,
                    "luminance_detail_correlation": 0.95,
                    "gates": {
                        "background_cast_reduction": True,
                        "luminance_detail_correlation": False,
                        "chroma_structure_correlation": True,
                        "gamut_clipping": True,
                        "median_luminance_shift": True,
                    },
                }],
            }],
        }
        validation = {
            "aggregate": {
                "held_out_candidates": 535,
                "safe_candidates": 0,
                "mean_background_cast_reduction_fraction": 0.0,
            },
            "method": None,
            "decision": "preserve",
        }
        integration = {
            "aggregate": {"page_count": 929, "corrected_pages": 0, "preserved_pages": 929},
            "method": None,
            "chromatic_result_identity": "a" * 64,
        }

        self.assertIn("Correction candidates: `673`", "\n".join(summary_lines("assess", assessment)))
        method_summary = "\n".join(summary_lines("compare", comparison))
        self.assertIn("Mean luminance-detail correlation", method_summary)
        self.assertIn("luminance detail correlation: 1", method_summary)
        self.assertIn("Decision: `preserve`", "\n".join(summary_lines("validate", validation)))
        self.assertIn("Corrected pages: `0`", "\n".join(summary_lines("integrate", integration)))

    def test_workflows_use_cached_release_and_cbe(self) -> None:
        root = Path(__file__).resolve().parents[1]
        core = (root / ".github/workflows/_core-chromatic-evidence.yml").read_text(encoding="utf-8")
        integration = (root / ".github/workflows/integrate-chromatic.yml").read_text(encoding="utf-8")
        orchestrator = (root / ".github/workflows/normalize.yml").read_text(encoding="utf-8")
        action = (root / ".github/actions/restore-immutable-release/action.yml").read_text(encoding="utf-8")
        for name in ("assess-chromatic.yml", "assess-chromatic-methods.yml", "validate-chromatic-method.yml"):
            dispatcher = (root / ".github/workflows" / name).read_text(encoding="utf-8")
            self.assertIn("uses: ./.github/workflows/_core-chromatic-evidence.yml", dispatcher)
        self.assertIn("uses: ./hth-pipeline/.github/actions/restore-immutable-release", core)
        self.assertIn("Evaluate chromatic evidence Canonical Build Evidence", core)
        self.assertIn("steps.cbe_plan.outputs.decision == 'execute'", core)
        self.assertIn("--scope \"${{ steps.stage_contract.outputs.scope }}\"", core)
        self.assertIn("hth-chromatic-assessment", core)
        self.assertIn("hth-chromatic-method-assessment", core)
        self.assertIn("hth-chromatic-validation", core)
        self.assertIn("tonal_result_identity", core)
        self.assertIn("Finalize chromatic evidence Canonical Build Evidence", core)
        self.assertIn("uses: ./hth-pipeline/.github/actions/restore-immutable-release", integration)
        self.assertIn("/tmp/.ar/.hth-release-cache", action)
        self.assertIn("uses: actions/cache@v5", action)
        self.assertIn("sha256sum --check", action)
        self.assertIn("python -m hth.release_provenance_summary", action)
        self.assertIn('--input-asset "$HTH_RELEASE_ASSET"', action)
        self.assertIn('--cache-source "$HTH_RELEASE_CACHE_SOURCE"', action)
        self.assertIn("--scope hth-chromatic-integration", integration)
        self.assertIn("decision != 'execute'", integration)
        self.assertIn("Existing chromatic release does not match deterministic rebuild", integration)
        self.assertEqual(integration.count("python -m hth.release_provenance_summary"), 2)
        self.assertIn('--release-tag "${fields[0]}"', integration)
        self.assertIn('--release-tag "${{ steps.integration.outputs.release_tag }}"', integration)
        self.assertIn("needs: integrate-tonal", orchestrator)
        self.assertIn("uses: ./.github/workflows/assess-perspective.yml", orchestrator)
        self.assertIn("needs: assess-chromatic", orchestrator)
        self.assertIn("needs: assess-chromatic-methods", orchestrator)
        self.assertIn("needs: validate-chromatic-method", orchestrator)
        self.assertIn("uses: ./.github/workflows/integrate-chromatic.yml", orchestrator)
        self.assertIn("start_stage:", orchestrator)
        self.assertIn("- crop-framing", orchestrator)
        self.assertIn("- orientation-deskew", orchestrator)
        self.assertIn("- chromatic-integration", orchestrator)
        self.assertIn("uses: ./.github/workflows/assess-crop-framing.yml", orchestrator)
        self.assertIn("uses: ./.github/workflows/assess-orientation-deskew.yml", orchestrator)
        self.assertIn("collection-marker: tonal-normalization-manifest.json", core)
        self.assertIn("steps.tonal_asset.outputs.collection-root", core)
        self.assertIn("collection-marker: tonal-normalization-manifest.json", integration)
        self.assertIn("steps.tonal_asset.outputs.collection-root", integration)
        self.assertEqual(core.count('--github-summary "$GITHUB_STEP_SUMMARY"'), 3)
        self.assertEqual(integration.count('--github-summary "$GITHUB_STEP_SUMMARY"'), 5)
        self.assertIn("Summarize chromatic evidence", core)
        self.assertIn("python -m hth.chromatic_summary", core)
        self.assertIn("--stage integrate", integration)
        self.assertIn('(\"tonal_result_identity\", \"source_identity\")', integration)
        self.assertIn('--source-commit "${{ steps.inputs.outputs.source_identity }}"', integration)
        self.assertNotIn('--source-commit "${{ steps.inputs.outputs.results_commit }}"', integration)

    def test_reusable_workflow_retention_inputs_are_explicitly_numeric(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cast = "${{ fromJSON(format('{0}', inputs.artifact_retention_days)) }}"
        callers = {
            "normalize.yml": 27,
            "assess-chromatic.yml": 1,
            "assess-chromatic-methods.yml": 1,
            "validate-chromatic-method.yml": 1,
        }
        for name, expected_count in callers.items():
            with self.subTest(workflow=name):
                text = (root / ".github/workflows" / name).read_text(encoding="utf-8")
                self.assertEqual(text.count(cast), expected_count)


if __name__ == "__main__":
    unittest.main()
