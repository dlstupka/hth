from __future__ import annotations

import re
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

from hth.canonical_build_evidence import (
    SCOPE_ARTIFACT_PROFILES,
    SCOPE_EVIDENCE_PATHS,
    canonical_hash,
    finalize,
)


ROOT = Path(__file__).resolve().parents[1]


class NormalizationCanonicalEvidenceCoverageTests(unittest.TestCase):
    def test_every_normalization_stage_family_has_a_registered_scope(self) -> None:
        expected = {
            "hth-crop-framing-assessment",
            "hth-orientation-deskew-assessment",
            "hth-normalization",
            "hth-perspective-assessment",
            "hth-photometric-assessment",
            "hth-photometric-method-assessment",
            "hth-photometric-validation",
            "hth-photometric-integration",
            "hth-tonal-assessment",
            "hth-tonal-method-assessment",
            "hth-tonal-validation",
            "hth-tonal-integration",
            "hth-chromatic-assessment",
            "hth-chromatic-method-assessment",
            "hth-chromatic-validation",
            "hth-chromatic-integration",
            "hth-denoising-assessment",
            "hth-denoising-method-assessment",
            "hth-denoising-validation",
            "hth-denoising-integration",
            "hth-sharpening-assessment",
            "hth-sharpening-method-assessment",
            "hth-sharpening-validation",
            "hth-sharpening-integration",
            "hth-binarization-assessment",
            "hth-binarization-method-assessment",
            "hth-binarization-validation",
            "hth-binarization-integration",
        }
        self.assertTrue(expected <= set(SCOPE_ARTIFACT_PROFILES))
        self.assertTrue(expected <= set(SCOPE_EVIDENCE_PATHS))

    def test_new_diagnostic_workflows_gate_expensive_work_on_cbe(self) -> None:
        workflows = {
            "assess-crop-framing.yml": ("hth-crop-framing-assessment", "Materialize immutable GS0002 images"),
            "assess-orientation-deskew.yml": ("hth-orientation-deskew-assessment", "Download immutable source release"),
            "assess-perspective.yml": ("hth-perspective-assessment", "Download immutable source release"),
            "assess-photometric.yml": ("hth-photometric-assessment", "Download immutable source release"),
            "assess-photometric-methods.yml": ("hth-photometric-method-assessment", "Download immutable source release"),
            "validate-photometric-method.yml": ("hth-photometric-validation", "Download immutable source release"),
        }
        for filename, (scope, expensive_step) in workflows.items():
            with self.subTest(workflow=filename):
                text = (ROOT / ".github/workflows" / filename).read_text(encoding="utf-8")
                self.assertIn(f"--scope {scope}", text)
                self.assertIn("python -m hth.canonical_build_evidence finalize", text)
                self.assertIn("python -m hth.canonical_build_evidence merge", text)
                download = text.split(f"- name: {expensive_step}", 1)[1].split("- name:", 1)[0]
                self.assertIn("if: steps.cbe.outputs.decision == 'execute'", download)

    def test_orientation_output_is_run_scoped_for_persistent_runners(self) -> None:
        text = (ROOT / ".github/workflows/assess-orientation-deskew.yml").read_text(encoding="utf-8")
        self.assertIn(
            "ASSESSMENT_OUTPUT: ${{ runner.temp }}/orientation-deskew-assessment-${{ github.run_id }}-${{ github.run_attempt }}",
            text,
        )
        self.assertIn('--output "$ASSESSMENT_OUTPUT"', text)
        self.assertIn("path: ${{ env.ASSESSMENT_OUTPUT }}", text)
        self.assertNotIn("--output orientation-deskew-assessment", text)

    def test_new_scope_artifact_contracts_finalize(self) -> None:
        scopes = (
            "hth-crop-framing-assessment",
            "hth-orientation-deskew-assessment",
            "hth-perspective-assessment",
            "hth-photometric-assessment",
            "hth-photometric-method-assessment",
            "hth-photometric-validation",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for scope in scopes:
                with self.subTest(scope=scope):
                    output = root / scope
                    output.mkdir()
                    for spec in SCOPE_ARTIFACT_PROFILES[scope]:
                        path = output / spec.generated_path
                        path.parent.mkdir(parents=True, exist_ok=True)
                        if spec.format == "binary":
                            path.write_text("canonical diagnostic evidence\n", encoding="utf-8")
                        else:
                            payload = {"schema_version": "test"}
                            if spec.generated_path in {"assessment.json", "validation.json"}:
                                payload["pages"] = [{"global_ordinal": 1, "decision": "preserve"}]
                            if scope == "hth-crop-framing-assessment" and spec.generated_path == "assessment.json":
                                payload["algorithms"] = {"axis-aligned": {}}
                                payload["pages"][0]["algorithm"] = "axis-aligned"
                            path.write_text(json.dumps(payload), encoding="utf-8")
                    effective_inputs = {
                        "source": {
                            "repository": "example/results",
                            "release": "immutable-input",
                            "release_manifest_sha256": "a" * 64,
                            "commit": "b" * 64,
                            "files": [{"path": "input.json", "sha256": "c" * 64}],
                        }
                    }
                    plan = {
                        "scope": scope,
                        "policy": "auto",
                        "decision": "execute",
                        "effective_build_identity": canonical_hash(effective_inputs),
                        "effective_inputs": effective_inputs,
                        "comparison_required": False,
                        "resource_utilization": {
                            "schema_version": "1",
                            "canonical_evidence_cache": {
                                "scope": scope,
                                "path": SCOPE_EVIDENCE_PATHS[scope],
                                "lookup_identity": canonical_hash(effective_inputs),
                                "lookup": "miss",
                                "action": "populated",
                                "canonical_result_identity": None,
                            },
                            "immutable_releases": [{
                                "role": "source",
                                "repository": "example/results",
                                "release": "immutable-input",
                                "release_manifest_sha256": "a" * 64,
                                "commit": "b" * 64,
                                "utilization": "consumed",
                            }],
                        },
                        "execution": {"evaluated_at_utc": "2026-01-01T00:00:00Z"},
                    }
                    plan_path = root / f"{scope}-plan.json"
                    plan_path.write_text(json.dumps(plan), encoding="utf-8")
                    evidence_output = output / "canonical-build-evidence.json"
                    result = finalize(SimpleNamespace(
                        plan=plan_path,
                        output_root=output,
                        evidence_store=root / f"{scope}-missing-store.json",
                        evidence_output=evidence_output,
                        github_output="",
                        github_summary="",
                    ))
                    self.assertEqual(result["scope"], scope)
                    self.assertTrue(evidence_output.is_file())

    def test_orchestrator_is_fail_closed_but_allows_explicit_stage_entry(self) -> None:
        text = (ROOT / ".github/workflows/normalize.yml").read_text(encoding="utf-8")
        unsafe = re.findall(
            r"needs\.[a-z0-9-]+\.result == 'success' \|\| needs\.[a-z0-9-]+\.result == 'skipped'",
            text,
        )
        self.assertEqual(unsafe, [])
        self.assertGreaterEqual(text.count("inputs.start_stage =="), 27)

    def test_normalization_publication_preserves_all_stage_evidence(self) -> None:
        text = (ROOT / ".github/workflows/normalize.yml").read_text(encoding="utf-8")
        self.assertIn('cp -a results-repo/normalization/. "$staging/normalization/"', text)
        self.assertIn('rm -f "$staging/normalization/applied-normalization-policy.json"', text)

    def test_ephemeral_full_artifact_is_opt_in(self) -> None:
        text = (ROOT / ".github/workflows/normalize.yml").read_text(encoding="utf-8")
        block = text.split("upload_full_artifact:", 1)[1].split("normalization_recipe:", 1)[0]
        self.assertIn("default: false", block)

    def test_successful_orchestration_publishes_resource_lifecycle(self) -> None:
        text = (ROOT / ".github/workflows" / "normalize.yml").read_text(encoding="utf-8")
        block = text.split("  audit-resource-lifecycle:", 1)[1]
        self.assertIn("needs: integrate-binarization", block)
        self.assertIn("needs.integrate-binarization.result == 'success'", block)
        self.assertIn("python -m hth.resource_lifecycle", block)
        self.assertIn("metadata/resource-lifecycle.json", block)
        self.assertIn("isLatest", block)
        self.assertIn("$MIRROR_REPOSITORY", block)
        self.assertIn("$EVIDENCE_CACHE_REPOSITORY", block)
        self.assertIn("--repository-root hth-pipeline", block)
        self.assertIn("Cleanup action: `none`", (ROOT / "hth/resource_lifecycle.py").read_text(encoding="utf-8"))

    def test_orchestrator_propagates_one_cbe_policy_to_every_reusable_stage(self) -> None:
        text = (ROOT / ".github/workflows/normalize.yml").read_text(encoding="utf-8")
        self.assertEqual(text.count("canonical_build_policy: ${{ inputs.canonical_normalization_policy }}"), 15)
        self.assertEqual(text.count('canonical_build_policy: "${{ inputs.canonical_normalization_policy }}"'), 12)
        for filename in (
            "_core-tonal-evidence.yml",
            "_core-chromatic-evidence.yml",
            "_core-restoration-evidence.yml",
        ):
            workflow = (ROOT / ".github/workflows" / filename).read_text(encoding="utf-8")
            self.assertIn("canonical_build_policy: {required: false, type: string, default: auto}", workflow)
            self.assertNotIn("--policy auto", workflow)


if __name__ == "__main__":
    unittest.main()
