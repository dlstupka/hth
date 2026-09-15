from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from hth.canonical_build_evidence import (
    EvidenceError,
    canonical_hash,
    canonicalize_result,
    finalize,
    merge_stores,
    prepare,
    validate_published_results,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class CanonicalBuildEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pipeline = self.root / "pipeline"
        self.source = self.root / "source"
        self.results = self.root / "results"
        self.output = self.root / "output"
        (self.pipeline / "hth").mkdir(parents=True)
        self.source.mkdir()
        self.results.mkdir()
        (self.pipeline / "hth" / "stage.py").write_text("VERSION = 1\n", encoding="utf-8")
        (self.pipeline / "config.json").write_text('{"threshold": 1}\n', encoding="utf-8")
        (self.pipeline / "requirements.txt").write_text("Pillow\n", encoding="utf-8")
        (self.source / "master.docx").write_bytes(b"immutable source")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def args(self, policy: str = "auto", *, artifact_required: bool = False) -> argparse.Namespace:
        return argparse.Namespace(
            scope="hth-preprocess",
            contract_version="1",
            policy=policy,
            mode="production",
            image_limit=0,
            operation=["extract", "analyze", "detect"],
            repository_root=self.pipeline,
            source_root=self.source,
            source_repository="owner/source",
            source_release="HTH-SOURCE-0002",
            source_manifest_sha256="a" * 64,
            source_commit="b" * 40,
            config=[self.pipeline / "config.json"],
            implementation=[self.pipeline / "hth"],
            runtime_contract=[self.pipeline / "requirements.txt"],
            runtime_package=[],
            runtime_component=["doc-ufcn=0.2.0rc4"],
            selection=None,
            results_root=self.results,
            evidence=self.results / "metadata/canonical-build-evidence.json",
            plan=self.root / "plan.json",
            artifact_required=artifact_required,
            pipeline_repository="owner/pipeline",
            pipeline_commit="c" * 40,
            workflow_run_id="123",
            runner_name="test-runner",
            runner_environment="self-hosted",
            runner_os="Windows",
            runner_arch="X64",
            github_output="",
            github_summary="",
        )

    def materialize_outputs(self) -> None:
        image = {
            "global_ordinal": 1,
            "source_docx": "master.docx",
            "source_ordinal": 1,
            "relationship_id": "rId1",
            "media_path": "word/media/image1.jpg",
            "embedded_sha256": "d" * 64,
            "sha256": "e" * 64,
            "analysis_sha256": "f" * 64,
            "thumbnail_sha256": "1" * 64,
            "word_crop_left": 0,
            "word_crop_top": 0,
            "word_crop_right": 0,
            "word_crop_bottom": 0,
            "pipeline_commit": "old",
        }
        analysis = {
            "global_ordinal": 1,
            "quality_status": "pass",
            "quality_score": 0.9,
            "analysis_error": "",
            "pipeline_commit": "old",
        }
        write_json(self.output / "metadata/image_manifest.json", {"records": [image]})
        write_json(self.output / "metadata/exact_duplicates.json", [])
        write_json(self.output / "summary.json", {"image_count": 1})
        write_json(self.output / "page-analysis/page-analysis.json", {"records": [analysis]})
        write_json(
            self.output / "page-analysis/analysis-summary.json",
            {"generated_at_utc": "2026-01-01T00:00:00Z", "page_count": 1},
        )

    def publish_evidence_and_results(self, evidence: dict) -> None:
        for record in evidence["canonical_result"]["artifacts"]:
            generated = next(
                path for logical, path in {
                    "image-manifest": "metadata/image_manifest.json",
                    "exact-duplicates": "metadata/exact_duplicates.json",
                    "preprocess-summary": "summary.json",
                    "page-analysis": "page-analysis/page-analysis.json",
                    "analysis-summary": "page-analysis/analysis-summary.json",
                }.items() if logical == record["logical_name"]
            )
            source = self.output / generated
            target = self.results / record["published_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        target = self.results / "metadata/canonical-build-evidence.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((self.output / "metadata/canonical-build-evidence.json").read_bytes())

    def establish(self) -> dict:
        plan = prepare(self.args())
        self.assertEqual(plan["decision"], "execute")
        self.materialize_outputs()
        final_args = argparse.Namespace(
            plan=self.root / "plan.json",
            output_root=self.output,
            evidence_store=self.results / "metadata/canonical-build-evidence.json",
            evidence_output=self.output / "metadata/canonical-build-evidence.json",
            github_output="",
            github_summary="",
        )
        evidence = finalize(final_args)
        self.publish_evidence_and_results(evidence)
        return evidence

    def test_canonical_result_ignores_telemetry_and_duplicated_provenance(self) -> None:
        left = {"value": 7, "generated_at_utc": "first", "pipeline_commit": "a", "nested": {"elapsed_ms": 1}}
        right = {"nested": {"elapsed_ms": 99}, "pipeline_commit": "b", "generated_at_utc": "second", "value": 7}
        self.assertEqual(canonical_hash(canonicalize_result(left)), canonical_hash(canonicalize_result(right)))

    def test_implementation_fingerprint_ignores_runtime_bytecode(self) -> None:
        cached = self.pipeline / "hth/__pycache__/stage.pyc"
        cached.parent.mkdir()
        cached.write_bytes(b"host-specific bytecode")
        first = prepare(self.args())["effective_build_identity"]
        cached.write_bytes(b"different bytecode")
        second = prepare(self.args())["effective_build_identity"]
        self.assertEqual(first, second)

    def test_runner_is_observational_and_does_not_change_effective_identity(self) -> None:
        first = prepare(self.args())
        second_args = self.args()
        second_args.runner_name = "github-hosted-runner"
        second_args.runner_environment = "github-hosted"
        second_args.runner_os = "Linux"
        second_args.runner_arch = "ARM64"
        second = prepare(second_args)
        self.assertEqual(first["effective_build_identity"], second["effective_build_identity"])
        self.assertNotIn("platform_system", first["effective_inputs"]["runtime"])
        self.assertNotIn("platform_machine", first["effective_inputs"]["runtime"])
        self.assertEqual(second["execution"]["runner"]["runner_os"], "Linux")
        self.assertEqual(second["execution"]["runner"]["runner_arch"], "ARM64")

    def test_first_auto_run_establishes_strict_evidence(self) -> None:
        evidence = self.establish()
        self.assertEqual(evidence["scope"], "hth-preprocess")
        self.assertEqual(evidence["execution"]["activity"], "EXECUTED")
        self.assertEqual(evidence["canonical_result"]["pages"][0]["domain_result"], "APPLY")
        self.assertEqual(len(evidence["effective_inputs"]["source"]["files"]), 1)

    def test_unchanged_auto_run_reuses_and_marks_every_page_skipped(self) -> None:
        evidence = self.establish()
        plan = prepare(self.args())
        self.assertEqual(plan["decision"], "reuse")
        self.assertEqual(plan["activity"], "REUSED")
        self.assertEqual(plan["page_evaluations"], [{
            "global_ordinal": 1,
            "operation_identity": evidence["canonical_result"]["pages"][0]["operation_identity"],
            "activity": "REUSED",
            "domain_result": "SKIP",
        }])

    def test_artifact_request_executes_but_requires_incumbent_equivalence(self) -> None:
        self.establish()
        plan = prepare(self.args(artifact_required=True))
        self.assertEqual(plan["decision"], "execute")
        self.assertTrue(plan["comparison_required"])

    def test_force_verify_fails_if_result_changes_for_same_identity(self) -> None:
        self.establish()
        prepare(self.args("force-verify"))
        write_json(self.output / "summary.json", {"image_count": 2})
        final_args = argparse.Namespace(
            plan=self.root / "plan.json",
            output_root=self.output,
            evidence_store=self.results / "metadata/canonical-build-evidence.json",
            evidence_output=self.output / "metadata/canonical-build-evidence.json",
            github_output="",
            github_summary="",
        )
        with self.assertRaisesRegex(EvidenceError, "Determinism verification failed"):
            finalize(final_args)

    def test_corrupt_persisted_result_fails_instead_of_silently_rebuilding(self) -> None:
        evidence = self.establish()
        write_json(self.results / "reports/preprocess-summary.json", {"image_count": 999})
        with self.assertRaisesRegex(EvidenceError, "Persisted canonical result mismatch"):
            validate_published_results(evidence, self.results)
        with self.assertRaisesRegex(EvidenceError, "Persisted canonical result mismatch"):
            prepare(self.args())

    def test_self_consistent_but_incomplete_artifact_contract_is_rejected(self) -> None:
        evidence = self.establish()
        evidence["canonical_result"]["artifacts"] = evidence["canonical_result"]["artifacts"][:-1]
        evidence["canonical_result"]["identity"] = canonical_hash({
            "artifacts": evidence["canonical_result"]["artifacts"],
            "pages": [
                {k: v for k, v in page.items() if k not in {"activity", "domain_result", "operation_identity"}}
                for page in evidence["canonical_result"]["pages"]
            ],
        })
        store_path = self.results / "metadata/canonical-build-evidence.json"
        store = json.loads(store_path.read_text(encoding="utf-8"))
        store["records"][evidence["effective_build_identity"]] = evidence
        write_json(store_path, store)
        with self.assertRaisesRegex(EvidenceError, "artifact contract"):
            prepare(self.args())

    def test_changed_input_is_new_identity_and_rebuilds_under_auto(self) -> None:
        prior = self.establish()
        (self.pipeline / "config.json").write_text('{"threshold": 2}\n', encoding="utf-8")
        plan = prepare(self.args())
        self.assertEqual(plan["decision"], "execute")
        self.assertNotEqual(plan["effective_build_identity"], prior["effective_build_identity"])
        self.assertFalse(plan["comparison_required"])

    def test_new_identity_is_added_without_discarding_prior_evidence(self) -> None:
        prior = self.establish()
        (self.pipeline / "config.json").write_text('{"threshold": 2}\n', encoding="utf-8")
        prepare(self.args())
        final_args = argparse.Namespace(
            plan=self.root / "plan.json",
            output_root=self.output,
            evidence_store=self.results / "metadata/canonical-build-evidence.json",
            evidence_output=self.output / "metadata/canonical-build-evidence.json",
            github_output="",
            github_summary="",
        )
        current = finalize(final_args)
        store = json.loads((self.output / "metadata/canonical-build-evidence.json").read_text(encoding="utf-8"))
        self.assertEqual(
            set(store["records"]),
            {prior["effective_build_identity"], current["effective_build_identity"]},
        )
        self.assertEqual(store["authoritative_identity"], current["effective_build_identity"])

    def test_publication_retry_merge_preserves_concurrent_identity(self) -> None:
        prior = self.establish()
        base = self.root / "base.json"
        base.write_bytes((self.results / "metadata/canonical-build-evidence.json").read_bytes())
        (self.pipeline / "config.json").write_text('{"threshold": 2}\n', encoding="utf-8")
        prepare(self.args())
        current = finalize(argparse.Namespace(
            plan=self.root / "plan.json",
            output_root=self.output,
            evidence_store=self.root / "missing-store.json",
            evidence_output=self.root / "incoming.json",
            github_output="",
            github_summary="",
        ))
        merged = merge_stores(argparse.Namespace(
            scope="hth-preprocess",
            base=base,
            incoming=self.root / "incoming.json",
            output=self.root / "merged.json",
        ))
        self.assertEqual(
            set(merged["records"]),
            {prior["effective_build_identity"], current["effective_build_identity"]},
        )

    def test_audit_and_force_verify_require_exact_current_evidence(self) -> None:
        for policy in ("audit", "force-verify"):
            with self.subTest(policy=policy):
                with self.assertRaisesRegex(EvidenceError, "requires exact persisted evidence"):
                    prepare(self.args(policy))

    def test_rebuild_rejects_unchanged_identity(self) -> None:
        self.establish()
        with self.assertRaisesRegex(EvidenceError, "use force-verify"):
            prepare(self.args("rebuild"))


class CanonicalBuildEvidenceWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.entry = (cls.root / ".github/workflows/preprocess.yml").read_text(encoding="utf-8")
        cls.core = (cls.root / ".github/workflows/_core-hth.yml").read_text(encoding="utf-8")

    def test_collection_dispatch_exposes_all_strict_policies(self) -> None:
        block = self.entry.split("canonical_build_policy:", 1)[1].split("permissions:", 1)[0]
        for policy in ("auto", "audit", "force-verify", "rebuild"):
            self.assertIn(f"- {policy}", block)
        self.assertIn("canonical_build_policy: ${{ inputs.canonical_build_policy }}", self.entry)

    def test_core_declares_effective_inputs_and_runtime(self) -> None:
        block = self.core.split("- name: Evaluate HTH Preprocess Canonical Build Evidence", 1)[1].split(
            "# STAGE_PREPROCESS", 1
        )[0]
        self.assertIn("python -m hth.canonical_build_evidence prepare", block)
        self.assertIn("--source-manifest-sha256", block)
        self.assertIn("--implementation hth-pipeline/hth", block)
        self.assertIn("--runtime-contract hth-pipeline/requirements.txt", block)
        self.assertIn("--selection \"$RUNNER_TEMP/preferred-document-detector.json\"", block)
        self.assertIn("--artifact-required", block)
        self.assertIn('--runner-environment "${{ runner.environment }}"', block)
        self.assertIn('--runner-os "${{ runner.os }}"', block)
        self.assertIn('--runner-arch "${{ runner.arch }}"', block)

    def test_production_execution_is_gated_and_results_are_finalized(self) -> None:
        self.assertIn("steps.cbe_plan.outputs.decision == 'execute'", self.core)
        self.assertIn("python -m hth.canonical_build_evidence finalize", self.core)
        self.assertIn(
            '--evidence-output "$OUTPUT_DIRECTORY/metadata/canonical-build-evidence.json"',
            self.core,
        )
        self.assertIn("--evidence-store results-repo/metadata/canonical-build-evidence.json", self.core)
        self.assertIn("python -m hth.canonical_build_evidence merge", self.core)
        self.assertIn("canonical_build_evidence:", self.core)
        self.assertIn("effective_build_identity:", self.core)
        self.assertIn("canonical_result_identity:", self.core)


if __name__ == "__main__":
    unittest.main()
