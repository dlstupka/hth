from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hth.canonical_build_evidence import (
    CROP_FRAMING_ASSESSMENT_SCOPE,
    CHROMATIC_INTEGRATION_SCOPE,
    EvidenceError,
    SCOPE_ARTIFACT_PROFILES,
    SCOPE_OPERATION_CONTRACTS,
    TONAL_ASSESSMENT_SCOPE,
    TONAL_INTEGRATION_SCOPE,
    TONAL_METHOD_ASSESSMENT_SCOPE,
    canonical_hash,
    canonical_operations,
    canonicalize_result,
    fingerprint_paths,
    fingerprint_source_inputs,
    finalize,
    merge_stores,
    prepare,
    restore_variant,
    snapshot_variant,
    validate_cache_snapshot,
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
        (self.pipeline / "hth" / "unrelated_report.py").write_text("FORMAT = 1\n", encoding="utf-8")
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
            operation=[
                "source-image-extract",
                "word-crop",
                "analysis-derivative",
                "thumbnail",
                "page-quality-analysis",
                "physical-document-detection",
            ],
            repository_root=self.pipeline,
            source_root=self.source,
            source_repository="owner/source",
            source_release="HTH-SOURCE-0002",
            source_manifest_sha256="a" * 64,
            source_commit="b" * 40,
            config=[self.pipeline / "config.json"],
            implementation=[self.pipeline / "hth/stage.py"],
            detector_implementation_root=None,
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

    def test_summary_links_existing_persisted_evidence_store(self) -> None:
        args = self.args()
        summary = self.root / "summary.md"
        write_json(args.evidence, {
            "schema_version": "1.0",
            "evidence_type": "canonical-build-evidence-store",
            "scope": "hth-preprocess",
            "records": {},
        })
        args.results_repository = "owner/results"
        args.results_ref = "abc123"
        args.github_summary = str(summary)

        prepare(args)

        self.assertIn(
            "[`metadata/canonical-build-evidence.json`](https://github.com/owner/results/blob/abc123/metadata/canonical-build-evidence.json)",
            summary.read_text(encoding="utf-8"),
        )

    def test_summary_links_first_build_evidence_through_durable_branch(self) -> None:
        args = self.args()
        summary = self.root / "summary.md"
        args.results_repository = "owner/results"
        args.results_ref = "abc123"
        args.github_summary = str(summary)

        prepare(args)

        self.assertIn(
            "[`metadata/canonical-build-evidence.json`](https://github.com/owner/results/blob/main/metadata/canonical-build-evidence.json)",
            summary.read_text(encoding="utf-8"),
        )

    def test_logical_source_inputs_ignore_physical_staging_names(self) -> None:
        first = self.source / "workflow-a.json"
        second_root = self.root / "refactored-workflow"
        second_root.mkdir()
        second = second_root / "generic-name.json"
        first.write_text('{"value": 1}\n', encoding="utf-8")
        second.write_bytes(first.read_bytes())

        original = self.args()
        original.source_input = [f"semantic/input.json={first}"]
        original_plan = prepare(original)

        refactored = self.args()
        refactored.source_root = second_root
        refactored.source_input = [f"semantic/input.json={second}"]
        refactored.plan = self.root / "refactored-plan.json"
        refactored_plan = prepare(refactored)

        self.assertEqual(
            original_plan["effective_build_identity"],
            refactored_plan["effective_build_identity"],
        )
        self.assertEqual(
            original_plan["effective_inputs"]["source"]["files"],
            [{
                "path": "semantic/input.json",
                "bytes": first.stat().st_size,
                "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
            }],
        )

    def test_registered_operation_contract_rejects_yaml_drift(self) -> None:
        args = self.args()
        args.operation = ["renamed-by-workflow-refactor"]
        with self.assertRaisesRegex(EvidenceError, "operation contract drift"):
            prepare(args)

    def test_every_cbe_scope_has_one_canonical_operation_contract(self) -> None:
        self.assertEqual(
            set(SCOPE_ARTIFACT_PROFILES),
            set(SCOPE_OPERATION_CONTRACTS) | {"hth-normalization"},
        )

    def test_consolidated_integrations_retain_pre_refactor_operation_contracts(self) -> None:
        self.assertEqual(canonical_operations(TONAL_INTEGRATION_SCOPE, []), [
            "consume-immutable-photometric-collection",
            "classify-every-page-by-tonal-metrics",
            "apply-only-validated-bounded-tonal-method",
            "preserve-exceptions-and-continue",
            "package-immutable-lossless-collection",
        ])
        self.assertEqual(canonical_operations(CHROMATIC_INTEGRATION_SCOPE, []), [
            "consume-immutable-tonal-collection",
            "classify-every-page-by-chromatic-metrics",
            "apply-only-validated-bounded-chromatic-method",
            "preserve-exceptions-and-continue",
            "package-immutable-lossless-collection",
        ])

    def test_logical_aliases_preserve_legacy_tonal_contract_fingerprints(self) -> None:
        legacy = self.root / "legacy-tonal-contract"
        refactored = self.root / "generic-contract"
        legacy.mkdir()
        refactored.mkdir()
        payloads = {
            "release.json": b"release",
            "photometric-normalization-manifest.json": b"manifest",
            "tonal-assessment.json": b"assessment",
            "tonal-method-assessment.json": b"methods",
        }
        generic_names = {
            "release.json": "upstream-release.json",
            "photometric-normalization-manifest.json": "upstream-manifest.json",
            "tonal-assessment.json": "assessment.json",
            "tonal-method-assessment.json": "method-assessment.json",
        }
        for name, payload in payloads.items():
            (legacy / name).write_bytes(payload)
            (refactored / generic_names[name]).write_bytes(payload)

        expected = fingerprint_paths([legacy], legacy)
        actual = fingerprint_source_inputs([
            f"{logical}={refactored / physical}"
            for logical, physical in generic_names.items()
        ])
        self.assertEqual(actual, expected)

    def test_tonal_scope_prepares_with_registered_evidence_path(self) -> None:
        args = self.args()
        args.scope = TONAL_INTEGRATION_SCOPE
        args.operation = []
        args.evidence = self.results / "normalization/tonal-integration/canonical-build-evidence.json"
        plan = prepare(args)
        self.assertEqual(plan["scope"], TONAL_INTEGRATION_SCOPE)
        self.assertEqual(plan["decision"], "execute")

    def test_tonal_assessment_scope_establishes_and_reuses_compact_evidence(self) -> None:
        args = self.args()
        args.scope = TONAL_ASSESSMENT_SCOPE
        args.operation = []
        args.evidence = self.results / "normalization/tonal/canonical-build-evidence.json"
        plan = prepare(args)
        self.assertEqual(plan["decision"], "execute")
        write_json(self.output / "assessment.json", {
            "assessment_identity": "1" * 64,
            "pages": [{
                "global_ordinal": 1,
                "input_pixel_sha256": "2" * 64,
                "measurement": {"decision": "preserve", "tonal_span": 0.75},
            }],
        })
        evidence_output = self.output / "canonical-build-evidence.json"
        evidence = finalize(argparse.Namespace(
            plan=args.plan,
            output_root=self.output,
            evidence_store=args.evidence,
            evidence_output=evidence_output,
            github_output="",
            github_summary="",
        ))
        published = self.results / "normalization/tonal/assessment.json"
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_bytes((self.output / "assessment.json").read_bytes())
        args.evidence.write_bytes(evidence_output.read_bytes())
        self.assertEqual(evidence["canonical_result"]["pages"][0]["global_ordinal"], 1)
        self.assertEqual(prepare(args)["decision"], "reuse")

    def test_crop_framing_collapses_algorithm_variants_into_canonical_page_evidence(self) -> None:
        args = self.args()
        args.scope = CROP_FRAMING_ASSESSMENT_SCOPE
        args.operation = []
        args.evidence = self.results / "normalization/crop-framing/canonical-build-evidence.json"
        prepare(args)
        write_json(self.output / "assessment.json", {
            "algorithms": {"axis-aligned": {}, "perspective": {}},
            "pages": [
                {"global_ordinal": 2, "algorithm": "perspective", "output_sha256": "2" * 64},
                {"global_ordinal": 1, "algorithm": "axis-aligned", "output_sha256": "3" * 64},
                {"global_ordinal": 2, "algorithm": "axis-aligned", "output_sha256": "4" * 64},
                {"global_ordinal": 1, "algorithm": "perspective", "output_sha256": "5" * 64},
            ],
        })
        write_json(self.output / "detector-selection.json", {"detector": "test"})
        write_json(self.output / "geometry-evidence.json", {"records": []})
        evidence = finalize(argparse.Namespace(
            plan=args.plan,
            output_root=self.output,
            evidence_store=args.evidence,
            evidence_output=self.output / "canonical-build-evidence.json",
            github_output="",
            github_summary="",
        ))
        self.assertEqual(
            [page["global_ordinal"] for page in evidence["canonical_result"]["pages"]],
            [1, 2],
        )
        self.assertNotEqual(
            evidence["canonical_result"]["pages"][0]["evidence_record_sha256"],
            evidence["canonical_result"]["pages"][1]["evidence_record_sha256"],
        )

    def test_crop_framing_rejects_duplicate_algorithm_variant(self) -> None:
        args = self.args()
        args.scope = CROP_FRAMING_ASSESSMENT_SCOPE
        args.operation = []
        args.evidence = self.results / "normalization/crop-framing/canonical-build-evidence.json"
        prepare(args)
        write_json(self.output / "assessment.json", {
            "algorithms": {"axis-aligned": {}},
            "pages": [
                {"global_ordinal": 1, "algorithm": "axis-aligned", "output_sha256": "3" * 64},
                {"global_ordinal": 1, "algorithm": "axis-aligned", "output_sha256": "4" * 64},
            ],
        })
        write_json(self.output / "detector-selection.json", {"detector": "test"})
        write_json(self.output / "geometry-evidence.json", {"records": []})
        with self.assertRaisesRegex(EvidenceError, "duplicate algorithm evidence"):
            finalize(argparse.Namespace(
                plan=args.plan,
                output_root=self.output,
                evidence_store=args.evidence,
                evidence_output=self.output / "canonical-build-evidence.json",
                github_output="",
                github_summary="",
            ))

    def test_empty_bounded_method_candidate_set_is_valid_compact_evidence(self) -> None:
        args = self.args()
        args.scope = TONAL_METHOD_ASSESSMENT_SCOPE
        args.operation = []
        args.evidence = self.results / "normalization/tonal-methods/canonical-build-evidence.json"
        prepare(args)
        write_json(self.output / "assessment.json", {
            "method_assessment_identity": "3" * 64,
            "candidate_count": 0,
            "pages": [],
        })
        evidence = finalize(argparse.Namespace(
            plan=args.plan,
            output_root=self.output,
            evidence_store=args.evidence,
            evidence_output=self.output / "canonical-build-evidence.json",
            github_output="",
            github_summary="",
        ))
        self.assertEqual(evidence["canonical_result"]["pages"], [])

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

    def materialize_preprocess_cache_companions(self) -> None:
        for relative in (
            "BUILD-INFO.yaml", "metadata/image_manifest.csv", "metadata/page_map_template.csv",
            "analysis/page-analysis.csv", "analysis/review-queue.csv",
        ):
            target = self.results / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("fixture\n", encoding="utf-8")

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

    def establish_second_variant(self) -> tuple[dict, dict]:
        prior = self.establish()
        (self.pipeline / "config.json").write_text('{"threshold": 2}\n', encoding="utf-8")
        self.assertEqual(prepare(self.args())["decision"], "execute")
        write_json(self.output / "summary.json", {"image_count": 2})
        current = finalize(argparse.Namespace(
            plan=self.root / "plan.json",
            output_root=self.output,
            evidence_store=self.results / "metadata/canonical-build-evidence.json",
            evidence_output=self.output / "metadata/canonical-build-evidence.json",
            github_output="",
            github_summary="",
        ))
        self.publish_evidence_and_results(current)
        (self.pipeline / "config.json").write_text('{"threshold": 1}\n', encoding="utf-8")
        return prior, current

    def test_canonical_result_ignores_telemetry_and_duplicated_provenance(self) -> None:
        left = {"value": 7, "generated_at_utc": "first", "pipeline_commit": "a", "nested": {"elapsed_ms": 1}}
        right = {"nested": {"elapsed_ms": 99}, "pipeline_commit": "b", "generated_at_utc": "second", "value": 7}
        self.assertEqual(canonical_hash(canonicalize_result(left)), canonical_hash(canonicalize_result(right)))

    def test_implementation_fingerprint_ignores_runtime_bytecode(self) -> None:
        args = self.args()
        args.implementation = [self.pipeline / "hth"]
        cached = self.pipeline / "hth/__pycache__/stage.pyc"
        cached.parent.mkdir()
        cached.write_bytes(b"host-specific bytecode")
        first = prepare(args)["effective_build_identity"]
        cached.write_bytes(b"different bytecode")
        second = prepare(args)["effective_build_identity"]
        self.assertEqual(first, second)

    def test_boundary_excludes_unrelated_code_and_tracks_selected_detector_closure(self) -> None:
        geometry = self.pipeline / "hth/geometry"
        geometry.mkdir()
        (geometry / "detector_selected.py").write_text(
            "from .helper import VALUE\nRESULT = VALUE\n", encoding="utf-8"
        )
        (geometry / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        (geometry / "detector_unselected.py").write_text("RESULT = 1\n", encoding="utf-8")
        selection = self.root / "selection.json"
        write_json(selection, {"detector": "selected", "parameters": {}})
        args = self.args()
        args.selection = selection
        args.detector_implementation_root = geometry
        first = prepare(args)["effective_build_identity"]

        (self.pipeline / "hth/unrelated_report.py").write_text("FORMAT = 2\n", encoding="utf-8")
        (geometry / "detector_unselected.py").write_text("RESULT = 2\n", encoding="utf-8")
        second = prepare(args)["effective_build_identity"]
        self.assertEqual(first, second)

        (geometry / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
        third = prepare(args)["effective_build_identity"]
        self.assertNotEqual(first, third)

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
        self.assertGreaterEqual(evidence["execution"]["elapsed_seconds"], 0)
        self.assertEqual(evidence["canonical_result"]["pages"][0]["domain_result"], "APPLY")
        self.assertEqual(len(evidence["effective_inputs"]["source"]["files"]), 1)

    def test_first_build_summary_explains_missing_evidence(self) -> None:
        args = self.args()
        args.github_summary = str(self.root / "summary.md")
        prepare(args)
        self.assertIn(
            "- Decision: `execute`\n- Why: No prior CBE evidence for this scope.",
            Path(args.github_summary).read_text(encoding="utf-8"),
        )

    def test_runtime_mismatch_summary_names_actual_versions(self) -> None:
        prior_runtime = {
            "python_implementation": "CPython", "python_version": "3.12.14",
            "python_cache_tag": "cpython-312", "packages": {"numpy": "2.5.3"},
        }
        current_runtime = {
            **prior_runtime, "python_version": "3.12.0",
            "packages": {"numpy": "2.4.6"},
        }
        with patch("hth.canonical_build_evidence.runtime_identity", return_value=prior_runtime):
            self.establish()
        args = self.args()
        args.github_summary = str(self.root / "summary.md")
        with patch("hth.canonical_build_evidence.runtime_identity", return_value=current_runtime):
            plan = prepare(args)
        self.assertEqual(plan["decision"], "execute")
        self.assertIn(
            "- Why: No exact CBE match; changed since authoritative build: "
            "Python `3.12.14` → `3.12.0`; numpy `2.5.3` → `2.4.6`.",
            Path(args.github_summary).read_text(encoding="utf-8"),
        )

    def test_configuration_mismatch_summary_names_changed_file(self) -> None:
        self.establish()
        (self.pipeline / "config.json").write_text('{"threshold": 2}\n', encoding="utf-8")
        args = self.args()
        args.github_summary = str(self.root / "summary.md")
        prepare(args)
        self.assertIn(
            "- Why: No exact CBE match; changed since authoritative build: "
            "configuration: `config.json`.",
            Path(args.github_summary).read_text(encoding="utf-8"),
        )

    def test_summary_reports_executed_and_reused_build_stage_time(self) -> None:
        args = self.args()
        summary = self.root / "summary.md"
        args.github_summary = str(summary)
        prepare(args)
        self.materialize_outputs()
        evidence_output = self.output / "metadata/canonical-build-evidence.json"
        evidence = finalize(argparse.Namespace(
            plan=args.plan,
            output_root=self.output,
            evidence_store=args.evidence,
            evidence_output=evidence_output,
            github_output="",
            github_summary=str(summary),
        ))
        rendered = summary.read_text(encoding="utf-8")
        self.assertIn("- Build stage time:", rendered)
        self.assertIn(f"(`{evidence['execution']['elapsed_seconds']}` seconds)", rendered)

        expected_elapsed = evidence["execution"].pop("elapsed_seconds")
        self.publish_evidence_and_results(evidence)
        reused_summary = self.root / "reused-summary.md"
        reused_args = self.args()
        reused_args.github_summary = str(reused_summary)
        prepare(reused_args)
        reused_rendered = reused_summary.read_text(encoding="utf-8")
        self.assertIn("- Why: Exact effective inputs and published artifacts verified.", reused_rendered)
        self.assertIn("- Build stage time:", reused_rendered)
        self.assertIn(f"(`{expected_elapsed}` seconds)", reused_rendered)

    def test_unchanged_auto_run_reuses_and_marks_every_page_skipped(self) -> None:
        evidence = self.establish()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            plan = prepare(self.args())
        self.assertEqual(plan["decision"], "reuse")
        self.assertEqual(plan["activity"], "REUSED")
        self.assertEqual(plan["page_evaluations"], [{
            "global_ordinal": 1,
            "operation_identity": evidence["canonical_result"]["pages"][0]["operation_identity"],
            "activity": "REUSED",
            "domain_result": "SKIP",
        }])
        log_line = stdout.getvalue()
        self.assertIn("[canonical-build-evidence]", log_line)
        self.assertIn("policy=auto decision=reuse activity=REUSED domain_result=SKIP", log_line)
        self.assertIn("pages_marked_unnecessary=1", log_line)
        self.assertIn(f"identity={plan['effective_build_identity']}", log_line)

    def test_artifact_request_executes_but_requires_incumbent_equivalence(self) -> None:
        self.establish()
        args = self.args(artifact_required=True)
        args.github_summary = str(self.root / "summary.md")
        plan = prepare(args)
        self.assertEqual(plan["decision"], "execute")
        self.assertTrue(plan["comparison_required"])
        self.assertIn(
            "- Why: Full artifact requested; rebuilt output must match exact CBE evidence.",
            Path(args.github_summary).read_text(encoding="utf-8"),
        )

    def test_explicit_policy_summaries_explain_the_decision(self) -> None:
        self.establish()
        for policy, expected in (
            ("audit", "Exact effective inputs and published artifacts audited; execution skipped."),
            ("force-verify", "Explicit verification of an exact CBE match; rebuilt output must match."),
        ):
            with self.subTest(policy=policy):
                args = self.args(policy)
                args.github_summary = str(self.root / f"{policy}-summary.md")
                prepare(args)
                self.assertIn(
                    f"- Why: {expected}",
                    Path(args.github_summary).read_text(encoding="utf-8"),
                )

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

    def test_prior_exact_record_is_restored_with_result_verification(self) -> None:
        prior, current = self.establish_second_variant()
        self.assertNotEqual(
            prior["canonical_result"]["identity"], current["canonical_result"]["identity"]
        )
        with self.assertRaisesRegex(EvidenceError, "currently published"):
            prepare(self.args("audit"))

        args = self.args()
        args.github_summary = str(self.root / "restore-summary.md")
        plan = prepare(args)
        self.assertEqual(plan["decision"], "execute")
        self.assertTrue(plan["comparison_required"])
        self.assertTrue(plan["restoring_prior_identity"])
        self.assertEqual(plan["incumbent_result_identity"], prior["canonical_result"]["identity"])
        self.assertIn(
            "restoring this result with incumbent-equivalence verification",
            Path(args.github_summary).read_text(encoding="utf-8"),
        )

        self.materialize_outputs()
        restored = finalize(argparse.Namespace(
            plan=args.plan,
            output_root=self.output,
            evidence_store=args.evidence,
            evidence_output=self.output / "metadata/canonical-build-evidence.json",
            github_output="",
            github_summary="",
        ))
        self.assertEqual(restored["canonical_result"]["identity"], prior["canonical_result"]["identity"])
        self.assertTrue(restored["execution"]["verified_against_incumbent"])
        self.publish_evidence_and_results(restored)
        self.assertEqual(prepare(self.args())["decision"], "reuse")

        (self.pipeline / "config.json").write_text('{"threshold": 2}\n', encoding="utf-8")
        reverse = prepare(self.args())
        self.assertEqual(reverse["decision"], "execute")
        self.assertTrue(reverse["restoring_prior_identity"])
        self.assertEqual(reverse["incumbent_result_identity"], current["canonical_result"]["identity"])

    def test_prior_record_does_not_hide_corrupt_current_publication(self) -> None:
        self.establish_second_variant()
        write_json(self.results / "reports/preprocess-summary.json", {"image_count": 999})
        with self.assertRaisesRegex(EvidenceError, "Persisted canonical result mismatch"):
            prepare(self.args())

    def test_prior_record_restoration_rejects_different_rebuilt_result(self) -> None:
        self.establish_second_variant()
        self.assertTrue(prepare(self.args())["restoring_prior_identity"])
        with self.assertRaisesRegex(EvidenceError, "Determinism verification failed"):
            finalize(argparse.Namespace(
                plan=self.root / "plan.json",
                output_root=self.output,
                evidence_store=self.results / "metadata/canonical-build-evidence.json",
                evidence_output=self.output / "metadata/canonical-build-evidence.json",
                github_output="",
                github_summary="",
            ))

    def test_two_runtime_variants_restore_from_verified_snapshots(self) -> None:
        prior = self.establish()
        self.materialize_preprocess_cache_companions()
        (self.results / "analysis/old-only.csv").write_text("old\n", encoding="utf-8")
        snapshot_args = argparse.Namespace(
            scope="hth-preprocess", source_root=self.results, cache_root=self.results,
            evidence=self.results / "metadata/canonical-build-evidence.json", identity="",
        )
        snapshot_variant(snapshot_args)
        # Capture companions as well as the five canonical CBE artifacts.
        snapshot_dir = self.results / "cbe-cache/hth-preprocess" / prior["effective_build_identity"]
        self.assertTrue(snapshot_dir.is_dir())

        (self.pipeline / "config.json").write_text('{"threshold": 2}\n', encoding="utf-8")
        self.assertEqual(prepare(self.args())["decision"], "execute")
        write_json(self.output / "summary.json", {"image_count": 2})
        current = finalize(argparse.Namespace(
            plan=self.root / "plan.json", output_root=self.output,
            evidence_store=snapshot_args.evidence,
            evidence_output=self.output / "metadata/canonical-build-evidence.json",
            github_output="", github_summary="",
        ))
        self.publish_evidence_and_results(current)
        (self.results / "analysis/old-only.csv").unlink()
        (self.results / "analysis/new-only.csv").write_text("new\n", encoding="utf-8")
        snapshot_variant(snapshot_args)
        (self.pipeline / "config.json").write_text('{"threshold": 1}\n', encoding="utf-8")
        first = prepare(self.args())
        self.assertEqual(first["decision"], "restore")
        self.assertEqual(first["activity"], "REUSED")
        self.assertFalse(first["comparison_required"])
        self.assertEqual(first["resource_utilization"]["canonical_evidence_cache"]["action"], "restored")
        restore_variant(argparse.Namespace(plan=self.root / "plan.json", results_root=self.results, github_summary=""))
        validate_published_results(prior, self.results)
        self.assertTrue((self.results / "analysis/old-only.csv").is_file())
        self.assertFalse((self.results / "analysis/new-only.csv").exists())
        self.assertEqual(prepare(self.args())["decision"], "reuse")

        (self.pipeline / "config.json").write_text('{"threshold": 2}\n', encoding="utf-8")
        self.assertEqual(prepare(self.args())["decision"], "restore")
        restore_variant(argparse.Namespace(plan=self.root / "plan.json", results_root=self.results, github_summary=""))
        validate_published_results(current, self.results)
        self.assertFalse((self.results / "analysis/old-only.csv").exists())
        self.assertTrue((self.results / "analysis/new-only.csv").is_file())

    def test_variant_snapshot_corruption_fails_closed(self) -> None:
        prior = self.establish()
        self.materialize_preprocess_cache_companions()
        snapshot_variant(argparse.Namespace(
            scope="hth-preprocess", source_root=self.results, cache_root=self.results,
            evidence=self.results / "metadata/canonical-build-evidence.json", identity="",
        ))
        path = self.results / "cbe-cache/hth-preprocess" / prior["effective_build_identity"] / "reports/preprocess-summary.json"
        path.write_text('{"image_count": 999}\n', encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "cache file mismatch"):
            validate_cache_snapshot(self.results, "hth-preprocess", prior)

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

    def test_normalization_scope_establishes_and_reuses_compact_page_evidence(self) -> None:
        args = self.args()
        args.scope = "hth-normalization"
        args.operation = [
            "canonical-source-reconstruction",
            "axis-aligned-document-crop",
            "lossless-png-encoding",
            "pixel-roundtrip-verification",
        ]
        args.evidence = self.results / "normalization/canonical-build-evidence.json"
        plan = prepare(args)
        self.assertEqual(plan["decision"], "execute")
        write_json(self.output / "normalization-manifest.json", {
            "schema_version": "1.0",
            "pages": [{
                "global_ordinal": 1,
                "source_sha256": "1" * 64,
                "output_sha256": "2" * 64,
                "output_pixel_sha256": "3" * 64,
                "crop_left": 1,
                "crop_top": 2,
                "crop_right_exclusive": 101,
                "crop_bottom_exclusive": 202,
                "source_width": 120,
                "source_height": 220,
                "output_width": 100,
                "output_height": 200,
            }],
        })
        evidence = finalize(argparse.Namespace(
            plan=args.plan,
            output_root=self.output,
            evidence_store=args.evidence,
            evidence_output=self.output / "canonical-build-evidence.json",
            github_output="",
            github_summary="",
        ))
        published_manifest = self.results / "normalization/normalization-manifest.json"
        published_manifest.parent.mkdir(parents=True, exist_ok=True)
        published_manifest.write_bytes((self.output / "normalization-manifest.json").read_bytes())
        args.evidence.write_bytes((self.output / "canonical-build-evidence.json").read_bytes())

        page = evidence["canonical_result"]["pages"][0]
        self.assertEqual(page["source_image_sha256"], "1" * 64)
        self.assertEqual(page["normalized_image_sha256"], "2" * 64)
        self.assertEqual(prepare(args)["decision"], "reuse")

    def test_normalization_runtime_variants_restore_without_reexecution(self) -> None:
        args = self.args()
        args.scope = "hth-normalization"
        args.operation = [
            "canonical-source-reconstruction", "axis-aligned-document-crop",
            "lossless-png-encoding", "pixel-roundtrip-verification",
        ]
        args.evidence = self.results / "normalization/canonical-build-evidence.json"

        def publish(evidence: dict) -> None:
            destination = self.results / "normalization/normalization-manifest.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((self.output / "normalization-manifest.json").read_bytes())
            args.evidence.write_bytes((self.output / "canonical-build-evidence.json").read_bytes())
            (self.results / "normalization/normalization-manifest.csv").write_text("page\n1\n", encoding="utf-8")
            (self.results / "normalization/summary.md").write_text(
                evidence["effective_build_identity"] + "\n", encoding="utf-8"
            )

        def finish() -> dict:
            return finalize(argparse.Namespace(
                plan=args.plan, output_root=self.output, evidence_store=args.evidence,
                evidence_output=self.output / "canonical-build-evidence.json",
                github_output="", github_summary="",
            ))

        self.assertEqual(prepare(args)["decision"], "execute")
        manifest = {"schema_version": "1.0", "pages": [{
            "global_ordinal": 1, "source_sha256": "1" * 64,
            "output_sha256": "2" * 64, "output_pixel_sha256": "3" * 64,
            "crop_left": 1, "crop_top": 2, "crop_right_exclusive": 101,
            "crop_bottom_exclusive": 202, "source_width": 120,
            "source_height": 220, "output_width": 100, "output_height": 200,
        }]}
        write_json(self.output / "normalization-manifest.json", manifest)
        prior = finish()
        publish(prior)
        cache_args = argparse.Namespace(
            scope=args.scope, source_root=self.results, cache_root=self.results,
            evidence=args.evidence, identity="",
        )
        snapshot_variant(cache_args)

        (self.pipeline / "config.json").write_text('{"threshold": 2}\n', encoding="utf-8")
        self.assertEqual(prepare(args)["decision"], "execute")
        manifest["pages"][0]["output_sha256"] = "4" * 64
        write_json(self.output / "normalization-manifest.json", manifest)
        current = finish()
        publish(current)
        snapshot_variant(cache_args)

        (self.pipeline / "config.json").write_text('{"threshold": 1}\n', encoding="utf-8")
        self.assertEqual(prepare(args)["decision"], "restore")
        restore_variant(argparse.Namespace(plan=args.plan, results_root=self.results, github_summary=""))
        validate_published_results(prior, self.results)
        self.assertEqual(
            (self.results / "normalization/summary.md").read_text(encoding="utf-8").strip(),
            prior["effective_build_identity"],
        )
        (self.pipeline / "config.json").write_text('{"threshold": 2}\n', encoding="utf-8")
        self.assertEqual(prepare(args)["decision"], "restore")
        restore_variant(argparse.Namespace(plan=args.plan, results_root=self.results, github_summary=""))
        validate_published_results(current, self.results)

    def test_photometric_integration_scope_establishes_and_reuses_page_complete_evidence(self) -> None:
        args = self.args()
        args.scope = "hth-photometric-integration"
        args.operation = []
        args.evidence = self.results / "normalization/photometric-integration/canonical-build-evidence.json"
        plan = prepare(args)
        self.assertEqual(plan["decision"], "execute")
        write_json(self.output / "photometric-normalization-manifest.json", {
            "pages": [{
                "global_ordinal": 1,
                "route": "apply",
                "pipeline_action": "corrected-and-continue",
                "evidence_source": "held-out",
                "input_pixel_sha256": "1" * 64,
                "output_pixel_sha256": "2" * 64,
                "output_sha256": "3" * 64,
                "output_width": 100,
                "output_height": 200,
                "method_safe": True,
            }],
        })
        for name in (
            "integration-plan.json",
            "materialization-evidence.json",
            "applied-integration-policy.json",
            "release.json",
        ):
            write_json(self.output / name, {"name": name, "identity": "4" * 64})
        evidence_output = self.output / "canonical-build-evidence.json"
        evidence = finalize(argparse.Namespace(
            plan=args.plan,
            output_root=self.output,
            evidence_store=args.evidence,
            evidence_output=evidence_output,
            github_output="",
            github_summary="",
        ))
        generated = {
            "photometric-normalization-manifest": "photometric-normalization-manifest.json",
            "integration-plan": "integration-plan.json",
            "materialization-evidence": "materialization-evidence.json",
            "applied-integration-policy": "applied-integration-policy.json",
            "release-record": "release.json",
        }
        for record in evidence["canonical_result"]["artifacts"]:
            target = self.results / record["published_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((self.output / generated[record["logical_name"]]).read_bytes())
        args.evidence.write_bytes(evidence_output.read_bytes())

        page = evidence["canonical_result"]["pages"][0]
        self.assertEqual(page["input_pixel_sha256"], "1" * 64)
        self.assertEqual(page["output_pixel_sha256"], "2" * 64)
        self.assertEqual(page["output_image_sha256"], "3" * 64)
        self.assertEqual(prepare(args)["decision"], "reuse")


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
        self.assertIn("--implementation hth-pipeline/hth/preprocess.py", block)
        self.assertIn("--implementation hth-pipeline/hth/analyze_pages.py", block)
        self.assertIn("--implementation hth-pipeline/hth/geometry/registry.py", block)
        self.assertIn("--detector-implementation-root hth-pipeline/hth/geometry", block)
        self.assertNotIn("--implementation hth-pipeline/hth \\", block)
        self.assertNotIn("--implementation hth-pipeline/hth/canonical_build_evidence.py", block)
        declared = re.findall(r"--implementation hth-pipeline/([^ ]+)", block)
        self.assertTrue(declared)
        for relative in declared:
            self.assertTrue((self.root / relative).is_file(), relative)
        self.assertIn("--runtime-contract hth-pipeline/requirements.txt", block)
        self.assertIn("--selection \"$RUNNER_TEMP/preferred-document-detector.json\"", block)
        self.assertIn("--artifact-required", block)
        self.assertIn('--runner-environment "${{ runner.environment }}"', block)
        self.assertIn('--runner-os "${{ runner.os }}"', block)
        self.assertIn('--runner-arch "${{ runner.arch }}"', block)
        self.assertIn('--results-repository "$RESULTS_REPOSITORY"', block)
        self.assertIn('--results-ref "$CBE_RESULTS_REF"', block)

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

    def test_runtime_variant_snapshots_are_published_and_restorable(self) -> None:
        normalize = (self.root / ".github/workflows/normalize.yml").read_text(encoding="utf-8")
        for workflow, scope, plan in (
            (self.core, "hth-preprocess", "cbe_plan"),
            (normalize, "hth-normalization", "normalization_plan"),
        ):
            with self.subTest(scope=scope):
                self.assertIn(f"/cbe-cache/{scope}/", workflow)
                self.assertIn(f"steps.{plan}.outputs.decision == 'restore'", workflow)
                self.assertIn(f"--scope {scope}", workflow)
                self.assertIn("python -m hth.canonical_build_evidence snapshot", workflow)
                self.assertIn("python -m hth.canonical_build_evidence restore", workflow)
                self.assertIn(f"cbe-cache/{scope}", workflow)


if __name__ == "__main__":
    unittest.main()
