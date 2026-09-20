from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from hth.canonical_build_evidence import finalize, prepare
from hth.resource_lifecycle import build_report


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


class ResourceLifecycleTests(unittest.TestCase):
    def test_builds_record_utilization_and_cleanup_is_conservative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = root / "pipeline"
            source = root / "source"
            results = root / "results"
            output = root / "output"
            (pipeline / "hth").mkdir(parents=True)
            source.mkdir()
            (pipeline / "hth/stage.py").write_text("VERSION = 1\n", encoding="utf-8")
            (pipeline / "hth/mirrors.py").write_text(
                'MIRROR = "HTH-MIRROR-CURRENT-MODEL"\n', encoding="utf-8"
            )
            (pipeline / "config.json").write_text('{"threshold": 1}\n', encoding="utf-8")
            (pipeline / "requirements.txt").write_text("Pillow\n", encoding="utf-8")
            (source / "input.json").write_text("{}\n", encoding="utf-8")
            store = results / "normalization/tonal/canonical-build-evidence.json"

            def build(release: str, manifest_sha: str) -> dict:
                plan = root / f"{release}-plan.json"
                args = argparse.Namespace(
                    scope="hth-tonal-assessment",
                    contract_version="1",
                    policy="auto",
                    mode="diagnostic-evidence",
                    image_limit=0,
                    operation=[],
                    repository_root=pipeline,
                    source_root=source,
                    source_repository="owner/releases",
                    source_release=release,
                    source_manifest_sha256=manifest_sha,
                    source_commit=manifest_sha,
                    config=[pipeline / "config.json"],
                    implementation=[pipeline / "hth/stage.py"],
                    detector_implementation_root=None,
                    runtime_contract=[pipeline / "requirements.txt"],
                    runtime_package=[],
                    runtime_component=[],
                    selection=None,
                    results_root=results,
                    evidence=store,
                    plan=plan,
                    artifact_required=False,
                    pipeline_repository="owner/pipeline",
                    pipeline_commit="c" * 40,
                    workflow_run_id="123",
                    runner_name="runner",
                    runner_environment="self-hosted",
                    runner_os="Linux",
                    runner_arch="X64",
                    github_output="",
                    github_summary="",
                    results_repository="owner/results",
                    results_ref="main",
                )
                prepared = prepare(args)
                self.assertEqual(prepared["decision"], "execute")
                write_json(output / "assessment.json", {
                    "assessment_identity": manifest_sha,
                    "pages": [{"global_ordinal": 1, "decision": "preserve"}],
                })
                incoming = root / f"{release}-store.json"
                evidence = finalize(argparse.Namespace(
                    plan=plan,
                    output_root=output,
                    evidence_store=store,
                    evidence_output=incoming,
                    github_output="",
                    github_summary="",
                ))
                store.parent.mkdir(parents=True, exist_ok=True)
                store.write_bytes(incoming.read_bytes())
                (results / "normalization/tonal/assessment.json").write_bytes(
                    (output / "assessment.json").read_bytes()
                )
                return evidence

            first = build("HTH-TONAL-PREVIOUS", "1" * 64)
            second = build("HTH-TONAL-CURRENT", "2" * 64)
            reuse_args = argparse.Namespace(
                scope="hth-tonal-assessment",
                contract_version="1",
                policy="auto",
                mode="diagnostic-evidence",
                image_limit=0,
                operation=[],
                repository_root=pipeline,
                source_root=source,
                source_repository="owner/releases",
                source_release="HTH-TONAL-CURRENT",
                source_manifest_sha256="2" * 64,
                source_commit="2" * 64,
                config=[pipeline / "config.json"],
                implementation=[pipeline / "hth/stage.py"],
                detector_implementation_root=None,
                runtime_contract=[pipeline / "requirements.txt"],
                runtime_package=[],
                runtime_component=[],
                selection=None,
                results_root=results,
                evidence=store,
                plan=root / "reuse-plan.json",
                artifact_required=False,
                pipeline_repository="owner/pipeline",
                pipeline_commit="c" * 40,
                workflow_run_id="124",
                runner_name="runner",
                runner_environment="self-hosted",
                runner_os="Linux",
                runner_arch="X64",
                github_output="",
                github_summary="",
                results_repository="owner/results",
                results_ref="main",
            )
            reused = prepare(reuse_args)
            self.assertEqual(reused["decision"], "reuse")
            self.assertEqual(reused["resource_utilization"]["canonical_evidence_cache"]["lookup"], "hit")
            self.assertEqual(reused["resource_utilization"]["canonical_evidence_cache"]["action"], "reused")
            inventory = root / "releases.json"
            write_json(inventory, [
                {"repository": "owner/releases", "tagName": "HTH-TONAL-PREVIOUS", "isLatest": True},
                {"repository": "owner/releases", "tagName": "HTH-TONAL-CURRENT", "isLatest": False},
                {"repository": "owner/releases", "tagName": "HTH-TONAL-UNREFERENCED", "isLatest": False},
                {"repository": "owner/releases", "tagName": "ORPHAN"},
                {"repository": "dlstupka/hth-mirror", "tagName": "HTH-MIRROR-CURRENT-MODEL"},
                {"repository": "dlstupka/hth-mirror", "tagName": "HTH-MIRROR-OLD-MODEL"},
                {"repository": "owner/collection-cache", "tagName": "HTH-EVIDENCE-DETECTOR-ABC"},
            ])

            report = build_report(results, release_inventory=inventory, repository_root=pipeline)
            cache = {item["identity"]: item for item in report["cache_elements"]}
            self.assertTrue(cache[first["effective_build_identity"]]["dirty"])
            self.assertEqual(cache[first["effective_build_identity"]]["lineage"], "previous")
            self.assertEqual(cache[first["effective_build_identity"]]["integrity"], "good")
            self.assertFalse(cache[first["effective_build_identity"]]["cleanup_eligible"])
            self.assertFalse(cache[second["effective_build_identity"]]["dirty"])
            self.assertEqual(cache[second["effective_build_identity"]]["lineage"], "current")
            releases = {item["release"]: item for item in report["release_elements"]}
            self.assertEqual(releases["HTH-TONAL-PREVIOUS"]["lineage"], "previous")
            self.assertEqual(releases["HTH-TONAL-PREVIOUS"]["publication"], "latest")
            self.assertEqual(releases["HTH-TONAL-PREVIOUS"]["integrity"], "good")
            self.assertTrue(releases["HTH-TONAL-PREVIOUS"]["dirty"])
            self.assertFalse(releases["HTH-TONAL-PREVIOUS"]["cleanup_eligible"])
            self.assertEqual(releases["HTH-TONAL-CURRENT"]["lineage"], "current")
            self.assertFalse(releases["HTH-TONAL-CURRENT"]["dirty"])
            self.assertEqual(releases["HTH-TONAL-UNREFERENCED"]["lineage"], "superseded")
            self.assertTrue(releases["HTH-TONAL-UNREFERENCED"]["cleanup_eligible"])
            self.assertTrue(releases["ORPHAN"]["dirty"])
            self.assertTrue(releases["ORPHAN"]["cleanup_eligible"])
            self.assertEqual(releases["HTH-MIRROR-CURRENT-MODEL"]["lineage"], "current")
            self.assertTrue(releases["HTH-MIRROR-CURRENT-MODEL"]["dirty"])
            self.assertFalse(releases["HTH-MIRROR-CURRENT-MODEL"]["cleanup_eligible"])
            self.assertEqual(releases["HTH-MIRROR-OLD-MODEL"]["lineage"], "unreferenced")
            self.assertTrue(releases["HTH-MIRROR-OLD-MODEL"]["cleanup_eligible"])
            evidence_cache = releases["HTH-EVIDENCE-DETECTOR-ABC"]
            self.assertEqual(evidence_cache["release_kind"], "learned-evidence-cache")
            self.assertEqual(evidence_cache["lineage"], "unreferenced")
            self.assertFalse(evidence_cache["cleanup_authority"])
            self.assertFalse(evidence_cache["cleanup_eligible"])
            self.assertEqual(
                evidence_cache["cleanup_reason"],
                "learned-evidence-utilization-ledger-required",
            )
            self.assertEqual(
                second["resource_utilization"]["canonical_evidence_cache"]["lookup"],
                "miss",
            )


if __name__ == "__main__":
    unittest.main()
