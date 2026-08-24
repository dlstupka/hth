import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import hth.write_regression_summary as report
from hth.regression.parameter_provenance import (
    build_provenance,
    parameter_identity_sha256,
)


class ParameterBuildHistoryIndexTests(unittest.TestCase):
    def _write_record(self, root, detector, build, status, parameters):
        record = root / "records" / f"run-{build}"
        record.mkdir(parents=True)
        config = {
            "parameter_schema_version": "1",
            "parameters": {
                "threshold": {"values": [0.1, 0.2, 0.3]},
                "radius": {"values": [1, 2]},
            },
            "profiles": {"baseline": {"threshold": 0.1, "radius": 1}},
        }
        provenance = build_provenance(
            detector,
            config,
            [],
            strategy="exhaustive",
            complete_cartesian=True,
        )
        (record / "parameter-provenance.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )
        (record / "reports").mkdir()
        (record / "reports" / "summary.json").write_text(
            json.dumps({"winner": {"parameter_set_id": report._short(report.parameter_identity_sha256(detector, parameters, schema_version="1"), 12), "parameters": parameters}}),
            encoding="utf-8",
        )
        return {
            "detector_id": detector,
            "calibration_status": status,
            "created_at_utc": f"2026-08-{int(build) % 28 + 1:02d}T00:00:00Z",
            "parameter_provenance_path": f"records/run-{build}/parameter-provenance.json",
            "record_path": f"records/run-{build}",
            "build": {
                "github_run_number": str(build),
                "run_url": f"https://example.invalid/{build}",
            },
        }, config

    def test_index_reads_each_provenance_once_and_reuses_it_for_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detector = "demo"
            target = {"threshold": 0.2, "radius": 2}
            entries = []
            config = None
            for build, status in ((100, "authoritative"), (300, "authoritative"), (400, "provisional")):
                entry, config = self._write_record(root, detector, build, status, target)
                entries.append(entry)
            index_path = root / "calibration-index.json"
            index_path.write_text(json.dumps({"entries": entries}), encoding="utf-8")
            full_sha = parameter_identity_sha256(detector, target, schema_version="1")

            original = report._read_json
            provenance_reads = []

            def counted(path):
                path = Path(path)
                if path.name == "parameter-provenance.json":
                    provenance_reads.append(path)
                return original(path)

            with patch.object(report, "_read_json", side_effect=counted):
                history = report._build_parameter_build_index(index_path)

            # Provisional/smoke build 400 is rejected before its provenance file
            # is even opened, so only the two full records are parsed.
            self.assertEqual(len(provenance_reads), 2)

            first = report._known_builds_for_parameter(
                detector=detector,
                full_sha=full_sha,
                parameters=target,
                info={},
                run_url="",
                parameter_build_index=history,
            )
            second = report._known_builds_for_parameter(
                detector=detector,
                full_sha=full_sha,
                parameters=target,
                info={},
                run_url="",
                parameter_build_index=history,
            )
            self.assertEqual(first, second)
            self.assertEqual([row[0] for row in first], ["300", "100"])
            self.assertNotIn("400", [row[0] for row in first])

    def test_canonical_indexes_directory_resolves_repository_relative_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detector = "demo"
            target = {"threshold": 0.2, "radius": 2}
            entry, _ = self._write_record(root, detector, 300, "authoritative", target)
            index_path = root / "indexes" / "calibration-index.json"
            index_path.parent.mkdir()
            index_path.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")

            history = report._build_parameter_build_index(index_path)
            full_sha = parameter_identity_sha256(detector, target, schema_version="1")
            builds = report._known_builds_for_parameter(
                detector=detector, full_sha=full_sha, parameters=target, info={}, run_url="",
                parameter_build_index=history,
            )

            self.assertEqual([row[0] for row in builds], ["300"])

    def test_current_build_is_sorted_with_historical_builds_newest_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detector = "demo"
            target = {"threshold": 0.2, "radius": 2}
            entry, _ = self._write_record(root, detector, 300, "authoritative", target)
            index_path = root / "calibration-index.json"
            index_path.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
            history = report._build_parameter_build_index(index_path)
            full_sha = parameter_identity_sha256(detector, target, schema_version="1")

            builds = report._known_builds_for_parameter(
                detector=detector,
                full_sha=full_sha,
                parameters=target,
                info={
                    "github_run_number": "450",
                    "github_run_url": "https://example.invalid/450",
                    "started_at_utc": "2026-08-17T00:00:00Z",
                },
                run_url="",
                parameter_build_index=history,
            )
            self.assertEqual([row[0] for row in builds], ["450", "300"])

    def test_equivalent_build_history_preserves_distinct_exact_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            detector = "demo"
            old = {"threshold": 0.1, "radius": 2}
            new = {"threshold": 0.3, "radius": 2}
            entry, _ = self._write_record(root, detector, 523, "authoritative", old)
            index_path = root / "calibration-index.json"
            index_path.write_text(json.dumps({"entries": [entry]}), encoding="utf-8")
            history = report._build_parameter_build_index(index_path)
            family_config = {"equivalence_parameters": ["threshold"]}
            family_id = report.parameter_set_equivalence_family_id(new, family_config)
            builds = report._known_builds_for_family(
                detector=detector, family_id=family_id, equivalence_parameters=["threshold"],
                current_parameter_set_id="currentexact", info={}, run_url="", parameter_build_index=history,
            )
            self.assertEqual([row[0] for row in builds], ["523"])
            self.assertEqual(builds[0][3], family_id)
            self.assertNotEqual(builds[0][4], "currentexact")


if __name__ == "__main__":
    unittest.main()
