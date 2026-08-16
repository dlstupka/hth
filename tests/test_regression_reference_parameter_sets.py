import json
import tempfile
import unittest
from pathlib import Path

from hth.calibration_store import resolve_best_parameter_reference
from hth.regression.parameter_provenance import build_provenance
from hth.regression.parameter_space import parameter_set_id


class RegressionReferenceParameterSetTests(unittest.TestCase):
    def test_historic_best_resolves_across_changed_detector_config_hash(self):
        config = {
            "detector": "example",
            "parameters": {"x": {"values": [1, 2, 3]}},
            "profiles": {"baseline": {"x": 1}},
        }
        best = {"x": 3}
        provenance = build_provenance(
            "example",
            config,
            [{"parameter_set_id": parameter_set_id(best), "parameters": best}],
            strategy="exhaustive",
            complete_cartesian=True,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            record = root / "records" / "old"
            record.mkdir(parents=True)
            (record / "parameter-provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
            index = {
                "entries": [{
                    "detector_id": "example",
                    "golden_set_sha256": "gold",
                    "detector_config_sha256": "OLD-CONFIG",
                    "calibration_id": "old",
                    "calibration_status": "authoritative",
                    "search": {"strategy": "exhaustive", "exhaustive_complete": True},
                    "selection": {
                        "recommended_parameter_set_id": parameter_set_id(best),
                        "best_avg_iou": 0.9,
                        "minimum_iou": 0.8,
                        "stddev_iou": 0.1,
                        "failure_count": 0,
                    },
                    "parameter_provenance_path": "records/old/parameter-provenance.json",
                    "record_path": "records/old",
                    "build": {"github_run_number": "371", "run_url": "https://example/run/371"},
                }]
            }
            index_path = root / "calibration-index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            resolved = resolve_best_parameter_reference(
                index_path,
                detector="example",
                golden_set_sha256="gold",
            )
        self.assertEqual(resolved["parameters"], best)
        self.assertEqual(resolved["historic_build_number"], "371")

    def test_shell_injects_historic_best_for_all_strategies(self):
        text = Path("tools/run-detector-regressions.sh").read_text(encoding="utf-8")
        self.assertIn("resolve-best-parameter", text)
        self.assertIn('--historic-best "$historic_best_file"', text)
        function_start = text.index("run_detector_config()")
        best_pos = text.index("resolve-best-parameter", function_start)
        strategy_pos = text.index('if [[ -n "${effective_limit:-}" ]]', function_start)
        self.assertLess(best_pos, strategy_pos)


if __name__ == "__main__":
    unittest.main()
