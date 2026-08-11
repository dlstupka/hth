import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hth.historical_rerank import rerank_run


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class HistoricalRerankTests(unittest.TestCase):
    def test_rerank_uses_raw_page_evidence_and_changes_stale_winner(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "run"
            (run / "raw").mkdir(parents=True)
            (run / "reports").mkdir()
            fields = [
                "run_id","parameter_set_id","profile","rank","completion_index",
                "completion_elapsed_seconds","search_fraction","global_ordinal","label",
                "layout_type","status","iou","left_error_px","top_error_px",
                "right_error_px","bottom_error_px","edge_error_mean_px",
                "edge_error_maximum_px","elapsed_ms","approved_bbox_json",
                "predicted_bbox_json","parameters_json","error_type","error_message",
            ]
            rows = []
            # stale winner: one gorgeous page, four failures -> canonical 0.19276
            for ordinal in range(1, 6):
                rows.append({
                    "run_id":"run-1","parameter_set_id":"stale","profile":"",
                    "rank":"","completion_index":"","completion_elapsed_seconds":"",
                    "search_fraction":"","global_ordinal":ordinal,"label":str(ordinal),
                    "layout_type":"single","status":"ok" if ordinal == 1 else "no_candidate",
                    "iou":"0.9638" if ordinal == 1 else "0","left_error_px":"",
                    "top_error_px":"","right_error_px":"","bottom_error_px":"",
                    "edge_error_mean_px":"1" if ordinal == 1 else "",
                    "edge_error_maximum_px":"1" if ordinal == 1 else "",
                    "elapsed_ms":"1","approved_bbox_json":"[0,0,1,1]",
                    "predicted_bbox_json":"[0,0,1,1]" if ordinal == 1 else "null",
                    "parameters_json":'{"x":1}',"error_type":"","error_message":"",
                })
            # proper winner: all five pages at 0.9
            for ordinal in range(1, 6):
                row = dict(rows[0])
                row.update({
                    "parameter_set_id":"good","global_ordinal":ordinal,"label":str(ordinal),
                    "status":"ok","iou":"0.9","parameters_json":'{"x":2}',
                    "predicted_bbox_json":"[0,0,1,1]",
                })
                rows.append(row)
            with (run/"raw/results.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            _write_json(run/"manifest.json", {"run_id":"run-1","detector":"gradient_vote","strategy":"exhaustive","status":"complete"})
            _write_json(run/"RUN-INFO.json", {"run_id":"run-1","possible_parameter_sets":2})
            _write_json(run/"parameters.json", {})
            _write_json(run/"reports/summary.json", {
                "run_id":"run-1","detector":"gradient_vote","strategy":"exhaustive",
                "page_ordinals":[1,2,3,4,5],"parameter_space":{"possible_parameter_sets":2},
                "winner":{"parameter_set_id":"stale"},"progress":{}
            })
            _write_json(run/"reports/calibration-intelligence.json", {
                "available":True,"detector":"gradient_vote",
                "calibration_identity":{"calibration_run_id":"run-1"},
                "regression_metadata":{},"calibration_status":"authoritative"
            })

            results = root/"results"
            results.mkdir()
            _write_json(results/"calibration-index.json", {"entries":[{
                "calibration_id":"run-1","calibration_status":"authoritative",
                "source_document_id":"source","build":{"github_run_number":"300"}
            }]})

            fake_entry = {
                "calibration_id":"run-1","compatibility_key":"key",
                "calibration_status":"authoritative","source_document_id":"source",
                "golden_set_id":"gold","detector_id":"gradient_vote",
                "created_at_utc":"2026-08-10T00:00:00Z"
            }
            with patch("hth.historical_rerank.publish_run", return_value=fake_entry), \
                 patch("hth.historical_rerank.update_index"):
                result = rerank_run(run, results)

            self.assertEqual(result["winner"], "good")
            self.assertTrue(result["winner_changed"])
            summary = json.loads((run/"reports/summary.json").read_text())
            self.assertEqual(summary["winner"]["parameter_set_id"], "good")
            self.assertAlmostEqual(summary["winner"]["summary"]["mean_iou"], 0.9)
            self.assertEqual(summary["historical_rerank"]["previous_winner_parameter_set_id"], "stale")


if __name__ == "__main__":
    unittest.main()
