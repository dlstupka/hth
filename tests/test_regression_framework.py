from __future__ import annotations
import csv, json
from pathlib import Path
from tempfile import TemporaryDirectory
from hth.regression.io import create_run_directory
from hth.regression.reports import write_rankings, write_raw_results

def sample():
    return [{"run_id":"run-test","rank":1,"parameter_set_id":"abc","profile":"baseline","parameters":{"x":1},"summary":{"mean_iou":.9,"minimum_iou":.8,"mean_edge_error_px":2.5,"failure_count":0,"elapsed_ms_total":10},"pages":[{"global_ordinal":5,"label":"fs_0005","layout_type":"spread","status":"ok","iou":.9,"edge_errors":{"left":1,"top":2,"right":3,"bottom":4},"edge_error_mean_px":2.5,"edge_error_maximum_px":4,"elapsed_ms":10,"approved_bbox":[0,0,10,10],"predicted_bbox":[1,2,13,14]}]}]

def test_run_directory_schema():
    with TemporaryDirectory() as td:
        rid,path=create_run_directory(Path(td),"grabcut","run-test")
        assert rid=="run-test"
        assert (path/"raw").is_dir() and (path/"reports").is_dir() and (path/"logs").is_dir()

def test_reports_preserve_per_page_observations():
    with TemporaryDirectory() as td:
        root=Path(td); write_raw_results(root/"results.csv",sample()); write_rankings(root/"rankings.csv",sample())
        rows=list(csv.DictReader((root/"results.csv").open()))
        assert len(rows)==1 and rows[0]["global_ordinal"]=="5" and rows[0]["left_error_px"]=="1"
        assert json.loads(rows[0]["parameters_json"])=={"x":1}
