import json
from hth.detector_catalog import configured_detectors
def test_optional_detector_is_excluded_from_automatic_catalog(tmp_path):
    (tmp_path/"ordinary.json").write_text(json.dumps({"detector":"ordinary"}),encoding="utf-8")
    (tmp_path/"optional.json").write_text(json.dumps({"detector":"optional","automatic":False}),encoding="utf-8")
    assert configured_detectors(tmp_path)==["optional","ordinary"]
    assert configured_detectors(tmp_path,automatic_only=True)==["ordinary"]
