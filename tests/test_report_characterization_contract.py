from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_regression_summary_uses_canonical_detector_characterization():
    text = (ROOT / "hth" / "write_regression_summary.py").read_text(encoding="utf-8")
    assert "from hth.regression.calibration_intelligence import detector_characterization" in text
    assert "DETECTOR_CHARACTERIZATION" not in text

def test_baseline_surpassed_rendering_uses_final_winner_and_baseline():
    text = (ROOT / "hth" / "write_regression_summary.py").read_text(encoding="utf-8")
    assert "baseline_surpassed(winner, baseline)" in text
    assert "progress.get('baseline_surpassed')" not in text
