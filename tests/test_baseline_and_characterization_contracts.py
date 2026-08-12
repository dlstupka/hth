from hth.domain.result_metrics import baseline_surpassed
from hth.regression.calibration_intelligence import detector_characterization


def _result(mean_iou):
    return {"summary": {"mean_iou": mean_iou, "failure_count": 0}}


def test_baseline_surpassed_uses_final_canonical_avg_iou():
    assert baseline_surpassed(_result(0.9678), _result(0.8846))
    assert not baseline_surpassed(_result(0.80), _result(0.80))
    assert not baseline_surpassed(_result(0.70), _result(0.80))


def test_recent_generators_have_canonical_characterization():
    expected = {
        "convex_hull": "Convex Hull Detector",
        "distance_transform": "Distance Transform Detector",
        "distance_transform_rect": "Distance-Transform Rectangle Proposal",
        "polar_boundary_vote": "Polar Boundary Voting",
        "star_convex": "Star-Convex Boundary Optimization",
    }
    for detector, friendly_name in expected.items():
        item = detector_characterization(detector)
        assert item["friendly_name"] == friendly_name
        assert item["role"] == "Generator"
        assert item["evidence"]
        assert "not yet been registered" not in " ".join(row[2] for row in item["evidence"])
