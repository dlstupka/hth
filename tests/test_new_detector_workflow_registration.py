from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_new_detectors_are_available_to_regression_and_optimizer_workflows():
    regression = (ROOT / ".github/workflows/regress-detector.yml").read_text(encoding="utf-8")
    optimizer = (ROOT / ".github/workflows/execution-optimizer.yml").read_text(encoding="utf-8")
    for detector in ("convex_hull", "distance_transform", "distance_transform_rect", "polar_boundary_vote", "star_convex", "radon_boundary", "text_flow", "whitespace_frame", "joint_rectangle_vote", "learned_page_mask"):
        assert f"          - {detector}\n" in regression
        assert f"          - {detector}\n" in optimizer
        if detector != "learned_page_mask":
            assert f'"{detector}"' in regression
