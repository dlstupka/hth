from __future__ import annotations

from hth.regress_detector import bbox_iou, edge_errors, exhaustive_parameter_sets, parameter_set_id
from hth.regression.adapters.components import pre_regression_report_sections
from hth.regression.runner import DETECTORS


def test_bbox_iou() -> None:
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]) == 1 / 3


def test_edge_errors_are_absolute() -> None:
    result = edge_errors([8, 12, 102, 95], [10, 10, 100, 100])
    assert result == {"left": 2, "top": 2, "right": 2, "bottom": 5, "mean": 2.75, "maximum": 5}


def test_exhaustive_parameter_sets_is_cartesian() -> None:
    config = {
        "parameters": {
            "a": {"values": [1, 2]},
            "b": {"values": ["x", "y", "z"]},
        }
    }
    results = exhaustive_parameter_sets(config)
    assert len(results) == 6
    assert {tuple(sorted(item.items())) for item in results} == {
        (("a", 1), ("b", "x")),
        (("a", 1), ("b", "y")),
        (("a", 1), ("b", "z")),
        (("a", 2), ("b", "x")),
        (("a", 2), ("b", "y")),
        (("a", 2), ("b", "z")),
    }


def test_parameter_set_id_is_order_independent() -> None:
    assert parameter_set_id({"a": 1, "b": 2}) == parameter_set_id({"b": 2, "a": 1})


def test_initial_black_box_detectors_are_registered() -> None:
    assert set(DETECTORS) >= {"components", "grabcut", "contour", "contour_quad", "contour_components", "contour_grabcut", "contour_projection", "consensus_quad", "hough", "lsd"}
    assert all(
        callable(DETECTORS[name])
        for name in ("components", "grabcut", "contour", "contour_quad", "hough", "lsd")
    )


def test_components_pre_regression_sections_are_ordered_and_research_focused() -> None:
    config = {
        "parameters": {
            "morphology_close_fraction": {"values": [0.004, 0.008]},
            "morphology_dilate_fraction": {"values": [0.01, 0.02, 0.03]},
            "minimum_component_area_fraction": {"values": [0.001]},
            "minimum_component_area_px": {"values": [25]},
            "merge_area_ratio": {"values": [0.02]},
            "merge_gap_fraction": {"values": [0.035]},
            "minimum_bbox_area_fraction": {"values": [0.12]},
            "minimum_selected_area_fraction": {"values": [0.04]},
            "bbox_padding_fraction": {"values": [0.0]},
        }
    }
    sections = pre_regression_report_sections(config)
    assert [section["title"] for section in sections] == [
        "Morphology Algorithm",
        "Morphology Search Space",
        "Connected Components Detector Configuration",
    ]
    algorithm = dict(sections[0]["rows"])
    search_space = dict(sections[1]["rows"])
    assert search_space["Morphology variants"] == 6
    assert algorithm["Operation sequence"] == "closing -> dilation"


def test_effect_strategy_falls_back_to_next_larger_domain() -> None:
    from hth.regression.runner import _resolve_effect_strategy
    metadata = {"domain_space": {
        "critical": {"parameter_set_count": 0},
        "important_plus": {"parameter_set_count": 0},
        "moderate_plus": {"parameter_set_count": 3},
    }}
    actual, domain, reason = _resolve_effect_strategy("critical", metadata)
    assert actual == "moderate+"
    assert domain["parameter_set_count"] == 3
    assert "fell back" in reason


def test_effect_strategy_defaults_to_exhaustive_without_metadata() -> None:
    from hth.regression.runner import _resolve_effect_strategy
    actual, domain, reason = _resolve_effect_strategy("non-dormant", None)
    assert actual == "exhaustive"
    assert domain is None
    assert "exhaustive" in reason
