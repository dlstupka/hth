from __future__ import annotations

from hth.regress_detector import bbox_iou, edge_errors, exhaustive_parameter_sets, parameter_set_id


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
