#!/usr/bin/env python3
"""Compatibility entry point for the HTH regression framework."""
from __future__ import annotations
from hth.regression.metrics import bbox_iou, edge_errors
from hth.regression.parameter_space import canonical_parameters, exhaustive_parameter_sets, parameter_set_id
from hth.regression.reports import ranking_key
from hth.regression.runner import evaluate_set, find_image, load_pages, main, parse_args, run

__all__=["bbox_iou","edge_errors","canonical_parameters","exhaustive_parameter_sets","parameter_set_id","ranking_key","evaluate_set","find_image","load_pages","parse_args","run","main"]
if __name__=="__main__": raise SystemExit(main())
