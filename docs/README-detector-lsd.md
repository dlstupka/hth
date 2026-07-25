# Line Segment Detector (OpenCV)

`hth/geometry/detector_lsd.py` uses OpenCV's Line Segment Detector (LSD) to
find long near-horizontal and near-vertical segments. Length-weighted outer
percentiles form a conservative axis-aligned document envelope.

The detector participates in the detector-regression framework through
`config/detectors/lsd.json`. Its black-box calibration space covers OpenCV
refinement mode and image scale, axis-segment length and angle classification,
outer-envelope percentiles, minimum envelope area, and final bounding-box
padding. The baseline profile preserves the original detector behavior.

The exhaustive search contains 2,187 parameter sets before the baseline is
deduplicated. Smoke regressions use the workflow's normal parameter-set limit.

The registry supplies timing, exception isolation, status normalization, and
provenance metadata. Key diagnostics include the effective parameter set, total
segment count, horizontal and vertical support, minimum accepted segment length,
envelope area, mask score, and support score. A normal miss returns
`no_candidate` after registry normalization.
