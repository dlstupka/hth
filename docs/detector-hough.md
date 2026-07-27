# Hough Lines Detector (OpenCV)

`hth/geometry/detector_hough.py` uses OpenCV's probabilistic Hough transform
(`HoughLinesP`) to extract long line segments from masked Canny edges. Long
near-horizontal and near-vertical lines are classified into edge families, and
length-weighted outer percentiles form an axis-aligned document envelope.

The detector participates in detector regression through
`config/detectors/hough.json`. Its black-box search covers the Canny threshold,
Hough vote threshold, minimum line length, maximum line gap, axis-angle
classification tolerance, outer-envelope percentile, and final bounding-box
padding. The minimum accepted envelope area remains fixed in the baseline
profile so the exhaustive space remains tractable.

The exhaustive search contains 2,187 parameter sets before baseline
deduplication. Smoke regressions use the workflow's normal parameter-set limit.

The registry supplies timing, exception isolation, status normalization, and
provenance metadata. Diagnostics include the effective parameter set, total
Hough line count, horizontal and vertical support, effective pixel thresholds,
envelope area, mask score, and support score. A normal miss returns
`no_candidate` after registry normalization.
