# Line Segment Detector (OpenCV)

`hth/geometry/detector_lsd.py` uses OpenCV's Line Segment Detector (LSD) to
find long near-horizontal and near-vertical segments. Length-weighted outer
percentiles form a conservative axis-aligned document envelope.

The detector is experimental evidence for v0.6.1. It does not replace Contour,
Connected Components, RANSAC, or Hough and does not change candidate selection.
The registry supplies timing, exception isolation, status normalization, and
provenance metadata.

Key diagnostics include total segment count, horizontal and vertical support,
minimum accepted segment length, envelope area, mask score, and support score.
A normal miss returns `no_candidate` after registry normalization.
