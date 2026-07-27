# GrabCut Detector (OpenCV)

`hth/geometry/detector_grabcut.py` uses the shared HTH document mask as a
probable foreground/background seed for OpenCV GrabCut. Border pixels are fixed
as background, an eroded mask core is fixed as foreground, and the refined
largest region becomes the candidate document geometry.

The detector is experimental evidence for v0.6.1. It does not replace existing
detectors or alter candidate selection. The registry supplies timing, exception
isolation, status normalization, and provenance metadata.

Key diagnostics include initial and refined foreground fractions, contour and
bounding-box area fractions, rectangularity, and scores against both the
refined and shared masks. Empty or implausibly small regions are normal
`no_candidate` results.
