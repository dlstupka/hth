# Convex Hull Detector

`convex_hull` is an HTH document/page detector in the authoritative geometry registry.

## Approach

The detector operates by foreground fragments are morphologically joined and summarized by a convex hull. Convex hull geometry and OpenCV morphology provide a broad page-envelope hypothesis.

Like the other registered detectors, it emits the common HTH candidate contract: page geometry, confidence/score, status, parameters, and detector-specific diagnostics. This allows the method to participate in the same Golden Set regression, calibration, optimizer, and comparison machinery as the rest of the detector registry.

## Role in HTH

This detector remains registered as independent evidence and a research/calibration candidate. Registry presence does not imply production approval; production inference resolves the strongest approved authoritative calibration for the requested Golden Set.
