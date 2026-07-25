# RANSAC Detector

The HTH RANSAC detector estimates a document envelope from four robustly fitted edge models. It samples the first and last foreground pixel along horizontal and vertical scan lines, groups those observations into left, right, top, and bottom edge families, fits one line per family with deterministic RANSAC, and intersects the fitted lines to form a quadrilateral and bounding box.

## Research stages

The regression banner records the fixed algorithm assumptions and the values swept by each stage:

1. **Boundary sampling** — scan density and the minimum foreground required for a scan line to contribute observations.
2. **Line fitting** — residual threshold, maximum RANSAC trials, and the minimum accepted mean inlier ratio.
3. **Candidate construction** — minimum envelope area and optional bounding-box padding.

The configured exhaustive search contains **1,458 parameter sets**. The baseline is included in that Cartesian space and remains a named production reference.

## Debug artifacts

Winner debug output follows the shared numbered convention:

```text
01-original.jpg
02-input-mask.png
03-boundary-samples.png
04-fitted-edge-models.png
05-ransac-inliers.png
06-candidate-quadrilateral.png
07-overlay.jpg
08-diagnostics.json
```

The images distinguish sampling failures from poor line fits, low inlier support, invalid intersections, and candidate-envelope rejection without exposing implementation noise.

## Configuration

Regression configuration lives at `config/detectors/ransac.json`. The detector is selectable as `ransac` in the detector-regression workflow and is included in automatic `all` smoke runs.
