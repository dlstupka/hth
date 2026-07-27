# Edge-Contour Hybrid Detector

The HTH Edge-Contour Hybrid generates document quadrilateral hypotheses from the analysis mask and verifies those hypotheses against line segments detected independently in the source image.

The stable method identifier is `edge_contour`.

## Algorithm

1. Normalize and optionally close the supplied document mask.
2. Generate external-contour hypotheses and, when enabled, a merged convex-hull hypothesis.
3. Sweep polygon-approximation epsilon values and retain plausible convex quadrilaterals.
4. Run OpenCV's Line Segment Detector once against the source image and discard segments below the configured minimum length.
5. Measure how much of each proposed quadrilateral boundary is supported by the retained line-segment evidence.
6. Reject candidates below the minimum edge-support threshold.
7. Rank surviving candidates using area, rectangularity, angle consistency, and independent edge support.

The contour and line-segment evidence remain independent: the mask proposes geometry, while the source image verifies it. A contour-only quadrilateral without sufficient line support is reported as `no_candidate`.

## Configuration boundary

`config/detectors/edge_contour.json` contains reusable detector defaults and calibration ranges. It does not contain collection identity, page ordinals, or document-specific exceptions.

## Regression

```bash
python -m hth.regress_detector \
  --detector-config config/detectors/edge_contour.json \
  --golden-set config/golden_set.json \
  --image-root /path/to/preprocessed/images \
  --output regression-output
```
