# Contour + Projection detector

The Contour + Projection detector uses contour-derived quadrilateral hypotheses and ranks them with independent projection-profile evidence from the perspective-normalized candidate interior. Its stable method identifier is `contour_projection`.

For every plausible contour quadrilateral, the detector:

1. orders the four corners and perspective-warps the candidate;
2. removes a configurable margin to reduce border influence;
3. adaptively thresholds the interior to isolate dark page content;
4. computes horizontal row-density and vertical coverage evidence; and
5. combines projection evidence with area, rectangularity, and angle consistency.

Projection evidence is a validator and ranking signal, not OCR. It is intended to favor page-shaped candidates whose normalized interior contains the repeated horizontal structure expected from handwritten or printed text.

`config/detectors/contour_projection.json` contains the reusable baseline and calibration space. Regression debug artifacts include contour hypotheses, the warped candidate, thresholded projection input, the horizontal projection visualization, the selected quadrilateral, final overlay, and diagnostics JSON.
