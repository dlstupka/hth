# Contour + GrabCut detector

The Contour + GrabCut detector uses contour-derived quadrilateral geometry as its generator and OpenCV GrabCut segmentation as independent pixel-level validation. Its stable method identifier is `contour_grabcut`.

The selected geometry remains the contour quadrilateral. GrabCut contributes a second candidate and an overlap score; configurations may reject hypotheses below `minimum_agreement_iou` or permit contour fallback when GrabCut cannot produce a usable region.

`config/detectors/contour_grabcut.json` defines the baseline and calibration space. Winner debug artifacts show the contour candidate, GrabCut candidate, agreement overlay, selected quadrilateral, final overlay, and diagnostics.
