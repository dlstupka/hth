# GrabCut + Contour detector

The GrabCut + Contour detector uses OpenCV GrabCut segmentation as the primary generator and Contour Quadrilateral as independent geometric validation. Its stable method identifier is `grabcut_contour`.

This detector is intentionally directional and complementary to `contour_grabcut`: successful output geometry comes from GrabCut, while contour evidence validates the result or provides an optional fallback.

`config/detectors/grabcut_contour.json` defines the baseline and calibration space. Winner debug artifacts show the GrabCut candidate, contour candidate, agreement overlay, selected quadrilateral, final overlay, and diagnostics.
