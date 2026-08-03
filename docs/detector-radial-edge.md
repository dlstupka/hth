# Radial Edge Search detector

The stable method identifier is `radial_edge`. The detector samples gradient magnitude along center-outward rays, selects the strongest credible transition on each ray, and fits a minimum-area quadrilateral to the supported boundary points. It supplies an independent page-boundary generator that does not depend on closed contours, connected components, line extraction, or segmentation.

Configuration: `config/detectors/radial_edge.json`.

The calibration space controls smoothing, ray density, radial search range, gradient threshold, minimum ray support, and scoring weights. Debug artifacts preserve the normalized radial gradient field and the selected edge points with the final quadrilateral.

Regression uses `regression.debug_artifacts: winner`, preserving the original page, detector-specific intermediate images, final overlay, diagnostics, and metadata for the winning parameter set on every Golden Set page, including fully successful runs.
