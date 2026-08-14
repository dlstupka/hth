# Adaptive Multi-Scale Radial Edge Search

`adaptive_multi_scale_radial_edge` (AMSRE) is a controlled descendant of Multi-Scale Radial Edge Search. Generation 1 deliberately changes **only angular sampling adaptation**.

The detector fixes MSRE's best-known calibrated multiscale evidence (`ddb7623ebb92`): base sigma `1.0`, scale ratio `3.5`, four scales, gradient percentile `96.875`, and the existing fixed radial/scoring controls. It then applies the same two-pass angular idea that made ARE useful: a coarse full-circle pass fits an initial quadrilateral, measures support independently for each side, and allocates a finer second pass only to weak sides.

The initial exhaustive calibration contains exactly **10,000 parameter sets** across five angular-allocation controls: coarse angle step, refined angle step, weak-side support threshold, side-assignment tolerance, and maximum refined sides. `maximum_refined_sides=0` is retained as a no-refinement control, and `coarse_angle_step_degrees=360/176` reproduces the angular density of the calibrated MSRE parent.

Verbose debug output includes the fused multiscale gradient, coarse/refined radial points, scale-space view, pass-1 fit, per-side support classification, and pass-2 fit.
