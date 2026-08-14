# Adaptive Multi-Scale Radial Edge Search

`adaptive_multi_scale_radial_edge` (AMSRE) is a controlled descendant of Multi-Scale Radial Edge Search. It deliberately changes **only angular sampling adaptation** while keeping MSRE's calibrated multiscale evidence fixed.

The detector fixes MSRE's best-known calibrated multiscale evidence (`ddb7623ebb92`): base sigma `1.0`, scale ratio `3.5`, four scales, gradient percentile `96.875`, and the existing fixed radial/scoring controls. It then applies the same two-pass angular idea that made ARE useful: a coarse full-circle pass fits an initial quadrilateral, measures support independently for each side, and allocates a finer second pass only to weak sides.

Generation 1 evaluated 10,000 exhaustive angular-allocation combinations and produced a small but repeatable improvement over MSRE (`0.9767` versus `0.9764` Avg IoU in the calibration report), with the strongest configurations clustered around the calibrated MSRE coarse angular density. Generation 2 therefore keeps the experiment angular-only but expands to exactly **50,000 parameter sets**, densely resolving coarse angle step, refined angle step, weak-side support threshold, side-assignment tolerance, and maximum refined sides around that basin while retaining useful control/anchor values.

`maximum_refined_sides=0` remains the no-refinement control, and `coarse_angle_step_degrees=360/176` remains an exact parent-density anchor.

Verbose debug output includes the fused multiscale gradient, coarse/refined radial points, scale-space view, pass-1 fit, per-side support classification, and pass-2 fit.
