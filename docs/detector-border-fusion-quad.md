# Border Fusion Quad

`border_fusion_quad` is a side-level fusion generator. It runs the reviewed baseline forms of Radial Edge Search, Polar Boundary Voting, and Gradient Boundary Voting, converts each successful child quadrilateral into top/right/bottom/left line hypotheses, and searches mixed-source side combinations for the strongest geometrically valid page quadrilateral.

Configuration: `config/detectors/border_fusion_quad.json`.

This differs from detector-level consensus: the returned quadrilateral does not have to come from any one child detector. One child may supply the top edge while another supplies a side that it sees more clearly. A fused proposal must use at least two distinct child sources and is independently validated against Sobel gradient support on all four selected sides. Child detectors use their baseline parameters so calibration of the fusion layer remains isolated from calibration of each child algorithm.

## Calibration generations

The initial exhaustive 244-set calibration showed a broad equivalence plateau at the top, but it also identified two strong active controls: `gradient_percentile` and `minimum_side_gradient_support`. The best result occurred at the low edge of the original gradient-percentile range and near the low edge of side-support filtering. `minimum_child_confidence` was effectively dormant, and non-zero `bbox_padding_fraction` was consistently worse than zero.

Generation 2 therefore declares exactly 50,000 Cartesian parameter sets and spends that budget on information the first run could not answer. Gradient percentile is extended well below the original 74 boundary, side-gradient support is sampled more densely below and around the original 0.08/0.16 best region, and minimum area retains both the prior winner and baseline anchors. The fusion scorer itself is also calibrated for the first time through `gradient_weight`, `source_confidence_weight`, and a two-state `source_diversity_weight` probe, allowing the regression to test whether page 5's weak result is caused by over-trusting local edge support, child-detector confidence, or the extra preference for three-source mixtures. The area-score weight remains fixed so the second generation stays interpretable rather than turning into an unconstrained weight search.

`minimum_child_confidence` and `bbox_padding_fraction` are intentionally omitted from the Generation-2 Cartesian grid and remain at their baseline values of zero. The original baseline values are still preserved in the named `baseline` profile, so historical comparisons remain intact.

Winner debug artifacts show the independent child quadrilaterals, the fusion gradient field, and the selected fused quadrilateral. Verbose debug colors the four selected sides by their source detector.
