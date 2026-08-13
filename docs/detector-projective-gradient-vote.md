# Projective Gradient Vote

`projective_gradient_vote` is a non-axis-aligned variant of Gradient Boundary Voting intended for page captures with perspective, skew, or converging opposite sides. Instead of accumulating only horizontal and vertical projection profiles, it detects long line segments, weights them by Sobel-gradient support, identifies two near-orthogonal orientation families, chooses opposing members of each family, and intersects the selected lines to form a projective quadrilateral.

Configuration: `config/detectors/projective_gradient_vote.json`.

The detector rejects candidates when the two orientation families are insufficient, line intersections overshoot the image by an implausible amount, page area is implausible, or any selected side lacks enough gradient support. Calibration focuses on the evidence-resolution and family-selection controls rather than exposing implementation-only constants.

Winner debug artifacts include the gradient field and the selected projective line votes. Verbose debug adds all long line-segment hypotheses used by the orientation-family search.
