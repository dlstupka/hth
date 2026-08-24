# Star-Convex Boundary Optimization

`star_convex` is an HTH document/page detector in the authoritative geometry registry.

## Approach

The detector operates by mask boundaries are sampled radially around a center to form a star-convex page hypothesis. The detector measures radial mask support and converts the supported envelope to page geometry.

Like the other registered detectors, it emits the common HTH candidate contract: page geometry, confidence/score, status, parameters, and detector-specific diagnostics. This allows the method to participate in the same Golden Set regression, calibration, optimizer, and comparison machinery as the rest of the detector registry.

## Role in HTH

This detector remains registered as independent evidence and a research/calibration candidate. Registry presence does not imply production approval; production inference resolves the strongest approved authoritative calibration for the requested Golden Set.
