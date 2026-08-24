# Radon Boundary Projection

`radon_boundary` is an HTH document/page detector in the authoritative geometry registry.

## Approach

The detector operates by edge evidence is integrated over a small range of projection angles to locate opposing boundaries. Projection-angle integration provides a transform-domain alternative to direct contour and line methods.

Like the other registered detectors, it emits the common HTH candidate contract: page geometry, confidence/score, status, parameters, and detector-specific diagnostics. This allows the method to participate in the same Golden Set regression, calibration, optimizer, and comparison machinery as the rest of the detector registry.

## Role in HTH

This detector remains registered as independent evidence and a research/calibration candidate. Registry presence does not imply production approval; production inference resolves the strongest approved authoritative calibration for the requested Golden Set.
