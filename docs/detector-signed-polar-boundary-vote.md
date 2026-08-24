# Signed Polar Boundary Voting

`signed_polar_boundary_vote` is an HTH document/page detector in the authoritative geometry registry.

## Approach

The detector operates by radial boundary votes retain the expected bright-inside transition polarity. Signed radial gradients reject transitions with the wrong page/background polarity.

Like the other registered detectors, it emits the common HTH candidate contract: page geometry, confidence/score, status, parameters, and detector-specific diagnostics. This allows the method to participate in the same Golden Set regression, calibration, optimizer, and comparison machinery as the rest of the detector registry.

## Role in HTH

This detector remains registered as independent evidence and a research/calibration candidate. Registry presence does not imply production approval; production inference resolves the strongest approved authoritative calibration for the requested Golden Set.
