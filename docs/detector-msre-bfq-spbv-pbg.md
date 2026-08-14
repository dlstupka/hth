# Fusion Gen1 — MSRE + BFQ + SPBV + Page Background

`msre_bfq_spbv_pbg` is the first HTH cross-family page-hypothesis fusion detector.

It gives one seat each to four strong but materially different calibrated detector families:

- Multi-Scale Radial Edge (`ddb7623ebb92`) — multiscale radial boundary evidence.
- Border Fusion Quad (`2370e6cea486`) — radial/polar/gradient side-level fusion.
- Signed Polar Boundary Voting (`8ddbe5f468cd`) — signed/polar transition evidence.
- Page Background (`afbe81a796a1`) — capture-background / negative-space evidence.

Radial Edge and Adaptive Radial Edge are intentionally excluded because MSRE is their stronger radial-lineage successor; including all three would give correlated radial evidence multiple votes.

## Fusion strategy

Each calibrated child emits a quadrilateral. Fusion Gen1 enumerates side-level recombinations, intersects the chosen side lines, and ranks valid quadrilaterals using cross-child side consensus, direct gradient support, and source diversity.

The initial calibration grid contains 2,187 exhaustive fusion parameter sets. Child calibrations are fixed provenance anchors during this generation so the regression tunes fusion behavior rather than silently retuning the underlying detectors.

Winner and verbose debug artifacts include all child quadrilaterals, the final fused quadrilateral, and a color-coded view of the child source selected for each side.
