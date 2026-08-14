# Fusion Gen2 — AMSRE + BFQ + SPBV + Page Background

`amsre_bfq_spbv_pbg` is the controlled successor to Fusion Gen1.

Generation 2 changes exactly one evidence family: calibrated MSRE is replaced by calibrated AMSRE winner `21ea516c3c5a`. Border Fusion Quad (`2370e6cea486`), Signed Polar Boundary Voting (`8ddbe5f468cd`), and Page Background (`afbe81a796a1`) remain unchanged.

The fusion baseline is calibrated Fusion Gen1 winner `7b7dbac43ea6`, so baseline behavior is a direct architectural A/B: Gen1 and Gen2 use the same fusion decision point, BFQ, SPBV, and Page Background; only MSRE versus AMSRE changes.

The full calibration retains Fusion Gen1's proven 50,176-set joint `minimum_side_consensus` × `consensus_tolerance_fraction` surface. This preserves the clean baseline comparison while still allowing the stronger radial child to reveal a shifted fusion optimum.

Winner and verbose debug artifacts include all child quadrilaterals, the final fused quadrilateral, and a color-coded side-source view.
