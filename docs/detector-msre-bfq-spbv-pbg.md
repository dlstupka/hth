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

The first calibration used 2,187 exhaustive fusion parameter sets. That run identified `minimum_side_consensus` and `consensus_tolerance_fraction` as the two critical, strongly interacting fusion controls; the other five fusion dimensions were dormant on the current Golden Set and configured grid.

The current refinement therefore keeps the child calibrations fixed, collapses the dormant fusion dimensions to the Gen1 baseline values, and exhaustively evaluates **50,176** joint combinations across:

- `minimum_side_consensus`: 224 values spanning `0.10` through `0.90`, explicitly retaining the original `0.25`, `0.50`, and `0.75` anchors.
- `consensus_tolerance_fraction`: 224 values spanning `0.004` through `0.050`, explicitly retaining the original `0.006`, `0.012`, and `0.024` anchors.

The original Gen1 baseline point (`minimum_side_consensus=0.50`, `consensus_tolerance_fraction=0.012`) remains exactly represented, so the refinement cannot lose the existing calibrated candidate merely because the grid changed. The expanded tolerance range also tests beyond the previous `0.024` search boundary rather than assuming the coarse-grid edge was sufficient.

Winner and verbose debug artifacts include all child quadrilaterals, the final fused quadrilateral, and a color-coded view of the child source selected for each side.
