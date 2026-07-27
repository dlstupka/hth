# Consensus Quad detector

Consensus Quad combines the independently scored quadrilateral results from the Contour Quadrilateral and Edge-Contour Hybrid detectors. It accepts a page only when both voters return usable quadrilaterals and their polygons agree within configured overlap and corner-distance limits.

The accepted polygon is a confidence-weighted average of corresponding corners. The detector therefore preserves the speed and strong hypotheses of the contour family while making agreement explicit rather than silently preferring one implementation.

## Evidence and rejection

The detector records both voter results, polygon IoU, mean and maximum corner separation, normalized voter weights, source confidence, and the final agreement score. It returns `no_candidate` when either voter is absent, polygon overlap is too low, corner separation is too large, or final confidence is below the configured minimum.

## Debug artifacts

Winner debug output includes the original page and input mask through the common regression framework, plus:

- Contour Quadrilateral vote;
- Edge-Contour vote;
- combined agreement overlay;
- selected consensus quadrilateral;
- final overlay; and
- diagnostics JSON containing both source candidates and consensus metrics.

The child detectors use their reviewed baseline parameters. Consensus calibration intentionally tunes only agreement and weighting behavior; child-detector calibration remains isolated in each detector's own configuration.
