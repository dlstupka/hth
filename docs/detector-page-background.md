# Page Background Detector

`page_background` approaches page extraction from the opposite direction of HTH's edge and contour detectors: it learns the surrounding capture background from the outer image border, then extracts the dominant region that does **not** resemble that background.

The detector converts the image to CIE Lab, estimates robust median/MAD background statistics from an outer border band, and measures every pixel's normalized distance from that model. Pixels sufficiently close to the border model are classified as background. The inverse mask is morphologically cleaned and candidate regions are ranked by page-area plausibility, rectangularity, and proximity to the image center before fitting a minimum-area quadrilateral.

This detector is intentionally complementary to `whitespace_frame`. Whitespace Frame assumes a bright background threshold; Page Background instead models whatever surrounds the page, allowing dark scanner borders, colored mounts, cradle material, and non-white capture backgrounds to be treated as negative-space evidence.

The first 2,187-set exhaustive calibration produced 0.9662 Avg IoU / 0.9476 Min IoU, and its winner pressed against six declared boundaries. Generation 2 therefore expanded to 500,000 sets and improved to 0.9690 Avg IoU / 0.9498 Min IoU. Its verbose top basin then showed that the acceptance thresholds were effectively dormant while every top configuration used the maximum tested Lab color-distance threshold of 10.0. Generation 3 therefore uses exactly 200,000 exhaustive sets: it collapses the dormant border-background and page-area acceptance thresholds to their winning values, pushes color distance through 22.0, and spends the recovered search budget resolving the local border-band, blur, and morphology basin around the Generation-2 winner.

Calibration-grid changes are declared execution-shape compatible for this detector: they do not invalidate completed optimizer evidence when detector implementation, Golden Set, image dimension, and other workload guards remain compatible.
