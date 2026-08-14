# Page Background Detector

`page_background` approaches page extraction from the opposite direction of HTH's edge and contour detectors: it learns the surrounding capture background from the outer image border, then extracts the dominant region that does **not** resemble that background.

The detector converts the image to CIE Lab, estimates robust median/MAD background statistics from an outer border band, and measures every pixel's normalized distance from that model. Pixels sufficiently close to the border model are classified as background. The inverse mask is morphologically cleaned and candidate regions are ranked by page-area plausibility, rectangularity, and proximity to the image center before fitting a minimum-area quadrilateral.

This detector is intentionally complementary to `whitespace_frame`. Whitespace Frame assumes a bright background threshold; Page Background instead models whatever surrounds the page, allowing dark scanner borders, colored mounts, cradle material, and non-white capture backgrounds to be treated as negative-space evidence.

The first 2,187-set exhaustive calibration produced a strong 0.9662 Avg IoU / 0.9476 Min IoU result, but six of seven winning calibration coordinates sat on the edge of the declared domain. Generation 2 therefore expands to exactly 500,000 exhaustive sets. It probes substantially smaller border bands, higher Lab-distance thresholds, finer near-zero blur/closing behavior, stronger opening, and lower required border-background coherence while retaining the original baseline and first winner as anchors. The page-area dimension remains comparatively sparse because its first winner was interior rather than boundary-limited.

Calibration-grid changes are declared execution-shape compatible for this detector: they do not invalidate completed optimizer evidence when detector implementation, Golden Set, image dimension, and other workload guards remain compatible.
