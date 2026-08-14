# Page Background Detector

`page_background` approaches page extraction from the opposite direction of HTH's edge and contour detectors: it learns the surrounding capture background from the outer image border, then extracts the dominant region that does **not** resemble that background.

The detector converts the image to CIE Lab, estimates robust median/MAD background statistics from an outer border band, and measures every pixel's normalized distance from that model. Pixels sufficiently close to the border model are classified as background. The inverse mask is morphologically cleaned and candidate regions are ranked by page-area plausibility, rectangularity, and proximity to the image center before fitting a minimum-area quadrilateral.

This detector is intentionally complementary to `whitespace_frame`. Whitespace Frame assumes a bright background threshold; Page Background instead models whatever surrounds the page, allowing dark scanner borders, colored mounts, cradle material, and non-white capture backgrounds to be treated as negative-space evidence.

Its first exhaustive calibration grid contains 2,187 parameter sets spanning border-model width, robust color-distance threshold, smoothing, morphology, required border-background coherence, and minimum page area.
