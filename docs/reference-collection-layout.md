# HTH layout reference editor

Open `tools/reference-collection-director.html` for detector and layout tabs, or
`tools/reference-collection-layout.html` directly. The detector tab defaults to
HTH-GOLDEN-0002; the layout tab defaults to its HTH-GOLDEN-0002-LAYOUT draft.
Both have independent manual Golden Set selectors. The director
can pass one selected image workspace to both tabs. Browsers do not grant local
image-file access automatically: choose the immutable source image folder once.

The initial `config/golden_sets/HTH-GOLDEN-0002-LAYOUT.draft.json` is an
**unapproved draft**. It carries the 18 frozen GS0002 page ordinals, source
image SHA-256 values, and approved page boxes, but zero layout regions. This
does not alter the approved detector Golden Set. Region truth uses source-image
pixel coordinates and cannot be applied directly to a normalized view without
the recorded geometric transform.

In the layout tab, choose the source image folder and verify its digest before
editing a page. Draw rectangular or polygonal regions, drag a selected vertex
to correct a polygon, choose the region class, then mark the page reviewed. An
expanded Kraken layout artifact may be imported as **unapproved proposals**;
the editor checks its Golden Set and source-file identities before displaying
the source-view polygons. Adopt selected proposals only after visual review.
Export the draft JSON regularly; it is not an approved or frozen release.

This first editor handles region polygons. Line baselines and reading-order
truth require additional annotation controls and a versioned extension to the
layout contract before automated scoring of those dimensions. Do not treat
Kraken output, even when adopted, as independently reviewed truth.
