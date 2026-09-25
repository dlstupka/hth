# HTH layout reference editor

Run `python tools/reference-collection-director.py` and open the localhost URL
it prints. Enter the public collection source-repository URL, then **Load both
tabs**. The director defaults to the Baptisms collection and HTH-GOLDEN-0002,
but each editor also has its own source-repository and Golden Set controls for
independent research. The director resolves the immutable release, downloads
its frozen Golden Set and source-image bundle, shows download/checking status,
and verifies the freeze record, bundle SHA-256, and every image SHA-256 before
sharing the images with both tabs. It does not need the results repository.
Opening the HTML directly from disk cannot perform automatic release downloads;
the ZIP and extracted-image controls remain as manual fallback.
The local director keeps release assets in a content-addressed cache outside
the repository (under the user's local application-data directory on Windows).
On subsequent loads it verifies the frozen image bundle's size and SHA-256
before using it, so the 36 MiB ZIP need not be downloaded again. A corrupt
cache entry is discarded and fetched afresh; the browser still verifies the
bundle and every image before either editor sees pixels. The status bar reports
whether the image bundle was downloaded or reused locally. Restart the Python
launcher after updating its code; refreshing the browser alone does not update
an already-running server.

The results repository is derived as `<source-repository>-results` and shown in
the director and both tabs; it is not another required input. **Git pull results
checkout** updates the matching sibling checkout with `--ff-only` after checking
its origin and tracked-file cleanliness, and reports whether the commit changed.
If using the detector tab's workspace picker, reopen it after the pull to read
the updated local files. The pull action does not change the frozen source release.

The initial `config/golden_sets/HTH-GOLDEN-0002-LAYOUT.draft.json` is an
**unapproved draft**. It carries the 18 frozen GS0002 page ordinals, source
image SHA-256 values, and approved page boxes, but zero layout regions. This
does not alter the approved detector Golden Set. Region truth uses source-image
pixel coordinates and cannot be applied directly to a normalized view without
the recorded geometric transform.

In the layout tab, load a source release (or use the director's shared load)
before editing a page. Choose a visibly active drawing tool: drag to draw a
rectangle, or click polygon vertices and use **Finish polygon**. Use
**Select / edit** to click an approved region on the image and drag its visible
vertex handles. Rectangles retain their axis-aligned shape: corner handles
resize two sides, and midpoint handles move one side. **Make rectangle** repairs
a previously skewed region by replacing its boundary with the region's bounding
box. Polygons allow free vertex dragging and double-clicking an edge to insert
a vertex. Click a vertex or edge, then use the arrow buttons or keyboard arrow
keys to move it one source-image pixel at a time. Rectangle edges move only
perpendicular to themselves; polygon edges move their two endpoints together.
A polygon vertex or the whole region can be deleted; **Undo** and
**Redo** also work while drawing a polygon. Layout Golden Set regions are green,
while algorithm proposals are amber. Both overlays are shown by default and
can be hidden independently above the page. Algorithm switches are generated
from the available algorithms; Kraken is the first supported proposal source.
Choose the region class, then mark the page reviewed. An
expanded Kraken layout artifact may be imported as **unapproved proposals**;
the editor checks its Golden Set and source-file identities before displaying
the source-view polygons. Adopt selected proposals only after visual review;
unwanted proposals can be dismissed from the current editing session.
Export the draft JSON regularly; it is not an approved or frozen release.
The page-only zoom carries forward to untouched pages and remembers each page
you adjust, including after a browser reload. Zoom preferences are browser-local
display state, not part of the exported Golden Set draft.

This first editor handles region polygons. Line baselines and reading-order
truth require additional annotation controls and a versioned extension to the
layout contract before automated scoring of those dimensions. Do not treat
Kraken output, even when adopted, as independently reviewed truth.
