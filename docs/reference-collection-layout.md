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
box. Polygons allow free vertex dragging. Select a polygon edge and click
**Add vertex** to split it at the midpoint, or double-click an edge to insert
a vertex at that point. Click a vertex or edge, then use the arrow buttons or keyboard arrow
keys to move it one source-image pixel at a time. Hold an on-screen arrow to
repeat nudges; the entire hold is one Undo step. Rectangle edges move only
perpendicular to themselves; polygon edges move their two endpoints together.
A polygon vertex or the whole region can be deleted; **Undo** and
**Redo** also work while drawing a polygon. Layout Golden Set regions are green,
while algorithm proposals are amber. Both overlays are shown by default and
can be hidden independently above the page. Algorithm switches are generated
from the available algorithms; Kraken is the first supported proposal source.
Choose the region class, then mark the page reviewed. The editor advances to
the next page needing review (wrapping around already-reviewed pages) and
keeps the completed page's confirmation in the status bar. After the last
page, it stays put and prompts you to export. An
independent **Region reading order** card lists stable region IDs in a
provisional spatial sequence: top-to-bottom, left-to-right within a row.
Select a region to give it an optional entry label, use the arrows to correct
the sequence, and click **Confirm reading order**. This saves each page's
`reading_order` (an ordered array of region IDs),
`reading_order_method` (`spatial_suggestion` or `manual`),
`reading_order_status`, and confirmation time in the version `0.2`
`regions-reading-order-v2` draft. Geometry review and reading-order review
are separate: older draft region reviews remain intact when reopened, while
their spatial order is only a suggestion until explicitly confirmed. Adding
or deleting a region invalidates its order confirmation. Region labels and
manual order changes also require reconfirmation. **Suggest from layout**
restores the numbered spatial suggestion; if it already matches, the editor
says so without changing the review state. Check the visible numbered rows,
then click **Confirm reading order**. A reviewed region page may be exported with
unreviewed reading order; the JSON distinguishes the two. The suggested
sequence is not semantic reading-order truth, especially on unusual spreads.
An
expanded Kraken layout artifact may be imported as **unapproved proposals**;
the editor checks its Golden Set and source-file identities before displaying
the source-view polygons. Adopt selected proposals only after visual review;
unwanted proposals can be dismissed without deleting their review decision.
Dismissed proposals are hidden by default; **Show dismissed** displays them in
muted gray so one can be selected and restored. The draft JSON stores stable
proposal IDs under each page's `dismissed_proposal_ids`, so the decisions return
when the draft and the same Kraken proposals are imported again. Use **Save
draft JSON…** after dismissing or restoring. Where supported, the browser asks
for a file on the first save and writes subsequent saves to that same file;
the editor reports success only after the write finishes. If the browser only
supports downloads, check its downloads list (Ctrl+J) to confirm the file was
created. The saved draft is not an approved or frozen release.
**Open draft JSON** can reopen the same file repeatedly. If the source Golden
Set and page image hashes match, reopening a layout draft keeps any currently
imported Kraken proposals; for a different source, import matching proposals
again. Selecting the same file in the picker also works on subsequent opens.
The page-only zoom carries forward to untouched pages and remembers each page
you adjust, including after a browser reload. Zoom preferences are browser-local
display state, not part of the exported Golden Set draft.

This editor now captures **region-level** reading order. It does not yet
capture line baselines or within-region line reading order, so those dimensions
cannot be scored from this draft. Do not treat Kraken output, even when
adopted, as independently reviewed truth.
