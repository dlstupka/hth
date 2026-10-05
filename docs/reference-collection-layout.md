# HTH layout reference editor

## Feature and shortcut cheat sheet

Use **Ctrl** on Windows/Linux or **Cmd** on macOS for the shortcuts below.
Geometry shortcuts apply to the canvas selection when you are not typing in
a text field. Button-only features have no dedicated keyboard shortcut.

| Feature | Shortcut or control | Result |
|---|---|---|
| Load verified source images | **Load both tabs** in the director; **Load source release** in the editor | Load the frozen Golden Set and verify its images. |
| Load images manually | **Open GS0002 image ZIP** / **Or open image folder** | Verify supplied images against the source identities. |
| Choose another layout Golden Set | Golden Set selector → **Choose layout Golden Set JSON…** | Load a different layout definition. |
| Update local results | **Git pull results checkout** | Update the matching results checkout. |
| Open a saved draft | **Open draft JSON** | Restore annotations and the last viewed page. |
| Save the draft | **Ctrl+S** / **Save draft JSON** | Write to the chosen file; preserve page, zoom and canvas position. |
| Save to another file | **Save draft as…** | Choose a new save destination. |
| Change pages | Page-number buttons / **Previous** / **Next** | View another page. |
| Hide or show top controls | **Hide top controls** / **Show top controls** | Reclaim canvas space while keeping the side panes visible. |
| Zoom the image | **Ctrl+mouse wheel** over the canvas / zoom **−** and **+** | Zoom the page; Ctrl+wheel remains Ctrl on macOS. |
| Fit the page | **Fit page width** | Fit the image to the canvas viewport width. |
| Pan the view | Canvas scrollbars / normal mouse wheel | Change the visible part of the image. |
| Draw a rectangle | **Draw rectangle**, then drag | Add an axis-aligned region; the tool stays active. |
| Draw a polygon | **Draw polygon**, then click its vertices | Build a free-form region; the tool stays active. |
| Finish a polygon | **Enter** / **Finish polygon** / double-click while drawing | Complete the polygon. |
| Cancel drawing or mirroring | **Escape** / **Cancel drawing**; **Cancel mirroring** for an active mirror | Return to Select / edit. |
| Select a region or handle | **Select / edit**, then click | Select the region, vertex or edge. |
| Select an overlapping handle | Repeat clicks at the same spot | Cycle all vertices and edges within 10 screen pixels, closest first. |
| Select multiple regions | **Ctrl+click** on regions or their list rows | Toggle each region in the selection. |
| Box-select regions | Drag from empty canvas | Select every region touched by the box. |
| Add to a box selection | **Shift+drag** from anywhere | Add touched regions to the selection. |
| Move selected regions | Drag inside a selected region | Move the selected group, preserving shapes and spacing. |
| Reshape a polygon | Drag a vertex or edge | Move a vertex, or move an edge's two endpoints together. |
| Resize a rectangle | Drag a corner or side | Resize while preserving the rectangle's shape. |
| Nudge a selection | **Arrow keys** / on-screen arrow buttons | Move the selected vertex, edge or region group by one source-image pixel. Hold an on-screen arrow to repeat in one Undo step. |
| Walk a polygon boundary | **Ctrl+Arrow** | Alternate adjoining edge/vertex selections within one polygon. Repeat the same arrow to continue; the opposite arrow reverses. From a whole-region selection, start at a vertex in that direction. |
| Select another polygon | **Ctrl+Shift+Arrow** | Select the next polygon in the arrow's direction. |
| Add a vertex from a selected vertex | **Ctrl+C**, then **Ctrl+V** / **Add vertex** | Split the adjacent edge on the right at its midpoint. |
| Duplicate a selected edge | **Ctrl+C**, then **Ctrl+V** in the same polygon | Extend from its ending vertex with an adjacent edge of the same length and direction; select the new edge. Repeated pastes extend it again. |
| Split a selected edge | **Add vertex** | Insert its midpoint; select the segment to the right for another split. |
| Add a vertex at a specific point | Double-click a polygon edge | Insert at the clicked point; suppressed when overlapping handles are cycling. |
| Add a vertex with only the polygon selected | **Add vertex** | Split its rightmost edge at mid-height. |
| Mirror a copied edge | **Ctrl+C** on the source edge, select another region or its facing edge, then **Ctrl+V** | Copy the full facing boundary chain to the target at identical coordinates. |
| Mirror using the tool | **Mirror boundary…**, then click the target edge | Share the source boundary with the neighboring region. |
| Clone regions | **Ctrl+C**, then **Ctrl+V** on whole regions / **Clone selected** | Try below, right, above, then left, with a 10-pixel group gap. Repeated pastes continue from the newest clones. |
| Delete a vertex | **Delete** / **Ctrl+X** / **Delete vertex** | Remove the selected polygon vertex; select the adjacent vertex farther right. |
| Delete an edge | **Delete** / **Ctrl+X** | Collapse the edge to its right-hand endpoint. |
| Delete regions | **Delete** / **Ctrl+X** / **Delete region** | Remove the selected region group. |
| Convert a shape | **Convert to rectangle** / **Convert to polygon** | Use the bounding rectangle, or unlock the rectangle's four corners for polygon editing. |
| Undo | **Ctrl+Z** / **Undo** | Undo an edit or drawing step while preserving the current canvas view. |
| Redo | **Ctrl+Y** / **Ctrl+Shift+Z** / **Redo** | Restore an undone edit or drawing step. |
| Assign a region class | **Region class** field | Set the selected region's class. |
| Label and arrange reading order | Selected-region label, reading-order rows and their arrow buttons | Label regions and arrange their sequence. |
| Suggest reading order | **Suggest from layout** | Restore the provisional spatial sequence for inspection. |
| Confirm reading order | **Confirm reading order** | Record the reviewed region sequence. |
| Review region geometry | **Mark page reviewed** | Confirm region geometry; when reading order is also confirmed, advance to the next incomplete page. |
| Add a page note | **Page-wide note** field | Save a note for the entire page. |
| Add or edit an anchored note | Select note anchors / **Use selected region**, then **Add anchored note**; note edit/delete controls | Attach a note to one or more regions; editing and deletion are undoable. |
| Clear an unfinished note | **Clear** | Clear the note editor. |
| Import model proposals | **Import Kraken proposals** | Load matching unapproved proposals for review. |
| Select proposals | Click; **Ctrl+click** to toggle; **Shift+click** for a range | Select visible proposals in the list or on the image. |
| Adopt or dismiss proposals | **Adopt selected**, **Dismiss selected**, **Adopt all on page**, **Dismiss all on page** | Apply an undoable batch decision. |
| Restore dismissed proposals | **Show dismissed**, select proposals, then **Restore dismissed** | Restore saved dismissed candidates. |
| Toggle overlays | **Layout GS regions**, algorithm checkboxes, **Show dismissed** | Change which outlines are visible. |
| Emphasize a selection | **Emphasis** menu | Mute or highlight selected regions or other outlines. |
| Read in-editor help | **Help** below Finish polygon | Expand the drawing and selection instructions. |

Convert rectangles to polygons before adding or deleting vertices or duplicating
or deleting edges. Edits that would produce an invalid polygon are rejected. Clipboard
geometry stays on its source page. Overlay emphasis and zoom affect the view;
save the draft to preserve annotation and proposal decisions.

## Setup and detailed behavior

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

The director and both editors keep their top controls on the left and their title
and descriptive text on the right. The layout editor keeps drawing tools on the
left of the canvas and region class, reading order, and review notes on the right.
Use **Hide top controls** in the director's tab bar to reclaim vertical space
while leaving the side panes visible. **Show top controls** restores the setup,
page, zoom, and overlay controls. When opening the layout editor directly, use
its own top-controls button.

In the layout tab, load a source release (or use the director's shared load)
before editing a page. Choose a visibly active drawing tool: drag to draw a
rectangle, or click polygon vertices and use **Finish polygon**. The drawing
tool stays active after each region, so you can draw another without reselecting
it. Use
**Select / edit** to click an approved region on the image and drag its visible
vertex handles. Rectangles retain their axis-aligned shape: corner handles
resize two sides, and midpoint handles move one side. **Make rectangle** repairs
a previously skewed region by replacing its boundary with the region's bounding
box. Polygons allow free vertex dragging. Select a polygon edge and click
**Add vertex** to split it at the midpoint; the segment to the right of the
new vertex is selected for the next split. With only a polygon selected,
**Add vertex** chooses its rightmost edge at mid-height. With a vertex selected,
it splits the adjacent edge on the right. Double-click an edge
to insert a vertex at that point. Click a vertex or edge, then use the arrow buttons or keyboard arrow
keys to move it one source-image pixel at a time. Hold an on-screen arrow to
repeat nudges; the entire hold is one Undo step. Rectangle edges move only
perpendicular to themselves; polygon edges move their two endpoints together.
Ctrl+Arrow (Cmd+Arrow on macOS) walks adjoining edge → vertex → edge selections
within the selected polygon. Repeating the same arrow continues around the
boundary; the opposite arrow reverses the walk. With a whole polygon selected,
it starts at a vertex in that direction. Ctrl+Shift+Arrow (Cmd+Shift+Arrow on
macOS) selects another polygon in the chosen direction. These shortcuts do not
move geometry or pan the canvas.
When polygon boundaries overlap, repeated clicks at the same spot cycle through
every approved-region vertex and edge within 10 screen pixels, closest first.
At such crowded spots, double-click insertion is suppressed so the second click
can select the next handle; select an edge and use **Add vertex** instead.
Undo and Redo restore the annotation without changing the canvas zoom or scroll
position. An edit from a different page can be undone without navigating away
from the page currently in view.
A polygon vertex or the whole region can be deleted. After deleting a vertex,
the adjacent vertex farther right is selected so repeated clicks can remove a
run of points; deletion still stops if it would make the polygon invalid.
With the canvas selection active, Ctrl+C then Ctrl+V adds a vertex when a
vertex was copied. Pasting a copied edge within its source polygon extends the
selected edge from its ending vertex with a new adjacent segment of the same
length and direction, and selects the new edge. Repeated pastes continue this
extension. Extensions outside the image or across the boundary are rejected.
An edge pasted onto a selected neighboring region or facing edge mirrors the boundary.
Copied whole regions are cloned in the first available direction.
Repeated region pastes continue from the newest clones. Ctrl+X or Delete removes the
selected vertex, edge, or region. Deleting a polygon edge removes its left-hand
endpoint and keeps the right-hand endpoint as the converged vertex. A rectangle
must be converted to a polygon before an edge can be deleted; edits that would
leave an invalid polygon are rejected. These shortcuts do not override typing
in text fields.
**Undo** and
**Redo** also work while drawing a polygon.
**Convert to rectangle** replaces a polygon with its bounding rectangle;
**Convert to polygon** keeps a rectangle's four corners while enabling free
vertex and edge editing. Both conversions are undoable. Layout Golden Set regions are green,
while algorithm proposals are amber. Both overlays are shown by default and
can be hidden independently above the page. Algorithm switches are generated
from the available algorithms; Kraken is the first supported proposal source.
The **Emphasis** control is display-only: choose normal, mute/highlight the
selected approved region or Kraken proposals, or mute/highlight all other
visible regions. Muted outlines fade while highlighted outlines become bright
magenta; image pixels and draft JSON are unchanged. Without a selection,
the control has no effect.
To make two neighboring regions share a boundary, select an edge on the source
region and click **Mirror boundary…**, then click a facing edge on the target
region. The editor finds each edge's full facing chain, replaces the target
chain with every source vertex at the exact same source-image coordinates, and
converts the target to an editable polygon if necessary. For irregular polygons,
it considers near-extreme corners on the side facing the other region, so an
upper corner a few pixels farther left or right does not hijack a lower seam.
It rejects crossed or
collapsed results, leaves the source untouched, and records a single Undo step.
Inspect the result before reviewing and saving the page.
In **Select / edit**, drag from empty canvas to box-select every approved
polygon the box touches. Shift-drag from anywhere to add polygons to the
selection; Ctrl/Cmd-click toggles individual polygons. Left-drag from inside
any selected polygon to translate all selected polygons together without
changing their geometry or relative spacing. The group stops at the image
border and can be undone in one step. Clicking an edge still selects it
for 1-pixel nudges, **Add vertex**, or **Mirror boundary…**; dragging an edge
adjusts only that edge, and dragging a vertex still reshapes it.
The arrow keys (or on-screen arrows) nudge the selected vertex, edge, or all
selected regions by one source-image pixel per press. Whole-region nudges
preserve each polygon's shape and the spacing between selected polygons.
Choose the region class, then mark the regions reviewed. The editor stays on
the page until its reading order is also confirmed; either review action can
come first. Once both are reviewed, it advances to the next page missing
either review and keeps the completed page's confirmation in the status bar.
After the last complete page, it stays put and prompts you to save. An
independent **Region reading order** card lists stable region IDs in a
provisional creation sequence, matching the order regions were drawn.
Select a region to give it an optional entry label, use the arrows to correct
the sequence, and click **Confirm reading order**. This saves each page's
`reading_order` (an ordered array of region IDs),
`reading_order_method` (`creation_order`, `spatial_suggestion`, or `manual`),
`reading_order_status`, and confirmation time in the version `0.2`
`regions-reading-order-v2` draft. Geometry review and reading-order review
are separate: older draft region reviews remain intact when reopened, while
an unconfirmed legacy spatial suggestion becomes creation order on import;
confirmed and manually arranged orders are preserved. Adding
or deleting a region invalidates its order confirmation. Region labels and
manual order changes also require reconfirmation. **Suggest from layout**
restores the numbered spatial suggestion; if it already matches, the editor
says so without changing the review state. Check the visible numbered rows,
then click **Confirm reading order**. A reviewed region page may be exported with
unreviewed reading order; the JSON distinguishes the two. The suggested
sequence is not semantic reading-order truth, especially on unusual spreads.
Select a region and click **Clone selected** to copy its geometry and class.
The editor tries below, then right, then above, then left, using the first
direction where every clone fits inside the source image. The gap is 10
source-image pixels between the selected group's outermost edge and the
cloned group's nearest edge. Ctrl/Cmd-click approved regions on the image or
in the region list to clone several at once; they move as one group and keep
their relative positions. New regions are selected. If no direction fits,
the action is rejected without changes.
Review notes can be page-wide (`notes`) or anchored to one or more approved
regions (`region_notes`). Each anchored note stores a stable note `id`, a
`region_ids` array, and `text` in the page's draft JSON. Choose both regions
for an uncertain shared boundary, add the note, and save the draft. Notes do
not confirm or change geometry or reading order. Editing or deleting a note
is undoable. If a region is deleted, it is removed from note anchors; a note
with no remaining anchors moves to page-wide notes instead of disappearing.
Saving records `last_viewed_page_ordinal` in the draft, and reopening that
draft returns the viewer to that page. Drafts saved before this change open
on the first page until saved again.
An
expanded Kraken layout artifact may be imported as **unapproved proposals**;
the editor checks its Golden Set and source-file identities before displaying
the source-view polygons. Adopt selected proposals only after visual review;
unwanted proposals can be dismissed without deleting their review decision.
In the Kraken list or on the image, Ctrl-click (Cmd-click on macOS) toggles
individual proposals and Shift-click selects the visible range from the last
clicked proposal. **Adopt selected**, **Dismiss selected**, and **Restore
dismissed** act on the eligible selected proposals in one undoable batch.
**Adopt all on page** and **Dismiss all on page** apply to every active
proposal on the page, including those outside the scrolled list.
Dismissed proposals are hidden by default; **Show dismissed** displays them in
muted gray so one can be selected and restored. The draft JSON stores stable
proposal IDs under each page's `dismissed_proposal_ids`, so the decisions return
when the draft and the same Kraken proposals are imported again. Use **Save
draft JSON** after dismissing or restoring. Where supported, the browser asks
for a file on the first save and writes subsequent saves to that same file.
Use **Save draft as…** to open the chooser again and select a different file or
location; subsequent regular saves use the newly selected file. Cancelling
Save As keeps the previous destination. The editor reports success only after
the write finishes. Saving preserves the current page, zoom, and canvas scroll
position. If the browser only supports downloads, check its
downloads list (Ctrl+J) to confirm the file was created; choosing a new
location requires browser download settings. The saved draft is not an approved
or frozen release.
**Open draft JSON** can reopen the same file repeatedly. If the source Golden
Set and page image hashes match, reopening a layout draft keeps any currently
imported Kraken proposals; for a different source, import matching proposals
again. Selecting the same file in the picker also works on subsequent opens.
The page-only zoom carries forward to untouched pages and remembers each page
you adjust, including after a browser reload. Zoom preferences are browser-local
display state, not part of the exported Golden Set draft.
Zoom can reach 800%; above 200%, the canvas enlarges its display without
allocating a correspondingly huge bitmap. This magnifies the existing source
pixels but cannot reveal detail absent from the verified source image.

This editor now captures **region-level** reading order. It does not yet
capture line baselines or within-region line reading order, so those dimensions
cannot be scored from this draft. Do not treat Kraken output, even when
adopted, as independently reviewed truth.
