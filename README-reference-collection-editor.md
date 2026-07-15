# HTH Reference Collection Editor v5

This release adds the requested review-workbench refinements.

## New in v5

### Page information ribbon

The ribbon above the image shows:

- FamilySearch/global image number
- layout type
- Stage 2 status
- confidence
- review status
- approval status

### Mute prediction

The setting is named **Mute prediction after approval**.

Once a page has an approved reference rectangle, the Stage 2 prediction remains available but is rendered as a much fainter dashed outline. It is not called a ghost prediction.

### Difference diagnostics

When both prediction and approval exist, the editor reports the pixel differences for:

- left edge
- top edge
- right edge
- bottom edge
- width
- height

This helps identify systematic detector errors.

### Draggable editing

- Draw a new green rectangle on empty image space.
- Drag inside it to move it.
- Drag corners or edges to resize it.

### Undo

- **Ctrl+Z** or the Undo button restores the previous box.
- Up to 50 box states are retained per page.

### Keyboard workflow

- `Space` — approve
- `A` — use Stage 2 prediction
- `N` / `P` — next / previous
- `F` — fit image
- `1` — 100%
- `C` — compare
- `Ctrl+Z` — undo
- `Esc` — cancel active drag
- `Ctrl+S` — approve

### Auto-advance and filters

Approval can automatically advance to the next unresolved review page. Thumbnail filters show or hide Review, Approved, and Other pages.

## Install

Replace:

```text
tools/reference-collection-editor.html
README-reference-collection-editor.md
```

with these files.

No workflow change is required. A push that changes only the editor should not run the preprocessing workflow unless its path filter intentionally includes `tools/**`.
