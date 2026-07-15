# HTH Reference Collection Editor v4

This version includes the complete review-workbench improvements discussed after the latest pushed editor.

## Added

- Red Stage 2 prediction and green approved reference remain visible together.
- Prediction fill is very light, so handwriting remains readable.
- Approved rectangles have draggable corners, edges, and whole-box movement.
- **Approve** can automatically advance to the next unresolved review page.
- Thumbnail filters for Review, Approved, and Other.
- Confidence is expressed as both a percentage and a word:
  - Excellent
  - Strong
  - Questionable
  - Weak
  - Poor
- Optional metadata overlay on the image.
- Compare mode:
  - prediction on the left
  - approved reference on the right
- Mouse-wheel zoom and fit-to-workspace.
- Exact coordinates remain available but collapsed by default.
- The app still opens the entire checked-out results-repository root.

## Install

Replace:

```text
tools/reference-collection-editor.html
README-reference-collection-editor.md
```

with these files.

## Review workflow

1. Pull the results repository.
2. Open the editor.
3. Click **Open results workspace** and select the results-repository root.
4. Inspect the red Stage 2 prediction.
5. Click **Use Stage 2 prediction**, or draw/resize the green approved rectangle.
6. Select the correct layout type.
7. Click **Approve**.
8. Export JSON.
9. Replace `hth/config/golden_set.json` with the exported file.

## Rectangle editing

- Drag empty image space to draw a new box.
- Drag inside the green box to move it.
- Drag a green edge or corner handle to resize it.
- Red dashed: automatic Stage 2 prediction.
- Green solid: approved reference.

## Compare mode

Click **Compare** to see:

```text
Stage 2 prediction | Approved reference
```

Drawing is disabled while comparing. Click **Exit compare** to resume editing.

## Pipeline previews

The application generates overlays dynamically from raw images and `page-analysis.json`, so permanent Stage 2 preview images are not required for normal review. Actions artifacts may still retain annotated previews as historical diagnostic material.
