# HTH Reference Collection Editor

A self-contained browser application for curating approved Stage 2 reference pages.

## Install

Place the files at:

```text
tools/reference-collection-editor.html
README-reference-collection-editor.md
```

No installation, server, or external JavaScript library is required. Open the HTML file in Chrome or Edge.

## Workflow

1. Click **Load reference JSON** and select `config/golden_set.json`.
2. Click **Load image folder** and select the extracted-image folder.
3. Images are matched by ordinals in filenames such as `fs_0001.png`, `fs-0001.jpg`, or `page_0001.png`.
4. Select the layout type.
5. Drag a rectangle around the complete physical document, sheet, cover, or open-book spread.
6. Add notes for unusual pages.
7. Click **Save page changes**.
8. Continue through the reference pages.
9. Click **Export JSON**.
10. Replace `config/golden_set.json` with the downloaded `reference_collection.json`.

Bounding boxes use original-image coordinates:

```text
[left, top, right, bottom]
```

The app runs entirely in the browser and does not upload the images or JSON.

## Current scope

The first version edits:

- global ordinal
- label
- layout type
- physical-document bounding box
- notes

It does not yet edit gutters, logical left/right page boxes, or visual regions. Those can be added after the physical-document reference collection is stable.

The repository filename can remain `golden_set.json` temporarily for validator compatibility, while the user-facing concept is the **reference collection**.
