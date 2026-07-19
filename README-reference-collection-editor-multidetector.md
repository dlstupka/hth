# HTH Reference Collection Editor — Multi-Detector Workbench

The workbench displays the Stage 2 selected geometry, every registered geometry detector, and the approved/manual reference box on one image.

| Overlay | Color |
|---|---|
| Stage 2 selected geometry | Red |
| Contour (HTH) | Cyan |
| Connected Components (OpenCV) | Blue |
| RANSAC (HTH) | Magenta |
| Hough Lines (OpenCV) | Amber |
| Line Segment Detector / LSD (OpenCV) | Lime |
| GrabCut (OpenCV) | Violet |
| Approved/manual reference | Green |

## Visibility and transparency

Each detector has two independent controls:

- A checkbox enables or disables the overlay without losing its opacity setting.
- An opacity selector supports `100%`, `75%`, `50%`, `25%`, and `0%`.

`0%` is fully transparent in both normal and compare views. **Disable all detectors** and **Enable all detectors** affect detector overlays only; the approved/manual box remains independent.

The **Overlay line width** control uses a constant screen-space width from 1–4 pixels. Lines therefore remain readable and do not become excessively thick while zooming. The currently selected detector is drawn one pixel wider than the configured base width.

## Selecting a detector

The selected detector drives:

- the confidence panel,
- detector metadata,
- the **Use selected detector** action,
- the emphasized overlay line.

A detector remains selectable even when its overlay checkbox is disabled. This allows reviewing its metadata or adopting its box without forcing the overlay to remain visible.

## Expected analysis schema

Candidates are read from each page record's `geometry_candidates` array:

```json
{
  "geometry_candidates": [
    {
      "method": "components",
      "detector_name": "Connected Components",
      "display_name": "Connected Components (OpenCV)",
      "bbox": [110, 60, 1280, 957],
      "confidence": 0.964,
      "status": "ok",
      "diagnostics": {}
    }
  ]
}
```

Recognized stable method IDs are:

```text
contour
components
ransac
hough
lsd
grabcut
```

The workbench also accepts compatibility aliases including `contour_quadrilateral`, `connected_components`, `ransac_edges`, `hough_lines`, `line_segment_detector`, and `grab_cut`.

The legacy Stage 2 selected box is read from the existing page-level geometry fields and is labeled **Stage 2 selected geometry** rather than treated as a separate detector implementation.

## Files

The workbench is self-contained:

```text
tools/reference-collection-editor-multidetector.html
```

Open it locally in a modern browser, then choose **Open results workspace** and select the results repository directory containing the raw images and `page-analysis.json`.
