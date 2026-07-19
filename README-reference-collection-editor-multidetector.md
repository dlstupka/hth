# HTH Reference Collection Editor — Multi-Detector Workbench

The workbench displays every registered geometry detector together with the Golden Set reference geometry on one image. The overlay model is intentionally limited to six detector outputs plus one reference overlay.

| Overlay | Color |
|---|---|
| Contour (HTH) | Cyan |
| Connected Components (OpenCV) | Blue |
| RANSAC (HTH) | Magenta |
| Hough Lines (OpenCV) | Amber |
| Line Segment Detector / LSD (OpenCV) | Lime |
| GrabCut (OpenCV) | Violet |
| Golden Set (Approved) | Green |

## Visibility and transparency

Each overlay has two independent controls:

- A checkbox enables or disables the overlay without losing its opacity setting.
- An opacity selector supports `100%`, `75%`, `50%`, `25%`, and `0%`.

`0%` is fully transparent in both normal and compare views. **Disable all detectors** and **Enable all detectors** affect detector overlays only. The Golden Set reference remains independently controllable.

The **Overlay line width** control uses a constant screen-space width from 1–4 pixels. Lines therefore remain readable and do not become excessively thick while zooming. The currently selected detector is drawn one pixel wider than the configured base width.

## Selecting a detector

The selected detector drives:

- the confidence panel,
- detector metadata,
- the **Use selected detector** action,
- the emphasized overlay line,
- comparison deltas against the Golden Set.

A detector remains selectable even when its overlay checkbox is disabled. This allows reviewing its metadata or adopting its box without forcing the overlay to remain visible.

The Golden Set is a reference, not a detector, and therefore never appears in the selected-detector list.

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

Legacy page-level geometry fields are no longer rendered as an additional detector overlay. Approved `physical_document_bbox` geometry belongs to the Golden Set reference; detector output belongs in `geometry_candidates`.

## Files

The workbench is self-contained:

```text
tools/reference-collection-editor-multidetector.html
```

Open it locally in a modern browser, then choose **Open results workspace** and select the results repository directory containing the raw images and `page-analysis.json`.
