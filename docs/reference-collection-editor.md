# HTH Reference Collection Editor — Detector Golden Set

Run `python tools/reference-collection-director.py` to open both editors through the localhost launcher. Enter the public collection source-repository URL and load the frozen Golden Set release; the current collection and HTH-GOLDEN-0002 are defaults. The director downloads and verifies the release once for both tabs. Each editor has independent source-repository and Golden Set controls, so research in one tab need not change the other. Download and hash-check status appears in the editor. A manually downloaded release ZIP or small extracted image folder remains a fallback. Do not select the full results checkout for source images.

The companion results repository is shown automatically as `<source-repository>-results`.
Use **Git pull results checkout** to fast-forward the matching local sibling
checkout; the status reports the resulting commit or explains why the update
was refused. **Open result repository workspace** remains available for its
derived analysis. Reopen the picker after a pull because browsers retain a
snapshot of the files previously selected.

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

The editor and local launcher are:

```text
tools/reference-collection-editor.html
tools/reference-collection-director.py
```

Open the localhost page in a modern browser for automatic release loading. **Open result repository workspace** remains an optional route for derived analysis and detector overlays such as `page-analysis.json`; it is not the source of frozen Golden Set images.
