# HTH Reference Collection Editor — Multi-Detector Candidate

This workbench version displays four geometry detectors simultaneously:

| Detector | Color |
|---|---|
| Current HTH detector | Red |
| Contour quadrilateral | Cyan |
| RANSAC four-edge | Magenta |
| Hough-line quadrilateral | Amber |
| Approved/manual reference | Green |

Each overlay has an independent opacity control:

```text
100% / 75% / 50% / 25% / 0%
```

The selected detector drives the confidence panel and the **Use selected detector** button. All available detector boxes may remain visible simultaneously.

## Expected analysis schema

The current detector remains backward compatible with the existing page fields.

Additional candidates are read from:

```json
{
  "geometry_candidates": [
    {
      "method": "contour",
      "bbox": [left, top, right, bottom],
      "confidence": 0.83,
      "score": 0.81,
      "diagnostics": {}
    },
    {
      "method": "ransac",
      "bbox": [left, top, right, bottom],
      "confidence": 0.91,
      "diagnostics": {}
    },
    {
      "method": "hough",
      "bbox": [left, top, right, bottom],
      "confidence": 0.72,
      "diagnostics": {}
    }
  ]
}
```

Aliases such as `contour_quadrilateral`, `ransac_edges`, and `hough_lines` are also accepted.

## Install

Replace:

```text
tools/reference-collection-editor.html
README-reference-collection-editor.md
```

The UI works immediately with the current detector. The other colors appear after `analyze_pages.py` publishes their geometry candidates.
