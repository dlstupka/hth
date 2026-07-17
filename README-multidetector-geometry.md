# HTH Multi-Detector Geometry Trial

This trial preserves the existing HTH detector and adds three established geometry candidates:

1. **Contour quadrilateral**
2. **RANSAC four-edge fitting**
3. **Hough-line quadrilateral**

The workbench displays all four alongside the approved/manual reference.

## Files

```text
hth/detect_geometry_candidates.py
tools/reference-collection-editor.html
README-multidetector-geometry.md
```

## Requirements

Add:

```text
opencv-python-headless
scikit-image
```

`numpy` is installed transitively by both packages, but it may also be listed explicitly.

## Pipeline step

Run this immediately after `analyze_pages.py`:

```yaml
- name: Generate geometry candidates
  shell: bash
  run: |
    set -euo pipefail

    python hth-pipeline/hth/detect_geometry_candidates.py \
      --manifest "$OUTPUT_DIRECTORY/metadata/image_manifest.json" \
      --analysis "$OUTPUT_DIRECTORY/page-analysis/page-analysis.json" \
      --image-root "$OUTPUT_DIRECTORY" \
      --output "$OUTPUT_DIRECTORY/page-analysis/page-analysis-with-candidates.json" \
      --overwrite

    mv \
      "$OUTPUT_DIRECTORY/page-analysis/page-analysis-with-candidates.json" \
      "$OUTPUT_DIRECTORY/page-analysis/page-analysis.json"
```

The replacement keeps all original analysis fields and adds:

```json
"geometry_candidates": [
  {
    "method": "contour",
    "bbox": [0, 0, 100, 100],
    "corners": [],
    "confidence": 0.8,
    "score": 0.8,
    "diagnostics": {}
  },
  {
    "method": "ransac",
    "...": "..."
  },
  {
    "method": "hough",
    "...": "..."
  }
]
```

## Workbench overlays

| Method | Color |
|---|---|
| Current HTH | Red |
| Contour | Cyan |
| RANSAC | Magenta |
| Hough | Amber |
| Approved/manual | Green |

Each has an independent `100 / 75 / 50 / 25 / 0%` opacity selector.

## Important expectation

This is an evaluation trial, not a declaration that any detector is correct.

The reference collection and physical-geometry validator decide which method performs best by page type. Retain every candidate and its diagnostics, even when it loses.
