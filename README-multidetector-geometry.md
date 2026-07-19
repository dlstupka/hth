# HTH Multi-Detector Geometry Trial

The geometry trial runs six registered candidates independently:

1. **Contour quadrilateral**
2. **Connected Components envelope**
3. **RANSAC four-edge fitting**
4. **Hough-line envelope**
5. **OpenCV Line Segment Detector (LSD)**
6. **OpenCV GrabCut segmentation**

Every candidate is retained with diagnostics and provenance. Existing workbench overlays remain backward compatible; UI support for additional overlays may be added separately.

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
  },
  {
    "method": "lsd",
    "...": "..."
  },
  {
    "method": "grabcut",
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

## Detector registry and provenance metadata

The geometry registry is the single source of truth for detector identity and
provenance. Detector implementations return only a `Candidate` containing the
algorithm result and diagnostics. The registry validates that result, records
elapsed time, isolates exceptions, and injects the metadata below before JSON
serialization and reporting.

| Field | Purpose |
|---|---|
| `method` | Stable machine identifier used by configuration, tests, and downstream consumers. Renaming a display label must not change this value. |
| `name` / `detector_name` | Human-readable detector name. `DetectorSpec.name` is serialized as `detector_name` for compatibility with the current page-analysis schema. |
| `origin` | Project or upstream source primarily credited for the detector implementation shown in runtime reports. |
| `foundation` | Ordered list of algorithms and libraries on which the implementation is built. This is always represented as a list. |
| `authors` | Ordered list of primary implementers or upstream contributor groups credited for the implementation. |
| `version` | Human-readable implementation version. HTH detectors use the HTH framework version; OpenCV detectors use the installed OpenCV version. Exact source reproduction remains anchored by the pipeline commit recorded elsewhere in page analysis. |
| `repository` | Canonical source repository for the credited implementation. HTH detectors point to the HTH repository; OpenCV detectors point to the OpenCV repository. |
| `entrypoint` | Python callable invoked by the registry. It is runtime configuration and is not serialized. |
| `display_name` | Derived presentation label in the form `Name (Origin)`; it is not independently configured. |

`family` is intentionally not part of `DetectorSpec`. All current entries are
geometry detectors and are already scoped by the geometry registry and pipeline
stage. A category field can be added later if multiple plugin types genuinely
share one registry.

Production registries should contain only reviewed and pinned detector entries.
Test and CI code may substitute a temporary `DetectorSpec` registry to make
algorithm experimentation inexpensive without weakening production loading.


## Detector-specific documentation

- [Connected Components](docs/README-detector-components.md)
- [Line Segment Detector](docs/README-detector-lsd.md)
- [GrabCut](docs/README-detector-grabcut.md)

LSD and GrabCut are registered as OpenCV-origin detectors. Their `version` is
the installed OpenCV version, their canonical repository is the OpenCV project,
and their implementation metadata is injected solely by `DetectorSpec`.
