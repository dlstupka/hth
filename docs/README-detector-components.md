# Connected Components Geometry Detector

The `components` detector is the first detector added after the HTH detector-registry refactor.

## Purpose

Connected Components provides an independent, deterministic estimate of the physical page envelope from the shared document mask. It complements contour, RANSAC, and Hough detection rather than replacing them.

## Method

1. Convert the shared document mask to a binary image.
2. Label 8-connected foreground components.
3. Remove components below a scale-relative area threshold.
4. Start with the largest meaningful component.
5. Merge nearby meaningful fragments into one conservative envelope.
6. Reject envelopes too small to plausibly represent a photographed page.
7. Score the candidate using mask coverage, component fill, and envelope size.

## Candidate output

The detector publishes the standard HTH `Candidate` contract:

```json
{
  "method": "components",
  "bbox": [35, 20, 265, 180],
  "corners": [[35.0, 20.0], [265.0, 20.0], [265.0, 180.0], [35.0, 180.0]],
  "confidence": 0.91,
  "score": 0.91,
  "diagnostics": {
    "component_count": 4,
    "significant_components": 2,
    "merged_components": 2,
    "elapsed_ms": 3.4
  },
  "status": "ok"
}
```

A normal inability to identify a plausible page returns `bbox: null`; the registry normalizes that result to `status: no_candidate`. Exceptions remain isolated by the registry and become `status: error` candidates without stopping other detectors.

## Registry order

The candidate order is now:

```text
contour
components
ransac
hough
```

The order is deterministic and does not imply that an earlier candidate is preferred.

## Validation

Run:

```bash
python -m unittest discover -s tests -v
python -m compileall -q hth
git diff --check
```

The synthetic tests verify:

- detection of a large connected page region;
- merging of nearby page fragments;
- normal `no_candidate` behavior for insignificant noise.
