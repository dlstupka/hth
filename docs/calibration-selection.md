# HTH Calibration Selection Workbench

The reference-collection editor supports explicit calibration membership so Golden Set and detector-research pages can be selected without changing the normal production or preprocess-test paths.

## Workbench controls

**Show images** supports:

- All images
- Calibration set only
- Not in calibration set
- Ordinal range
- Explicit ordinals such as `1-20, 25, 31-35`
- Needs review
- Approved

Each page can carry:

```json
"calibration_selected": true
```

Missing values default to `true` for backward compatibility.

The workbench can:

- include or exclude the current page;
- include or exclude all currently visible pages;
- export `calibration_manifest.json`;
- preserve calibration membership in `reference_collection.json`.

## Current pipeline separation

Calibration is deliberately separate from production preprocessing and the fast `preprocess-test.yml` smoke path.

`calibrate-geometry.yml` and the shared workflow core are responsible for selecting the requested Golden Set/calibration scope, evaluating detector parameter sets, persisting calibration evidence, and publishing calibration intelligence without overwriting production or test outputs.

Authoritative detector choice is resolved from persisted calibration evidence. Production preprocessing uses the strongest **Approved** authoritative calibration for the requested Golden Set rather than rerunning detector research.

## Golden Set discipline

Calibration pages tune detector parameters and selection. Frozen Golden Sets give persisted calibration evidence a stable identity and SHA-256. `HTH-0001` is the frozen legacy identity and must never be edited in place; corrected or expanded truth receives an identity matching its release tag, such as `HTH-GOLDEN-0002`.

A broader Golden Set should be driven by evidence from full-collection inference: low-confidence pages, structurally distinct capture regimes, detector-disagreement cases, and representative controls. Detector confidence is useful for prioritization but is not ground truth.

See:

- [`golden-set.md`](golden-set.md)
- [`regression.md`](regression.md)
- [`document-detector-review.md`](document-detector-review.md)
