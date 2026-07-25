# Connected Components detector

The `components` detector is the first detector added after the HTH detector-registry refactor.

## Identity and provenance

The stable machine identifier remains `components`. Human-facing output uses:

```text
Connected Components (OpenCV)
```

The registry records its name, origin, installed OpenCV version, and upstream repository. That metadata is attached automatically to every candidate; the detector itself remains concerned only with detection.

## Algorithm

1. Build the shared binary document mask.
2. Apply scale-relative morphological closing and dilation so nearby ink fragments can form meaningful regions.
3. Label 8-connected foreground components with OpenCV `connectedComponentsWithStats`.
4. Remove components below a scale-relative area threshold.
5. Seed the envelope with the largest meaningful component.
6. Merge nearby, sufficiently large fragments.
7. Reject implausibly small envelopes.
8. Score the surviving envelope using mask coverage, fill ratio, and page-area coverage.

## Regression calibration

Connected Components participates in the same detector-agnostic regression
framework as Contour and GrabCut. Its calibration space is defined in
`config/detectors/components.json`; it can be selected directly from the
workflow **Algorithm** input as `components`, or included through `all`.

The baseline preserves the detector's original behavior. Regression can tune:

- minimum component area, expressed as both a scale-relative fraction and an
  absolute pixel floor;
- the minimum area of fragments relative to the largest component;
- the distance over which nearby fragments may merge;
- minimum plausible envelope and selected-component area fractions;
- conservative output bounding-box padding;
- scale-relative morphology closing and dilation kernels.

The exhaustive calibration space contains 19,683 parameter sets. Smoke runs
retain the workflow's normal 10-set cap, while a full run with a blank limit
evaluates the complete space.

## Candidate output

A successful result includes both normalized plugin metadata and algorithm diagnostics:

```json
{
  "method": "components",
  "detector_name": "Connected Components",
  "origin": "OpenCV",
  "version": "<installed OpenCV version>",
  "repository": "https://github.com/opencv/opencv",
  "status": "ok",
  "confidence": 0.84,
  "diagnostics": {
    "elapsed_ms": 11.2,
    "significant_components": 2,
    "merged_components": 2,
    "bbox_area_fraction": 0.61,
    "fill_ratio": 0.73,
    "largest_component_fraction": 0.31,
    "largest_merged_fraction": 0.44,
    "envelope_fraction": 0.61,
    "text_density": 0.08
  }
}
```

A normal miss uses `status: no_candidate`; a plugin exception is isolated and represented as `status: error` so other detectors still run.

## Plugin design direction

Detector implementations return a `Candidate`. The registry supplies provenance, timing, validation, exception isolation, and reporting. Production loading can therefore be strongly vetted while test/CI registries can substitute experimental `DetectorSpec` entries with very little ceremony.

## Provenance metadata

The detector registry owns provenance so detector implementations stay small.
Each registered detector may declare:

- `origin`: the project or upstream source credited in runtime reports;
- `foundation`: algorithms and libraries on which the detector is built;
- `authors`: the primary source authors or implementers;
- `version`: an optional implementation or library version;
- `repository`: an optional source repository.

Connected Components is reported as `Connected Components (OpenCV)`, with
OpenCV recorded as both its origin and foundation.

### Field ownership and version policy

`DetectorSpec` in `hth/geometry/registry.py` is the authoritative source for all
identity and provenance fields. `detector_components.py` does not construct or
maintain them. The registry overwrites any detector-supplied metadata before the
candidate is serialized, preventing drift between implementations and reports.

Field meanings are documented in `README-multidetector-geometry.md`. In
particular, `version` is the installed OpenCV version for this detector and
`repository` is the canonical OpenCV source repository. The exact HTH pipeline
commit remains recorded separately in every page-analysis record.

## Regression debug images

Connected Components regression debug directories include the common source, input-mask, overlay, and diagnostics files plus two detector-specific intermediates:

- `after-morphology.png` shows the mask after closing and dilation;
- `component-labels.png` renders each resulting connected component in a deterministic distinct color.

These images make it possible to distinguish thresholding problems from morphology, fragmentation, filtering, and envelope-merging failures.
