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
2. Label 8-connected foreground components with OpenCV `connectedComponentsWithStats`.
3. Remove components below a scale-relative area threshold.
4. Seed the envelope with the largest meaningful component.
5. Merge nearby, sufficiently large fragments.
6. Reject implausibly small envelopes.
7. Score the surviving envelope using mask coverage, fill ratio, and page-area coverage.

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
    "fill_ratio": 0.73
  }
}
```

A normal miss uses `status: no_candidate`; a plugin exception is isolated and represented as `status: error` so other detectors still run.

## Plugin design direction

Detector implementations return a `Candidate`. The registry supplies provenance, timing, validation, exception isolation, and reporting. Production loading can therefore be strongly vetted while test/CI registries can substitute experimental `DetectorSpec` entries with very little ceremony.
