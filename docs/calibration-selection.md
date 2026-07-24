# HTH Calibration Selection Workbench

This update keeps the fast test/CI workflow unchanged and adds a separate, editable calibration selection to the reference collection editor.

## New workbench controls

**Show images** defaults to **All images** and supports:

- All images
- Calibration set only
- Not in calibration set
- Ordinal range
- Explicit ordinals such as `1-20, 25, 31-35`
- Needs review
- Approved

Each page has:

```json
"calibration_selected": true
```

Missing values default to `true`, preserving the requested “show/use all by default” behavior.

The workbench can:

- include or exclude the current page;
- include or exclude all currently visible pages;
- export `calibration_manifest.json`;
- preserve calibration membership in `reference_collection.json`.

## Recommended pipeline separation

Keep the existing `preprocess-test.yml` exactly as the fast break/fix and sneak-preview workflow.

Add a separate manual calibration workflow later:

```text
calibrate-geometry.yml
```

Its responsibilities should be:

1. Read `reference_collection.json` or `calibration_manifest.json`.
2. Select only pages where `calibration_selected` is true.
3. Run every detector.
4. Score each detector against the approved physical geometry.
5. Produce the leaderboard and page-type summaries.
6. Never publish over the normal test/latest output.

Suggested output namespace:

```text
calibration/<run-id>/
    calibration-manifest.json
    detector-scores.csv
    detector-leaderboard.json
    detector-leaderboard.md
    page-results.json
```

## Scientific safeguard

Use calibration pages to tune thresholds and detector selection.

Reserve a small holdout set that is not used for tuning. The holdout can be selected later with a separate field such as:

```json
"evaluation_role": "calibration"
```

or:

```json
"evaluation_role": "holdout"
```

For the first pass, pages 1–20 can be the calibration set. Add pages in batches of 10 and monitor whether detector rankings and mean IoU materially change.
