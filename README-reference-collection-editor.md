# HTH Reference Collection Editor

This version starts from the latest repository editor and adds the review-workbench improvements:

- opens the complete results repository
- loads `test/latest/reference_collection.json`
- loads `test/latest/raw/`
- loads `test/latest/analysis/page-analysis.json`
- loads `test/latest/analysis/review-queue.csv`
- fits the current image to the available workspace
- mouse-wheel zoom
- red Stage 2 prediction rectangle
- green approved reference rectangle
- **Use Stage 2 prediction** button
- visible Stage 2 quality/review information
- thumbnail strip with approved/review/failure status
- exact coordinates collapsed by default
- approval timestamp and status written to exported JSON

## Install

Replace:

```text
tools/reference-collection-editor.html
```

with the downloaded file.

## Use

1. Pull the results repository.
2. Open the editor in Chrome or Edge.
3. Click **Open results workspace**.
4. Select the results repository root.
5. Review the red Stage 2 prediction.
6. Click **Use Stage 2 prediction** when it is a useful starting point, or drag a new rectangle.
7. Click **Approve**.
8. Export JSON.
9. Replace `hth/config/golden_set.json` with the exported file.

## Display

- Red dashed box: automatic Stage 2 prediction.
- Green solid box: manually approved reference.
- Green thumbnail border: approved.
- Yellow thumbnail border: included in the Stage 2 review queue.
- Red thumbnail border: Stage 2 failure.

The editor still stores boxes in original-image coordinates regardless of display zoom.
