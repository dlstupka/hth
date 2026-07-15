# HTH Reference Collection Editor v2

This version can open the **entire checked-out results repository** or a downloaded test workspace in one step.

## Recommended repository layout

```text
hth-baptisms-...-results/
├── config/
│   └── golden_set.json                 # optional repository copy
└── test/
    └── latest/
        ├── raw/
        │   ├── fs_0001.png
        │   └── ...
        ├── analysis/
        │   ├── page-analysis.json
        │   └── review-queue.csv
        ├── metadata/
        ├── BUILD-INFO.yaml
        └── reference_collection.json   # preferred editor input
```

The editor searches the selected folder recursively and automatically discovers:

- `reference_collection.json`, preferred when present
- otherwise `golden_set.json`
- images beneath any `raw/` directory
- `page-analysis.json`
- `review-queue.csv`

## Use

1. Clone or pull the results repository.
2. Open `reference-collection-editor-v2.html` in Chrome or Edge.
3. Click **Open results workspace**.
4. Select the results-repository folder.
5. Review and edit pages.
6. Click **Export JSON**.
7. Copy the exported JSON back to the HTH pipeline repository as:

```text
config/golden_set.json
```

The browser cannot overwrite the Git repository directly, so export remains explicit.

## Pipeline publication change

The automatic test workflow must publish the ten raw test images and a copy of the reference collection into `test/latest/`.

In `Prepare test publication`, create:

```bash
mkdir -p \
  "$LATEST/raw" \
  "$LATEST/metadata" \
  "$LATEST/analysis" \
  "$HISTORY"
```

Then copy:

```bash
cp -a "$OUTPUT_DIRECTORY/raw/." "$LATEST/raw/"

cp hth-pipeline/config/golden_set.json \
   "$LATEST/reference_collection.json"
```

Only the small ten-page test raw set should be committed. Do not publish all 928 production images into the lightweight results repository unless that policy is reconsidered deliberately.

## Why both repositories are still involved

- The results repository is the convenient review workspace.
- The pipeline repository owns the approved reference collection used by CI.
- The editor exports an updated file that is committed back into `hth/config/golden_set.json`.
