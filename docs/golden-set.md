# HTH Stage 2 Golden Test Set

The golden set is a small, manually approved set of representative images used to detect regressions in Stage 2 document and layout analysis.

## Install

Place the files at:

```text
config/golden_set.json
hth/validate_golden_set.py
README-golden-set.md
```

## Choose representative pages

Include examples of:

- clean and difficult two-page spreads
- a single manuscript page
- title and index sheets
- blank or nearly blank pages
- damaged pages
- overlays or pasted slips
- marginalia, signatures, seals, or unfamiliar marks
- early, middle, and late capture styles

Start with 10–20 pages. Add another page only when it represents a new failure class.

## Coordinates

Bounding boxes use original-image coordinates:

```text
[left, top, right, bottom]
```

Populate `physical_document_bbox` only after manually approving the correct boundary. Entries with `null` boxes are skipped by validation.

## Validation command

```bash
python hth-pipeline/hth/validate_golden_set.py \
  --golden hth-pipeline/config/golden_set.json \
  --analysis "$OUTPUT_DIRECTORY/page-analysis/page-analysis.json" \
  --output "$OUTPUT_DIRECTORY/page-analysis/golden-set-report.json"
```

## Add to preprocess-test.yml

Place this after Stage 2 analysis:

```yaml
- name: Validate Stage 2 golden set
  shell: bash
  run: |
    set -euo pipefail

    python hth-pipeline/hth/validate_golden_set.py \
      --golden hth-pipeline/config/golden_set.json \
      --analysis "$OUTPUT_DIRECTORY/page-analysis/page-analysis.json" \
      --output "$OUTPUT_DIRECTORY/page-analysis/golden-set-report.json"
```

## Acceptance defaults

- Intersection-over-union at least `0.95`
- Maximum edge error `20 px`
- Exact layout-class match

The golden set is a regression suite, not a manual annotation of all 928 images. Pass 1 analyzes everything automatically; Pass 2 adds overrides only for review exceptions.
