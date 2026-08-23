# HTH Stage 2 Golden Test Set

The golden set is a small, manually approved set of representative images used to detect regressions in Stage 2 document and layout analysis.

## Install

Place the files at:

```text
config/golden_set.json
hth/validate_golden_set.py
README-golden-set.md
```


## Frozen Golden Set identity

`HTH-0001` is the original five-page calibration/validation set (global ordinals 1, 5, 6, 9, and 10) and is now frozen. Its existing `config/golden_set.json` bytes are intentionally unchanged so all previously persisted calibration evidence remains compatible with the same Golden Set SHA-256.

The external freeze manifest at `config/golden_sets/HTH-0001.freeze.json` records the immutable content hash, membership, source-document identity, and freeze policy. CI runs `hth/validate_golden_set_freeze.py` whenever the Golden Set or freeze metadata changes. Any in-place change to HTH-0001 fails validation; new or corrected truth data must receive a new Golden Set identity such as `HTH-0002`.

HTH-0001 remains a permanent legacy validation subset even after a broader Golden Set is introduced. This preserves the historical meaning of every calibration already keyed to `HTH-0001` while allowing future sets to challenge generalization across a wider corpus.

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

`HTH-0001` intentionally remains the frozen five-page legacy set. A future `HTH-0002` should be evidence-driven rather than expanded by quota: add pages when full-collection inference exposes a distinct failure class, capture regime, detector-disagreement case, or structural layout not represented by the frozen set. Include representative controls as well as difficult pages.

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
