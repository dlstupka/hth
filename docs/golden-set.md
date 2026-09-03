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

HTH-0001 follows the same release contract retroactively. Its canonical source-repository release is `HTH-GOLDEN-0001`, containing `HTH-0001.golden-set.json` and `HTH-0001.freeze.json`. The Golden Set asset must contain the exact existing `config/golden_set.json` bytes (SHA-256 `135c0ff576876ef8911296e2502193ed20d159799079a4f8a58994854fcbba8e`); do not re-export, reformat, or add approval fields to that legacy JSON. The freeze manifest records this canonical release identity.

## Golden Set ownership and releases

Golden Sets are curated claims about a particular source corpus, so the canonical copy belongs in that source repository. Publish each approved set as a separate immutable source-repository release (for example, release/tag `HTH-GOLDEN-0002`) with these assets:

- `HTH-0002.golden-set.json`, exported by the multidetector reference editor;
- `HTH-0002.freeze.json`, exported immediately afterward from the same approved editor state; and
- an optional human-readable review note or screenshot bundle.

Never replace assets on an established Golden Set release. A membership, approved geometry, acceptance threshold, or source-identity change creates a new Golden Set ID and release. Editorial notes that do not affect the JSON may be added to the release description without changing its identity.

The pipeline repository vendors the exact released JSON and freeze manifest under `config/golden_sets/`. CI validates every `*.freeze.json` file against its pinned Golden Set bytes. `config/golden_set.json` remains the active/default compatibility path used by existing workflows; changing that pointer does not delete or mutate older versioned sets. Calibration and runtime evidence continue to key on both Golden Set ID and SHA-256.

This division keeps responsibility clear:

- source repository: canonical approval record and immutable Golden Set release;
- pipeline repository: pinned copies, schemas, validators, and the active default;
- results repository: run evidence referring to the exact Golden Set ID and hash.

## Creating HTH-0002 from SOURCE-0002

1. Expand the SOURCE-0002 production review artifact and open `tools/reference-collection-editor-multidetector.html`.
2. Choose **Open results workspace** and select the expanded artifact directory.
3. Under **Image selection**, choose **Explicit ordinals**, enter the proposed ordinals, and choose **Apply view**.
4. Choose **Replace membership with visible**. This makes the visible list the complete set membership; source collections otherwise start with no calibration pages selected.
5. Review every selected image, correct its box/layout metadata, and choose the per-page **Approve** action.
6. Enter `HTH-0002` plus creator, reviewer, approver, and change-note provenance. Choose **Approve current state**.
7. Export the Golden Set and then its freeze manifest without changing the editor state between exports.
8. Publish both files in an immutable `HTH-GOLDEN-0002` release in the SOURCE-0002 repository, then vendor those exact files into `config/golden_sets/` here.

The proposed SOURCE-0002 transfer-validation selection is:

```text
3, 64, 65, 95, 100, 101, 155, 197, 251, 298, 300, 308,
367, 380, 381, 400, 500, 676, 700, 710, 711, 728, 821, 920
```

Pages 1, 2, and 929 are collection/roll frames and are intentionally excluded from this initial set. This selection validates transfer of the existing approved Fusion Gen3 calibration; it does not itself authorize recalibration.

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
