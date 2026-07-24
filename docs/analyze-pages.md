# HTH Stage 2 — Physical Page Analysis

`analyze_pages.py` performs Stage 2 of the Hidden Texas History Research Framework. It reads the Stage 1 image manifest and extracted images, then creates compact page-geometry and image-quality reports for the collection’s `*-results` repository.

It does **not** perform OCR, handwriting recognition, layout segmentation, transcription, translation, or historical interpretation.

## Dependencies

- Python 3.12+
- Pillow
- Python standard library

No change is required to the current `requirements.txt` if it already includes Pillow.

## Input

```text
build/preprocessed/
├── raw/
│   ├── fs_0001.png
│   └── ...
└── metadata/
    └── image_manifest.json
```

## Output

```text
page-analysis/
├── page-analysis.json
├── page-analysis.csv
├── review-queue.csv
├── analysis-summary.json
└── annotated-previews/          # optional
```

The reports include provenance, image dimensions, estimated skew, content bounds, margins, brightness, contrast, sharpness proxy, entropy, background uniformity, a conservative bleed-through proxy, quality score, status, and review reasons.

## Local example

```bash
python hth/analyze_pages.py \
  --manifest build/preprocessed/metadata/image_manifest.json \
  --image-root build/preprocessed \
  --output build/preprocessed/page-analysis \
  --annotated-previews \
  --overwrite
```

## GitHub Actions integration

Place this step immediately after `Run full preprocessor`:

```yaml
- name: Run physical page analysis
  shell: bash
  env:
    SOURCE_REPOSITORY: ${{ inputs.source_repository }}
    SOURCE_COMMIT: ${{ steps.build.outputs.source_commit }}
    PIPELINE_REPOSITORY: ${{ github.repository }}
    PIPELINE_COMMIT: ${{ steps.build.outputs.pipeline_commit }}
  run: |
    set -euo pipefail

    python hth-pipeline/hth/analyze_pages.py \
      --manifest "$OUTPUT_DIRECTORY/metadata/image_manifest.json" \
      --image-root "$OUTPUT_DIRECTORY" \
      --output "$OUTPUT_DIRECTORY/page-analysis" \
      --source-repository "$SOURCE_REPOSITORY" \
      --source-commit "$SOURCE_COMMIT" \
      --pipeline-repository "$PIPELINE_REPOSITORY" \
      --pipeline-commit "$PIPELINE_COMMIT" \
      --annotated-previews \
      --overwrite
```

Extend build verification:

```bash
test -f "$OUTPUT_DIRECTORY/page-analysis/page-analysis.json"
test -f "$OUTPUT_DIRECTORY/page-analysis/page-analysis.csv"
test -f "$OUTPUT_DIRECTORY/page-analysis/review-queue.csv"
test -f "$OUTPUT_DIRECTORY/page-analysis/analysis-summary.json"
```

Publish the compact reports:

```bash
rm -rf results-repo/analysis
mkdir -p results-repo/analysis

cp "$OUTPUT_DIRECTORY/page-analysis/page-analysis.json" results-repo/analysis/
cp "$OUTPUT_DIRECTORY/page-analysis/page-analysis.csv" results-repo/analysis/
cp "$OUTPUT_DIRECTORY/page-analysis/review-queue.csv" results-repo/analysis/
cp "$OUTPUT_DIRECTORY/page-analysis/analysis-summary.json" results-repo/analysis/
```

Keep `annotated-previews/` in the temporary Actions artifact initially. They can be published later after measuring their size and usefulness.

## Optional configuration

Create `config/analyze-pages.json`:

```json
{
  "max_analysis_dimension": 1800,
  "background_threshold": 238,
  "minimum_component_area_fraction": 0.00002,
  "bbox_padding_fraction": 0.008,
  "skew_search_degrees": 3.0,
  "skew_step_degrees": 0.25,
  "blank_dark_fraction_max": 0.004,
  "low_contrast_stddev": 22.0,
  "blurry_sharpness_threshold": 8.0,
  "overexposed_brightness": 244.0,
  "underexposed_brightness": 65.0,
  "review_quality_threshold": 0.62,
  "fail_quality_threshold": 0.35
}
```

Then add:

```bash
--config hth-pipeline/config/analyze-pages.json
```

The first full run is calibration. Review the highest and lowest values before treating thresholds as archival conclusions.

## Suggested results structure

```text
*-results/
├── BUILD-INFO.yaml
├── metadata/
├── reports/
├── analysis/
│   ├── page-analysis.json
│   ├── page-analysis.csv
│   ├── review-queue.csv
│   └── analysis-summary.json
├── ocr/
├── transcriptions/
├── translations/
├── indexes/
├── citations/
├── research-notes/
└── test/
```

## Interpretation and limitations

- **Content bounds:** a threshold-based first estimate, not a text-layout model.
- **Skew:** small-angle projection analysis; unusual pages may need review.
- **Sharpness:** a collection-relative edge-variance score.
- **Bleed-through:** a triage proxy only, not a physical diagnosis.
- **Orientation:** EXIF orientation is applied; manuscript 90°/180°/270° recognition is intentionally deferred.
- **Quality status:** `pass`, `review`, `fail`, or `error`; human review remains authoritative.

This version does not yet identify gutters, segment records or lines, detect handwriting baselines, perform OCR, or alter source images.
