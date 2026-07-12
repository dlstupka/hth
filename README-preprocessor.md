# HTH Preprocessor v0.1

This tool performs the mechanical preprocessing needed before historical analysis.

## It does both, but non-destructively

### Metadata
It extracts every embedded image from the DOCX masters in document order and records:

- source DOCX and image ordinal,
- relationship/media path,
- pixel dimensions and image format,
- byte size and SHA-256,
- exact duplicate groups,
- empty Pass-2 fields for manuscript pages, dates, record numbers, transitions, priests/hands, condition, confidence, and research notes.

Outputs:

- `metadata/image_manifest.csv`
- `metadata/image_manifest.json`
- `metadata/page_map_template.csv`
- `metadata/exact_duplicates.json`
- `summary.json`

### Images
It preserves an exact-byte extracted copy in `raw/`.

With `--derive`, it also creates separate analysis PNGs using grayscale and autocontrast. These are derivative working images only; the DOCX and raw extracted image are never changed.

It also creates thumbnails and, with `--contact-sheets`, labeled review sheets for quickly finding page-number runs, year transitions, blank/obstructed pages, volume boundaries, and capture errors.

## It does not yet pretend to read the handwriting

Version 0.1 intentionally does not auto-fill names, dates, priests, or manuscript page numbers. Those fields exist in the map template, but weak OCR guesses are not promoted to historical facts.

## Local Windows use

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python tools\preprocess.py `
  --input data\source `
  --output build\preprocessed `
  --config config\preprocess.json `
  --derive `
  --contact-sheets `
  --overwrite
```

## Overlapping/replacement DOCX files

Use `config/preprocess.json` to specify `skip_first`, `skip_last`, and `global_start` for overlapping captures such as the first 100 images.

## GitHub Actions

The included workflow runs the same preprocessing and uploads the result as a GitHub Actions artifact. It does not commit generated images back into Git history.

## Large source files

Use Git LFS for DOCX masters over normal GitHub file limits, or run locally and commit only the code and generated metadata.
