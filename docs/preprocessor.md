# HTH Preprocessor

This tool performs the mechanical preprocessing needed before historical analysis.

## It does both, but non-destructively

### Metadata
It extracts every embedded image from the DOCX masters in document order and records:

- source DOCX and image ordinal,
- relationship/media path,
- canonical pixel dimensions and image format,
- canonical byte size and SHA-256,
- original embedded byte size and SHA-256, and
- Word DrawingML crop percentages and whether they were applied,
- exact duplicate groups,
- empty Pass-2 fields for manuscript pages, dates, record numbers, transitions, priests/hands, condition, confidence, and research notes.

Outputs:

- `metadata/image_manifest.csv`
- `metadata/image_manifest.json`
- `metadata/page_map_template.csv`
- `metadata/exact_duplicates.json`
- `summary.json`

### Images
It writes the canonical document-view image to `raw/`. When a DOCX image has no DrawingML crop, this remains an exact-byte copy of the embedded image. When Word stores a non-destructive `a:srcRect` crop, preprocessing applies the signed percentage offsets before writing `raw/`; the manifest retains both the embedded-image identity and the canonical output identity.

With `--derive`, it also creates separate analysis PNGs using grayscale and autocontrast. These are derivative working images only; the DOCX is never changed.

It also creates thumbnails and, with `--contact-sheets`, labeled review sheets for quickly finding page-number runs, year transitions, blank/obstructed pages, volume boundaries, and capture errors.

## It does not yet pretend to read the handwriting

The preprocessor intentionally does not auto-fill names, dates, priests, or manuscript page numbers. Those fields exist in the map template, but weak OCR guesses are not promoted to historical facts.

## Source boundary

HTH is a collection-independent framework. Collection source material belongs in a separate source repository or another explicitly supplied external path; the framework repository does not reserve or populate a local source-data tree.

`--input` is therefore required when invoking the preprocessor directly. It may name one DOCX master or a directory containing DOCX masters.

Cloud production and test workflows already materialize the selected external source collection and pass its resolved path explicitly through the shared workflow core.

## Local Windows use

From the HTH framework repository, point the launcher at the external collection source path:

```powershell
.\tools\run-preprocess.ps1 ..\hth-baptisms-san-antonio-1788-1824--1858-1898\images
```

Or invoke the preprocessor directly:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python hth\preprocess.py `
  --input ..\hth-baptisms-san-antonio-1788-1824--1858-1898\images `
  --output build\preprocessed `
  --config config\preprocess.json `
  --derive `
  --contact-sheets `
  --overwrite
```

The San Antonio path above is only an example of the current reference collection. HTH itself does not depend on that repository name or layout beyond receiving a path containing the source masters.

## Overlapping/replacement DOCX files

Use `config/preprocess.json` to specify `skip_first`, `skip_last`, and `global_start` for overlapping captures such as the first 100 images.

## GitHub Actions

The included workflow runs the same preprocessing and uploads the result as a GitHub Actions artifact. It does not commit generated images back into Git history.

## Large source files

HTH source masters are not stored in Git history. Publish each immutable DOCX master as an asset on a versioned GitHub Release in the collection source repository. `source-release-manifest.json` records the size and SHA-256 of every source asset, and cloud preprocessing verifies every download before processing. See `source-releases.md`.
