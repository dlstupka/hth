# Document normalization

`HTH normalize GS0002` is the artifact-only validation workflow for HTH's
first production normalization operation. It applies the selected
`axis-aligned-detector-envelope-v1` policy to the 18 immutable
`HTH-GOLDEN-0002` images.

`HTH normalize collection` is the production-scale continuation of that
approved experiment. It applies the identical policy to every canonical image
in the published collection manifest. The current San Antonio collection has
929 pages.

The policy takes the enclosing axis-aligned rectangle of the preferred
document detector's stored quadrilateral. It performs a pixel slice only and
writes the result as lossless PNG. It does not rotate, warp, resize, enhance,
or binarize the source image.

## Canonical preprocess handoff

Normalization does not run a document detector. The workflow checks out the
published image manifest, page analysis, and Canonical Build Evidence from the
collection results repository. It validates the authoritative evidence record
and every canonical published JSON artifact before processing.

For every GS0002 page, the immutable release image SHA-256 must agree with all
three existing authorities:

- the approved Golden Set image identity;
- the canonical preprocess image manifest; and
- the corresponding Canonical Build Evidence page record.

The crop is rejected if any identity differs or the stored preferred detector
geometry is missing. This makes normalization a strict downstream consumer of
the proven preprocess result rather than a second inference path.

## Review artifact

The uploaded `hth-gs0002-normalized-*` artifact contains:

```text
normalization-manifest.json
normalization-manifest.csv
preprocess-evidence.json
geometry-evidence.json
summary.md
index.html
normalized/
contact-sheets/
```

The normalization manifest records the half-open crop coordinates, source and
output dimensions, byte and pixel hashes, detector identity, canonical
preprocess identities, normalization policy identity, and canonical
normalization-result identity. Each PNG is decoded after writing and compared
pixel-for-pixel with the in-memory crop.

## Complete collection workflow

The complete workflow reconstructs canonical source pixels directly from the
immutable `HTH-SOURCE-0002` DOCX release. Reconstruction is deliberately
narrower than preprocessing: it creates no thumbnails, analysis derivatives,
contact-sheet corpus, or detector results. Every reconstructed image must match
the published image manifest and preprocess Canonical Build Evidence before it
can be cropped.

The full normalized image package is a temporary workflow artifact. The
results repository receives only compact durable material under
`normalization/`:

- `normalization-manifest.json` and `.csv`;
- `summary.md`; and
- `canonical-build-evidence.json`.

The evidence scope is `hth-normalization`. Its effective identity fingerprints
the complete canonical preprocess handoff, normalization policy/configuration,
the small implementation closure, and runtime contract. `auto` therefore has
two useful modes:

- with **Upload full artifact** enabled, the workflow reconstructs the package
  and proves it equals any incumbent canonical result;
- with artifact upload disabled, an exact incumbent is reused without source
  download or image processing.

`audit`, `force-verify`, and `rebuild` have the same strict meanings as the
preprocess Canonical Build Evidence policy. Publication uses hardened,
conflict-aware persistence and never stores full-resolution normalized images
in Git.

The complete artifact contains every normalized PNG plus manifests and a
bounded review surface. Contact sheets are generated for the first and last
pages and every 25th page to keep review useful without adding hundreds of
megabytes of duplicate JPEG previews.

Neither workflow reruns document detection or mutates preprocess, calibration,
or runtime intelligence.

The next diagnostic stage is documented in
[Orientation and deskew assessment](orientation-deskew-assessment.md). It
reconstructs and proves only a stratified sample of these canonical crops,
then compares gross-orientation views and two conservative small-angle deskew
estimators without publishing a production transform.

Perspective correction, tonal correction, and binarization remain separate
future normalization operations with independent evidence contracts.
