# Document normalization

`HTH normalize HTH-GOLDEN-0002` is the bounded validation workflow for HTH's canonical
normalization recipe. It applies either the persisted prepared recommendation
or the explicit `axis-aligned-only` fallback to the 18 immutable
`HTH-GOLDEN-0002` images.

`HTH normalize collection` is the production-scale continuation. Its default
is the latest compatible prepared recommendation; a researcher may select
`axis-aligned-only` to preserve the crop without deskew. The current San
Antonio collection has 929 pages.

The base policy takes the enclosing axis-aligned rectangle of the preferred
document detector's stored quadrilateral. It performs a pixel slice only and
writes the result as lossless PNG. A compatible prepared recommendation may
then apply conservative expanded-canvas Hough deskew only to pages that pass
every recorded safety gate. It never changes gross orientation, performs
perspective warping, resizes, enhances, or binarizes the source image.

## Researcher workflow

From an existing canonical crop, this stage has two human actions:

1. Run **HTH normalize prepare recommendation**. Assessment, recommendation,
   persistence, and review packaging happen in that one action.
2. Review the plain-language result and run **HTH normalize collection**. Keep
   the default prepared recommendation to approve it, or select
   `axis-aligned-only` to preserve pixels after cropping.

No artifact paths, evidence identities, policy files, or command-line options
must be transferred between those actions. The approval boundary remains
because deskew resamples archival pixels. This deliberately avoids using the
current multi-step Golden Set creation flow as the usability model; that flow
has its own simplification work to do.

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
review-manifest.json
summary.md
index.html
normalized/
contact-sheets/
```

The normalization manifest records the half-open crop coordinates, source and
output dimensions, byte and pixel hashes, detector identity, canonical
preprocess identities, normalization policy identity, and canonical
normalization-result identity. Each PNG is decoded after writing and compared
pixel-for-pixel with the in-memory crop. Review cadence and contact-sheet
membership live separately in `review-manifest.json`; they are not canonical
normalization results.

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
the pixel-affecting normalization engine, and runtime contract. The separate
review/report renderer is deliberately excluded: changing contact sheets, HTML,
CSV formatting, Markdown, or review cadence cannot invalidate proven normalized
pixels. `auto` therefore has
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
pages, every 25th page, and every page on which a pixel-changing transform was
actually applied. This keeps regular review bounded while ensuring the changed
pages are never hidden between sampling intervals.

Neither workflow reruns document detection or mutates preprocess, calibration,
or runtime intelligence.

The orientation preparation and recommendation stage is documented in
[Orientation and deskew assessment](orientation-deskew-assessment.md). It
reconstructs and proves only a stratified sample of these canonical crops,
then compares gross-orientation views and two conservative small-angle deskew
estimators and preserves a compatible recommendation without applying it.

The next independent stage is documented in
[Perspective assessment](perspective-assessment.md). It measures line-family
convergence on a proven stratified sample and persists an explicit `preserve`
or `withhold` recommendation. A `preserve` recommendation is positive evidence
that projective correction should be skipped; it does not silently omit the
stage and it does not change normalized pixels.

The following independent stage is documented in
[Photometric assessment](photometric-assessment.md). It measures bright-paper
background uniformity, usable tonal range, endpoint clipping, and spatial color
variation. It consumes the compatible perspective-preservation decision and
persists its own `preserve` or `withhold` recommendation without changing
pixels.

Binarization remains a separate future normalization operation with its own
evidence contract.
