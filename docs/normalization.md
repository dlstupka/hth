# Document normalization

`HTH normalize collection` is the single production entry point. Its default
is the compatible recommendation prepared within the orchestration; a researcher may select
`axis-aligned-only` to preserve the crop without deskew. The current San
Antonio collection has 929 pages.

## Automation contract

Normalization is evidence-driven and automation-first. A deterministic safe
path must advance without page-by-page selection or an operator relaying
identities, artifacts, or commands between stages. Every automatic decision is
still fingerprinted, persisted, and rendered for audit. Human attention is
reserved for the bounded exception set: failed safety gates, inconclusive
evidence, or explicit review routes. A small exception population—especially
below one percent of the collection—is a review queue, not a reason to make the
whole collection interactive.

Downstream stages must revalidate upstream identities and hard eligibility
invariants rather than trusting labels alone. They fail closed when evidence is
missing or contradictory; they do not silently widen the exception set or
apply a transform to an ineligible page.

A page-level normalization miss is non-blocking. The page retains its latest
canonical pixels, records the reason, and continues to downstream recognition.
Only invalid provenance, contradictory evidence, or an unreadable canonical
input stops the pipeline. Optional normalization quality must not prevent a
meaningful later HTR result.

The base policy takes the enclosing axis-aligned rectangle of the preferred
document detector's stored quadrilateral. It performs a pixel slice only and
writes the result as lossless PNG. A compatible prepared recommendation may
then apply conservative expanded-canvas Hough deskew only to pages that pass
every recorded safety gate. It never changes gross orientation, performs
perspective warping, resizes, enhances, or binarizes the source image.

## Current researcher workflow

From an existing canonical crop, run **HTH normalize collection**. Keep the
default prepared recommendation to permit evidence-backed conservative deskew,
or select `axis-aligned-only` to preserve pixels after cropping. Assessment,
recommendation, validation, persistence, and downstream integration remain
separate internal jobs within that orchestration.

No artifact paths, evidence identities, policy files, or command-line options
must be transferred between actions. The explicit recipe selection remains the
approval boundary because deskew resamples archival pixels. This deliberately
avoids using the current multi-step Golden Set creation flow as the usability
model; that flow has its own simplification work to do.

## Canonical preprocess handoff

Normalization does not run a document detector. The workflow checks out the
published image manifest, page analysis, and Canonical Build Evidence from the
collection results repository. It validates the authoritative evidence record
and every canonical published JSON artifact before processing.

For every Golden Set page, the immutable release image SHA-256 must agree with
all three existing authorities:

- the approved Golden Set image identity;
- the canonical preprocess image manifest; and
- the corresponding Canonical Build Evidence page record.

The crop is rejected if any identity differs or the stored preferred detector
geometry is missing. This makes normalization a strict downstream consumer of
the proven preprocess result rather than a second inference path.

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

When deterministic method assessment and complete held-out validation pass,
the production integration described in
[Photometric method assessment and validation](photometric-method-assessment.md#production-integration)
publishes a new immutable Results release. It selectively applies only the
validated illumination/background-field correction. Corrected and preserved
pages are both annotated and continue as downstream HTR inputs; optional
normalization quality never removes a page from the transcription pipeline.

Photometric integration uses the `hth-photometric-integration` Canonical Build
Evidence scope and the standard `auto`, `audit`, `force-verify`, and `rebuild`
policies. Its first run establishes page-complete evidence for the immutable
release. Thereafter, `auto` validates and reuses an exact result without source
download, page reconstruction, correction, or repackaging; `audit` additionally
verifies the persisted release asset digest without rebuilding it.

The subsequent restoration families are documented in
[Denoising, artifact suppression, and detail enhancement](restoration-normalization.md).
They run two complete four-stage sequences: denoising first, then sharpening,
with a separately reusable immutable result at each integration boundary.

Binarization is the final normalization family and is documented in
[Binarization normalization](binarization-normalization.md). It runs a complete
four-stage assessment, bounded-method comparison, held-out validation, and
CBE-backed integration sequence after sharpening.
## Full normalization orchestration

`HTH normalize collection` is the single production entry point and the full
normalization regression. It reassesses crop/framing and orientation/deskew,
builds or reuses canonical geometric normalization, assesses perspective, runs
the complete photometric method-selection and integration sequence, then runs
the contrast/tonal and chromatic four-stage families, denoising/artifact
suppression, sharpening/detail enhancement, and final binarization. CBE-enabled
diagnostic and apply stages reuse exact canonical results and audit their
immutable releases. Crop/framing, orientation/deskew, perspective, and all
three pre-integration photometric stages establish the same identity-keyed CBE
records as the later normalization families. Their reuse decision is made from
compact authoritative manifests, policies, detector selection, configuration,
implementation, and runtime identities before any source-image download,
materialization, detector inference, or page evaluation.

This is a development contract, not a legacy-evidence adapter. The first
`auto` run after a stage adopts CBE executes once to seed its current contract;
older artifacts without that exact identity are not promoted into reusable
evidence. Subsequent unchanged runs may reuse the seeded result, and `audit`
requires that exact persisted evidence to exist.

The `start_stage` menu supports development and recovery without creating an
alternative production path. Selecting a stage skips earlier jobs and runs that
stage plus every downstream stage. Skipped prerequisites must already have
valid persisted evidence; missing or incompatible evidence fails closed. The
internal stage workflows remain independently testable but are invoked only by
the collection orchestrator. A skipped dependency is accepted only when the
selected `start_stage` intentionally enters at the dependent stage; a
failure-caused skip stops the downstream graph instead of silently consuming
an older result.

The optional complete normalized-image artifact is disabled by default because
requesting an ephemeral copy necessarily forces canonical normalization to
execute even when its durable evidence is exact. Enable `upload_full_artifact`
only when that review artifact is itself required. Canonical normalization
publication overlays its own rebuilt artifacts while preserving the complete
identity-keyed evidence history for every downstream stage.

After a successful complete graph, the orchestrator inventories durable result,
model-mirror, and release-backed learned-evidence-cache releases and publishes
`metadata/resource-lifecycle.json`. The report connects
every CBE build to its evidence-cache lookup and immutable release use, marks
superseded cache/release elements dirty, protects anything still referenced by
authoritative or historical audit evidence, and identifies only truly
unreferenced inventoried releases as cleanup eligible. Learned-evidence cache
releases are labeled but remain protected until their own utilization ledger
can prove cleanup eligibility. The report records eligibility; it does not
perform cleanup.

The same final audit job renders and persists
`reports/full-normalization-summary.md` from the exact results snapshot. The
normalization build appends that persisted file to its Actions summary; the
manual `HTH report` workflow's `full-normalization-summary` choice uses the
same `hth.report_generator` command and renderer. The canonical sections cover
the seven transformation outcomes, every authoritative normalization CBE
scope, source-release utilization, and the resource-lifecycle ledger. Missing
or malformed durable inputs produce an explicit `INCOMPLETE` report rather
than inferred success. Recommendations appear only when the evidence supports
a specific action, and neither reporting path recomputes pixels or performs
cleanup.
