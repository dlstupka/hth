# Denoising, artifact suppression, and detail enhancement

HTH treats denoising and sharpening as two independent, ordered normalization
families. Denoising consumes the immutable chromatic-normalized collection;
sharpening consumes the immutable denoising result. Every page continues to the
next stage. When evidence does not justify a safe transform, the original pixels
are copied losslessly and the manifest records `preserve-and-continue`.

Each family uses the same four-stage contract:

1. **Assess** every page and route it to correction, preservation, or review.
2. **Compare bounded methods** on the deterministic development partition.
3. **Validate** the selected method on every held-out candidate.
4. **Integrate with CBE**, reproducing validated outputs, preserving all other
   pages, and publishing or auditing an immutable release.

The normal ordinal partition remains stable for ordinary populations. If a
sparse candidate set of at least two pages happens to land entirely on one
side, assessment deterministically moves the lowest or highest candidate needed
to guarantee both development and held-out evidence. Review and preservation
routes are never moved, and partition balancing never relaxes a safety gate.

## Denoising and artifact suppression

Assessment measures robust high-frequency noise, impulse deviation, edge
density, detail energy, luminance, and clipping. The bounded comparison covers
edge-preserving bilateral filters and a median impulse filter. Safety gates
require measurable noise reduction while preserving local detail, edges,
luminance, and clipping behavior. A method must be safe for every development
candidate and every held-out candidate before any page is changed.

## Sharpening and detail enhancement

Sharpening runs only after denoising so residual noise can be measured before a
detail transform is considered. Assessment withholds noisy pages from automatic
enhancement and identifies low-detail candidates. The bounded comparison uses
conservative unsharp-mask strengths. Gates bound detail gain, noise
amplification, clipping, luminance shift, and detail correlation. If no method
is globally safe, all pages are preserved.

## Provenance, reuse, and diagnostics

The eight CBE scopes are independent: assessment, method assessment,
validation, and integration for each family. Effective identities include the
immutable upstream release record and manifest, relevant configuration,
pixel-affecting implementation, and runtime contract. Reporting-only changes do
not invalidate image results.

The shared immutable-release restore action checks the runner-local release
cache first, then GitHub's cache, then the durable Results release. Every source
is verified against the recorded SHA-256 before extraction. Exact CBE matches
reuse the compact evidence and audit the durable release without image work.

Integration summaries identify and link the result identity and release, report
release activity and Results commit, and retain the original build-stage time.
Comparison summaries expose a compact method table with safe-page counts,
measured gains, correlations, and failed gates.

`HTH normalize collection` runs both complete four-stage sequences by default.
Its `start_stage` menu can begin at any of the eight stages for development or
recovery; persisted prerequisites remain mandatory. The individual assess,
compare, and validate workflows are diagnostic entry points, while the generic
integration workflow supports each normalization domain.

The immutable sharpening result is consumed by the final
[binarization normalization](binarization-normalization.md) family.
