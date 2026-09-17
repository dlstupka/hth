# Photometric method assessment

`HTH normalize photometric method assessment` turns a photometric `withhold` result into a
small, deterministic method-selection experiment. It consumes only persisted
`paper-page` correction candidates; preserved dark frames, mixed-polarity
pages, and ordinary review pages cannot enter the experiment by accident.
Persisted candidates must also carry affirmative geometry eligibility and an
empty exclusion list. The workflow rejects the entire candidate plan if any
candidate violates that contract.

The workflow is diagnostic. It does not change the canonical normalized
collection or authorize a full-collection photometric transform.

## Automated workflow

The action validates the canonical normalization evidence, the complete
photometric assessment, its `manual-review-required` policy, and immutable
source provenance. It then:

1. derives the exact candidate list from fingerprinted photometric evidence;
2. reconstructs those canonical normalized pages and verifies their stored
   pixel hashes;
3. applies the fixed variants in
   `config/photometric-method-assessment.json`;
4. evaluates every output with common safety gates; and
5. persists a machine-readable validation recommendation and a visual artifact.

There is no interactive page selection, random sampling, or environment-based
method choice. Identical inputs and configuration produce identical candidate
plans, output pixels, measurements, rankings, and identities.
Review is limited to the exception evidence and generated comparison artifact;
it is not required to construct or relay the candidate list.

## Bounded methods and safety gates

The implemented family estimates a smooth paper-background field on the L
channel of CIE LAB and changes luminance only. Fixed subtraction and division
variants are compared at configured strengths. The original color channels
are reconstructed without applying sharpening, denoising, binarization, or a
global contrast curve.

A variant is safe for a candidate only when it meets every configured gate:

- minimum reduction in spatial background span;
- minimum high-frequency correlation with the canonical normalized page;
- maximum new endpoint clipping; and
- maximum absolute median-luminance shift.

The workflow recommends one method only if the same configured method passes
every gate on every persisted candidate. Among those methods it chooses the
lowest correction strength, then the highest explicit score, then the stable
method ID. This minimum-intervention rule prevents a stronger transform from
winning merely because it flattens more background. If no common method passes,
the action persists a `withhold` decision and preserves current pixels.

## Evidence and next decision

The results repository receives compact evidence in
`normalization/photometric-methods/` and the current
`normalization/photometric-method-policy.json`. Identities bind the canonical
normalization result, upstream photometric assessment, exact candidate plan,
configuration, page measurements, and output pixel hashes. Recommendation
generation rejects altered evidence.

The temporary artifact contains lossless output variants and contact sheets.
A `validation-candidate` result means the method is ready for an explicit
validation run over a larger held-out sample. It does not yet mean that the
method should be applied to the full collection.

## Complete held-out validation

`HTH normalize photometric method validation` consumes the persisted validation candidate
and audits every canonical normalized page that was not used to select the
method. The development candidates are excluded by identity rather than by a
manually maintained page list. The workflow reconstructs and hash-verifies the
complete held-out population, reruns the photometric classifier, applies the
fixed method only to newly discovered paper-page candidates, and preserves all
other pixels.

The integration decision requires full held-out coverage, enough newly
discovered targets, every target passing every method safety gate, sufficient
mean background improvement, and byte-equivalent preservation of every dark or
mixed-polarity control. A single unsafe target blocks collection-wide
integration. If the complete held-out population contains too few targets,
the result is `integration-not-justified`; current pixels remain authoritative.

Compact evidence is persisted in `normalization/photometric-validation/` and
the current decision in `normalization/photometric-integration-policy.json`.
Only discovered targets and preselected controls receive review images, which
keeps the temporary artifact practical while the machine-readable audit still
covers every held-out page.

## Production integration

`HTH normalize photometric integration` is the automated apply stage. It runs
only when the persisted method policy and complete held-out validation both
authorize the same method. It joins the development and held-out evidence into
an exact, non-overlapping partition of the canonical collection; a missing,
duplicated, or unaccounted page stops publication.

For every page, the workflow reconstructs and hash-verifies the canonical
normalized input, reruns the deterministic classifier, and requires its route
to reproduce the persisted assessment or validation route. Eligible candidates
receive the selected illumination/background-field method and must reproduce
the previously validated output pixel hash. All other pages remain byte-level
pixel equivalent. Every manifest row records either
`corrected-and-continue` or `preserve-and-continue`, so an excluded or
uncorrected page remains available to later HTR rather than becoming a pipeline
failure.

The resulting lossless PNG collection is packaged as a deterministic Zip64
asset in an immutable Results repository release tagged
`HTH-PHOTOMETRIC-<result-identity>`. Compact manifests, the applied policy,
source materialization evidence, release URL, asset SHA-256, and result identity
are persisted under `normalization/photometric-integration/`. An exact rerun
rebuilds the same bytes and reuses the release only after the remote asset
digest matches. The integrated operation changes spatial illumination and
paper-background uniformity only; it does not add a global contrast curve,
tonal-range normalization, sharpening, denoising, or binarization.
