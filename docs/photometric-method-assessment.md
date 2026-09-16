# Photometric method assessment

`HTH assess photometric methods` turns a photometric `withhold` result into a
small, deterministic method-selection experiment. It consumes only persisted
`paper-page` correction candidates; preserved dark frames, mixed-polarity
pages, and ordinary review pages cannot enter the experiment by accident.

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
