# Binarization normalization

HTH treats binarization as the final optional pixel-normalization family. It
consumes the immutable sharpening result and emits a separately reusable
immutable collection. Every page continues downstream. Pages that cannot be
shown safe retain their exact sharpening-stage pixels.

The family follows the standard four-stage contract:

1. **Assessment** measures luminance span, Otsu foreground coverage, and
   foreground/background contrast for every page. Implausible coverage or weak
   separation is routed to review instead of being transformed.
2. **Bounded method comparison** evaluates global Otsu, adaptive Gaussian, and
   Sauvola candidates on the deterministic development partition.
3. **Held-out validation** requires the selected method to pass every safety
   gate on every held-out correction candidate.
4. **Integration** applies the method only after complete validation, proves
   the generated pixel identities, packages a lossless immutable release, and
   publishes Canonical Build Evidence.

Safety evidence includes foreground agreement with the measured source
structure, output foreground coverage, connected-component inflation, and edge
correlation. A method must be safe on every development candidate and every
held-out candidate. Otherwise integration selects `preserve` for the complete
collection; no page is dropped.

The scopes are `hth-binarization-assessment`,
`hth-binarization-method-assessment`, `hth-binarization-validation`, and
`hth-binarization-integration`. The integration release tag is
`HTH-BINARIZATION-<binarization-result-identity>`. Job summaries link the
immutable sharpening input and the durable binarization result, including the
release asset digest, cache-resolution source, Results commit, and original
build-stage time.

`HTH normalize collection` runs this sequence after sharpening. Its
`start_stage` menu can begin at any of the four binarization stages for focused
development or recovery, while still preserving the single production path.
