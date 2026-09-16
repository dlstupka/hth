# Photometric assessment

`HTH assess photometric normalization` evaluates whether the canonical
normalized collection would benefit from illumination, contrast, tonal, or
background-color correction. It is diagnostic only: the workflow never changes
or republishes normalized pixels.

## Researcher workflow

Run **HTH assess photometric normalization** after the perspective assessment
has persisted a compatible `preserve` decision. One action selects a stratified
sample, reconstructs and proves its canonical normalized pixels, measures the
photometric evidence, writes a plain-language recommendation, persists compact
evidence, and uploads review contact sheets.

`skip-recommended` means preserve the existing pixels. `manual-review-required`
means withhold automatic correction while researchers review the affected
pages and a bounded correction policy is developed. It does not authorize a
generic enhancement filter. When credible paper-page candidates exist, run
**HTH assess photometric methods** to compare the bounded methods described in
[Photometric method assessment](photometric-method-assessment.md).

## Measurements

Each sampled page is divided into a regular grid. The estimator uses the bright
portion of every tile as a robust paper-background observation, reducing the
influence of handwriting and printed text. It records:

- spatial background-luminance span;
- usable page tonal span;
- shadow and highlight endpoint occupancy; and
- spatial background-chroma variation.

Before correction candidacy is evaluated, endpoint occupancy assigns a
photometric archetype. Predominantly dark roll targets and end markers are
explicitly preserved as `dark-polarity-frame`. Pages combining a dark insert
with ordinary paper are marked `mixed-polarity-page` and held for visual
review. Only credible `paper-page` images can become automatic correction
candidates.

Dense black ink and bright paper legitimately occupy the luminance endpoints,
so clipping is review evidence but cannot independently trigger a correction
recommendation. A correction candidate requires strong uneven-background,
compressed-content, or spatial color evidence. Configuration and thresholds
are explicit in `config/photometric-assessment.json`.

## Provenance and persistence

The stage validates Canonical Build Evidence, the canonical normalization
result, and the persisted perspective policy before reconstructing source
pixels. Every materialized image must match the stored normalized pixel hash.

The results repository receives `normalization/photometric/` and the current
`normalization/photometric-policy.json`. The assessment identity fingerprints
the normalization result, sample, configuration, and all page measurements;
recommendation generation rejects modified evidence.

The temporary artifact adds visual background maps beside the canonical pages,
plus JSON, CSV, Markdown, and HTML review material. Full normalized images are
not duplicated in the results repository.
