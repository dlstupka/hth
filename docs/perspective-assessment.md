# Perspective assessment

`HTH assess perspective` is an independent diagnostic stage after canonical
crop and conservative deskew. It answers a narrow question: does a stratified
sample contain credible projective convergence that justifies evaluating a
perspective warp?

The expected answer for a flat, consistently photographed collection is
`preserve`. That is a measured recommendation, not an assumption or a missing
pipeline step.

## Researcher workflow

Run **HTH assess perspective** and review its plain-language recommendation.
The workflow automatically selects the sample, reconstructs the corresponding
canonical normalized pages, verifies their stored pixel hashes, evaluates the
evidence, persists the compact result, and uploads contact sheets. Researchers
do not need to transfer artifact paths or evidence identifiers.

The stage never modifies production pixels. `skip-recommended` means keep the
current canonical normalization. `manual-review-required` means withhold any
projective correction until the focused evidence has been reviewed and a
separate correction policy has been developed.

## Independent evidence

The selected document detector cannot answer this question by itself. Its
stored quadrilaterals are effectively rectangular envelopes, so their geometry
is excellent crop evidence but cannot demonstrate the absence of keystone
distortion inside the page.

The perspective assessment therefore detects near-horizontal and near-vertical
line families directly in the normalized image. It fits angle change across
each family and requires independently confident convergence in both axes
before calling a page a correction candidate. A signal from only one family is
sent to review because handwriting, ledger rules, and page layout can imitate
one-axis convergence.

The recommendation gates require:

- a sufficient stratified sample;
- correction-candidate prevalence below the configured skip limit; and
- enough conclusive pages to support a collection-level decision.

All thresholds and sampling rules live in
`config/perspective-assessment.json`; none are collection-specific code.

## Durable evidence

The results repository receives compact material under
`normalization/perspective/` plus the current
`normalization/perspective-policy.json`. The assessment identity fingerprints
the canonical normalization result, sample identity, configuration, and every
page result. Recommendation generation rejects altered or mismatched evidence.

The temporary review artifact contains `assessment.json`, `assessment.csv`,
`perspective-policy.json`, the sample and materialization evidence, Markdown
summaries, `index.html`, and bounded contact sheets. Full-resolution normalized
images are not duplicated in the results repository.
