# Layout reconnaissance: frozen Golden Sets

This is a bounded layout evaluation, not a production layout stage or a model
selection. For GS0002, `tools/layout-smoke-inputs.py` reconstructs the 18 source
and final normalized page views from the frozen Golden Set and pinned Results manifests.
It verifies the frozen Golden Set, each source file and source pixel digest,
the canonical crop pixels, and each final normalized pixel digest before
emitting a pair. A later layout stage can therefore compare input views by
page identity without silently substituting a visually similar image. The
source view is the original/pre-normalization page; the normalized view is
the final persisted normalization result.

The manual `HTH layout` Actions workflow selects an immutable Golden Set and
a `smoke` or `full` mode. For HTH-GOLDEN-0002, it verifies source and final
normalized views against the selected Results ref, then invokes managed Kraken
on both views. The older HTH-GOLDEN-0001 uses a different source release and
is evaluated source-only; the summary explicitly says so rather than implying
an unverified normalized comparison. `smoke` evaluates every page in the
selected immutable Golden Set: 18 of 18 for GS0002 and 5 of 5 for GS0001.
`full` means the complete 929-page collection, as in the upstream flows. That
collection materialization is not wired into this workflow yet, so `full` fails
at scope validation rather than silently running only the Golden Set. The smoke
uploads compact native geometry, per-page health
metrics, batch timings/warning counts, logs, and provenance as an artifact.
It does not publish to the Results repository or upload the source/normalized
pixel bundles; those are reproducible from the frozen release and manifests.

`tools/layout-smoke-report.py` summarizes Kraken's native baseline output for
both views. Native polygons and baselines remain the research output; the
summary carries only provenance and diagnostics that answer a decision. The
evaluation deliberately does **not** report an accuracy score: GS0002 has approved
page geometry, not line, region, or reading-order ground truth. Line-count
changes between views are review triggers, not evidence that either view is
better.

## Metrics worth retaining

| Signal | Decision it supports |
| --- | --- |
| Source release, Golden Set identity, page ordinal, input pixel digest, normalization result identity, model file digest, package version, and inference configuration | Exact provenance, compatible reuse, and cache invalidation. Reuse existing CBE fields rather than copying them into a second authority. |
| Native region polygons, line baselines/boundaries, line-to-region membership, and explicit reading order when available | These are the layout result itself, needed by downstream work and review. Preserve view coordinate system and transformation identity. |
| Per-page status and categorized warnings/errors, empty output, invalid or out-of-bounds geometry, orphaned line associations, and missing/incomplete reading order | Catch plausible-looking but unusable output and direct failure review. Preserve the category/count, not unbounded logs. |
| Per-page region/line counts and paired-view deltas | Prioritize visual review and flag suspicious splits, merges, or dropped content; **not** quality scores. |
| Model-load time, per-page inference time, batch wall time, and peak worker memory (in a production runner) | Choose batch size, executor shape, and capacity; do not persist arbitrary internal timings. |
| CBE hit/miss and rebuild reason (in a production runner) | Verify reuse and explain performance regressions without reconstructing cache decisions from logs. |
| Region-class match/IoU, baseline match precision/recall at a declared tolerance, split/merge/miss counts, and reading-order accuracy **only after line/region truth exists** | Compare models and guard quality regressions. Always retain the truth-set and matching-policy identity with the score. |

Do not retain model activations, full probability maps, arbitrary confidence
histograms, or OCR/HTR text for this stage. Native geometry is necessary;
large intermediates are temporary diagnostics unless a concrete decision
depends on them. Cache expensive inference by input *pixel* digest plus model,
runtime, and configuration identity, while preserving separate view provenance.
Native Kraken JSON uses generated IDs, so its raw byte hash is not a suitable
semantic reuse key. Stable geometry identity needs canonicalization before a
production cache is introduced. Repeated inference on page 381 produced the
same line geometry but different native JSON byte hashes, confirming this is
not merely theoretical.

If Kraken survives evaluation, its model acquisition/load and one-time
per-image inference should be shared with HTH's existing Kraken page-mask
adapter. The page-envelope post-processing is not a layout contract and must
not become one by reuse. Similarly, source and normalized views should be
separate logical inputs backed by the same reusable execution/evidence
machinery, not two copied workflows.

## Current smoke

The candidate is Kraken's bundled generic BLLA segmenter, producing native
line baselines and text-region polygons on each paired page. This is a
reconnaissance candidate, not a canonical HTH runtime dependency: HTH's
existing `kraken_page_mask` adapter pins Kraken 7.0.2 for page-envelope
detection; this smoke uses isolated Kraken 7.1.1 and does not change that
adapter or production approval. The detailed native outputs and image overlays
belong in local temporary artifacts, not the Results repository.
The Actions workflow uses managed Kraken 7.0.2, so its outputs must be
evaluated as a separate runtime variant rather than assumed identical to the
local 7.1.1 numbers below.

The smoke used the bundled model with SHA-256
`77a638a83c9e535620827a09e410ed36391e9e8e8126d5796a0f15b978186056`,
`segment -bl`, CPU, and four threads per concurrent view. The pinned Results
snapshot was `a9b7ba081c21311c4da0b81a2ee3597048e3b113`. All 18 paired inputs
passed source and normalized pixel-digest verification; 3 pages took the
photometric `apply` route and 15 preserved the canonical crop pixels.

| Smoke diagnostic | Source | Final normalized |
| --- | ---: | ---: |
| Pages with at least one baseline | 18/18 | 18/18 |
| Baselines | 1,522 | 1,461 |
| Median baselines per page | 104 | 102.5 |
| Text regions | 78 | 81 |
| Invalid line / region geometry | 0 / 0 | 0 / 0 |
| Lines with no valid region association | 1 | 1 |
| Reading order emitted by Kraken `segment -bl` | No | No |

Fourteen paired pages changed baseline count. The largest drops after
normalization were pages 197 (-26, photometric correction) and 155 (-25, crop
only); these are **review priorities**, not proof of improvement. Native
polygonizer warnings occurred on source pages 500 and 920 and normalized page
920, even though all images produced output. Sample overlay review found some
lines on page edges and large text regions that merge distinct entries; those
are substantive limitations for record-level layout. The generic model emits
only `text` regions here and no reading order. Therefore the smoke passes
basic output/geometry health but does not establish layout correctness.

The two 18-page CPU batches ran concurrently on a local Windows machine and
took roughly 20 minutes each. This is not a controlled capacity benchmark.
Before productionizing, measure model-load time, per-page inference time,
batch wall time, and peak worker memory on the intended runner, then decide
whether batching, precomputed evidence, or a different model has worthwhile
throughput ROI.

The next acceptance step requires representative line/region annotations and
explicit reading-order truth, followed by a bounded comparison with other
historical-page layout candidates. Until then, geometry sanity and manual
review may reject a candidate, but cannot establish a winner.
