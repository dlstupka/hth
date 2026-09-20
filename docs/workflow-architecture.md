# HTH Workflow Architecture

## Workflows

Reusable cores:

```text
.github/workflows/_core-hth.yml
.github/workflows/_core-report.yml
.github/workflows/_core-normalization-evidence.yml
```

Thin entry workflows:

```text
.github/workflows/preprocess.yml
.github/workflows/preprocess-test.yml
.github/workflows/generate-report.yml
```

The wrappers select mode, source, publication behavior, retention, validation policy, and runner. Manual core-backed workflows expose the common runner vocabulary and default to GitHub-hosted execution unless a different runner is selected. Preprocessing and persisted-evidence report generation use separate reusable cores so changes to one execution surface cannot accidentally alter the other.

Standalone review, preparation, and normalization workflows:

```text
.github/workflows/assess-crop-framing.yml
.github/workflows/assess-orientation-deskew.yml
.github/workflows/assess-perspective.yml
.github/workflows/assess-photometric.yml
.github/workflows/assess-photometric-methods.yml
.github/workflows/validate-photometric-method.yml
.github/workflows/integrate-photometric.yml
.github/workflows/integrate-normalization.yml
.github/workflows/normalize-gs0002.yml
.github/workflows/normalize.yml
```

`assess-crop-framing.yml` resolves the approved detector, materializes the
immutable GS0002 image bundle, and compares axis-aligned, rotation-crop, and
projective framing. It uploads temporary assessment evidence only; it neither
publishes normalization output nor mutates calibration intelligence.

`assess-orientation-deskew.yml` is presented as **HTH prepare normalization
recommendation**. It validates the persisted canonical crop evidence,
reconstructs only a stratified sample from the immutable source release, and
proves every sampled crop by pixel hash. It compares all four gross-orientation
views plus bounded projection-profile and Hough-line deskew candidates, creates
a plain-language recommendation, and persists its compact evidence and proposed
policy. It does not apply a pixel-changing transform.

`normalize-gs0002.yml` is the bounded validation entry point for the selected
normalization recipe. It validates authoritative Canonical Build Evidence and
the published preprocess artifacts, materializes immutable GS0002 source
images, applies the stored preferred-detector geometry, and optionally validates
the persisted prepared deskew recommendation. It does not rerun detector
inference or publish normalized collection assets.

`normalize.yml` applies the selected recipe to the complete canonical
collection. It reconstructs manifest-verified pixels from the immutable source
release, consumes stored preferred-detector geometry and any compatible
prepared transform recommendation, publishes compact
`hth-normalization` Canonical Build Evidence, and optionally uploads the full
normalized PNG package. Exact evidence-only reruns reuse the canonical result
without downloading or processing collection images.

The full normalization entry point owns the collection routing contract through
its `results_repository` input and passes that exact value to every reusable
stage. Tonal, chromatic, denoising, sharpening, and binarization evidence share
`_core-normalization-evidence.yml`; their terminal stages share
`integrate-normalization.yml`. Domain-specific Python implementations and
configuration remain distinct while checkout, CBE, persistence, release, and
artifact behavior have one YAML implementation.

`integrate-photometric.yml` is the terminal illumination/background stage. It
requires a fully passing method policy and held-out validation, reconstructs
and proves the entire canonical normalized population, reproduces every prior
candidate route and output hash, and preserves all non-candidates. It publishes
the complete transformed corpus as a deterministic immutable Results release
while committing compact manifests and release provenance. Every page remains
eligible for downstream HTR whether its action is `corrected-and-continue` or
`preserve-and-continue`.

The user-facing design is intentionally not modeled on the current Golden Set
creation sequence. Internal commands remain composable for testing and
maintenance, while a researcher sees one preparation action followed by one
explicit apply-or-preserve choice. Golden Set creation should eventually adopt
the same guided orchestration rather than serve as the template here.

## Canonical workflow stages

```text
STAGE_PREPROCESS
STAGE_DETECT_CURRENT
STAGE_DETECT_CANDIDATES
STAGE_VALIDATE_GEOMETRY
STAGE_VALIDATE_OUTPUTS
STAGE_PUBLISH_PRODUCTION
STAGE_PUBLISH_TEST
```

Only the publication stage matching the active mode runs. Candidate detection
and geometry validation currently run outside production until calibration and
acceptance policy are mature.

## Banners and timing

Every stage begins through:

```bash
python hth-pipeline/hth/stage_timing.py start --stage "STAGE_NAME"
```

and completes through:

```bash
python hth-pipeline/hth/stage_timing.py finish \
  --stage "STAGE_NAME" \
  --start-epoch "..." \
  --started-at "..." \
  --status "success" \
  --timings-file "$RUNNER_TEMP/hth-stage-timings.jsonl"
```

The start command writes step outputs and a visible `HTH :: STAGE_NAME` banner.
The finish command records UTC completion, status, and elapsed duration.
Completion steps use `always()` so failed stages still leave timing evidence.

## Reports

- **Publication Manifest** records provenance and publication identity.
- **Pipeline Health** records counts, timestamps, stage performance, and output presence.

`hth/write_run_summary.py` reads generated JSON and the stage-timing JSONL file.
The workflow does not scrape human-readable logs.

## Future stage naming

Planned additions retain the same vocabulary:

```text
STAGE_OCR
STAGE_TRANSCRIBE
STAGE_TRANSLATE_<LANG>
STAGE_EXTRACT
STAGE_REASON
STAGE_PUBLISH
```


## Manual report regeneration

`.github/workflows/generate-report.yml` is manual-only and delegates to `_core-report.yml`. The report selector regenerates the detector-calibration manifest, execution-optimizer intelligence, or the full normalization audit summary. Optimizer reporting defaults to all detectors with completed persisted evidence and may be narrowed to one detector. Report generation performs no detector evaluation or preprocessing; it reads persisted intelligence from the results repository, writes the selected report to the Actions job summary, and republishes only the regenerated report files.

The report workflow defaults to `github-hosted` but exposes the same manual runner choices as other HTH builds. Detector-calibration report generation selects one best compatible persisted calibration record per detector for the configured Golden Set. Execution-optimizer report generation renders only the latest persisted optimizer execution for the selected detector; historical optimizer observations remain available in the intelligence indexes but are not mixed into the regenerated current-execution report.

## Execution optimizer

`.github/workflows/execution-optimizer.yml` is a direct manually dispatched job on the selected runner. It holds that runner for the entire optimization experiment and serially executes the same detector-regression workload across each requested execution shape. Shards equal pipelines, and the canonical execution plan divides the runner aggregate thread budget across active pipelines while honoring the configured per-pipeline thread bounds.

Pipeline enumeration defaults to exhaustive integer progression and also supports `powers-of-2` sampling plus adaptive peak/plateau search. The manual `resume` input defaults to `auto`; on a self-hosted runner it may reuse completed shapes from the latest compatible unpublished local optimizer checkpoint, while `no` forces a fresh execution and an explicit prior optimizer run ID requires that checkpoint. Resume skips only completed shapes and does not yet join live work or partial shapes. Each shape runs to completion before the next begins on the same physical runner, minimizing provisioning and environment variance. The optimizer captures only execution-shape observations into `indexes/parallelism-index.json` and derived optimizer intelligence; it does not publish calibration intelligence, regression manifests, or normal regression artifacts.

The optimizer intentionally remains a direct job rather than routing detector execution through `_core-hth.yml`, because the experiment must keep one runner allocation and one environment alive while iterating shapes. It mirrors the normal regression setup sequence and calls the same shared detector-regression shell driver as `.github/workflows/regress-detector.yml`, preserving the normal queue, shard, LOAD/START/UNLOAD, heartbeat, and detector log format for every repeated shape. Core-backed preprocessing and report generation are isolated in `_core-hth.yml` and `_core-report.yml`, respectively.

## Results-repository checkout policy

Read-oriented workflows explicitly consume the authoritative `main` results branch. Normalization reporting uses a shallow compact evidence checkout. Detector reporting first checks out only its indexes, then hydrates the one indexed smoke record selected per detector instead of materializing the complete calibration tree. Optimizer reporting temporarily retains bounded history for migration of summaries that predate durable per-run records. Report publication retries fetch/reset to the latest `origin/main`, regenerate from that current tree, and commit on top. Workflows that genuinely require historical Git traversal must opt into deeper history explicitly rather than inheriting it accidentally.
