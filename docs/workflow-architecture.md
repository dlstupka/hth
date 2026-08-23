# HTH Workflow Architecture

## Workflows

Reusable core:

```text
.github/workflows/_core-hth.yml
```

Thin entry workflows:

```text
.github/workflows/preprocess.yml
.github/workflows/preprocess-test.yml
.github/workflows/calibrate-geometry.yml
.github/workflows/generate-report.yml
```

The wrappers select mode, source, publication behavior, retention, validation policy, and runner. Manual core-backed workflows expose the common runner vocabulary and default to GitHub-hosted execution unless a different runner is selected. Processing and report-only behavior remain centralized in the reusable core where practical to prevent drift.

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

`.github/workflows/generate-report.yml` is manual-only and delegates to `_core-hth.yml` in `report` mode. The report selector currently regenerates either the persisted detector-calibration manifest or execution-optimizer intelligence. Optimizer reporting defaults to all detectors with completed persisted evidence and may be narrowed to one detector. Report generation performs no detector evaluation or preprocessing; it reads persisted intelligence from the results repository, writes the selected report to the Actions job summary, and republishes only the regenerated report files.

The report workflow defaults to `github-hosted` but exposes the same manual runner choices as other HTH builds. Detector-calibration report generation selects one best compatible persisted calibration record per detector for the configured Golden Set. Execution-optimizer report generation renders only the latest persisted optimizer execution for the selected detector; historical optimizer observations remain available in the intelligence indexes but are not mixed into the regenerated current-execution report.

## Execution optimizer

`.github/workflows/execution-optimizer.yml` is a direct manually dispatched job on the selected runner. It holds that runner for the entire optimization experiment and serially executes the same detector-regression workload across each requested execution shape. Shards equal pipelines, and the canonical execution plan divides the runner aggregate thread budget across active pipelines while honoring the configured per-pipeline thread bounds.

Pipeline enumeration defaults to exhaustive integer progression and also supports `powers-of-2` sampling plus adaptive peak/plateau search. The manual `resume` input defaults to `auto`; on a self-hosted runner it may reuse completed shapes from the latest compatible unpublished local optimizer checkpoint, while `no` forces a fresh execution and an explicit prior optimizer run ID requires that checkpoint. Resume skips only completed shapes and does not yet join live work or partial shapes. Each shape runs to completion before the next begins on the same physical runner, minimizing provisioning and environment variance. The optimizer captures only execution-shape observations into `parallelism-index.json` and derived optimizer intelligence; it does not publish calibration intelligence, regression manifests, or normal regression artifacts.

The optimizer intentionally remains a direct job rather than routing detector execution through `_core-hth.yml`, because the experiment must keep one runner allocation and one environment alive while iterating shapes. It mirrors the normal regression setup sequence and calls the same shared detector-regression shell driver as `.github/workflows/regress-detector.yml`, preserving the normal queue, shard, LOAD/START/UNLOAD, heartbeat, and detector log format for every repeated shape. Core-backed preprocessing, geometry, and report workflows centralize their common behavior in `_core-hth.yml`.

## Results-repository checkout policy

Read-oriented workflows consume the authoritative current results tree with a shallow `main` checkout (`fetch-depth: 1`). In particular, manual report regeneration does not need repository history: persisted calibration, optimizer, runtime, and learned-evidence indexes already live in the current tree. Report publication retries fetch/reset to the latest `origin/main`, regenerate from that current tree, and commit on top without converting the job into a full-history clone. Workflows that genuinely require historical Git traversal must opt into deeper history explicitly rather than inheriting it accidentally.
