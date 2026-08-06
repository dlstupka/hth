# HTH Workflow Architecture — v0.6

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
```

The wrappers select mode, source, publication behavior, retention, and
validation policy. Processing behavior remains centralized in the reusable
core to prevent drift.

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

## Execution optimizer

`.github/workflows/execution-optimizer.yml` is a manually dispatched experiment driver for one detector. It launches full exhaustive child regressions sequentially, sets shards equal to the requested pipeline count, and leaves threads on `auto` so the canonical execution plan divides the selected runner's aggregate thread budget across active pipelines.

Auto enumeration uses representative pipeline counts through the runner budget. Manual enumeration accepts an inclusive pipeline minimum and maximum. Every successful child regression contributes its measured execution shape to `parallelism-index.json`; the optimizer summary links each child run.

The optimizer validates the selected execution runner with the same environment and toolchain checks as a normal detector regression, then releases it. A GitHub-hosted controller dispatches and monitors the child regressions so it cannot deadlock a single self-hosted runner by occupying that runner while waiting. The controller emits a UTC heartbeat during each child run.
