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

`.github/workflows/execution-optimizer.yml` is a thin manually dispatched wrapper around the reusable HTH core. The core holds the selected runner for the entire optimization experiment and serially executes the same detector regression workload across each requested execution shape. Shards equal pipelines and threads remain `auto`, so the canonical execution plan divides the runner's aggregate thread budget across the active pipelines.

Auto enumeration uses representative pipeline counts through the runner budget. Manual enumeration accepts an inclusive pipeline minimum and maximum. Each shape runs to completion before the next begins on the same physical runner, minimizing provisioning and environment variance. The optimizer captures only execution-shape observations into `parallelism-index.json`; it does not publish calibration intelligence, manifests, or normal regression artifacts.

The optimizer uses the same reusable core environment checks as normal HTH work: runner diagnostics, isolated Python, dependency ABI verification, toolchain reporting, OpenCV build reporting, OpenCV benchmark, and the core heartbeat. Its detector execution loop calls the same shared detector-regression shell driver as `.github/workflows/regress-detector.yml`, preserving the normal queue, shard, LOAD/START/UNLOAD, and detector log format for every repeated shape.
