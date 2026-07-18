# HTH Architecture — v0.6

## Design objective

HTH is designed to stand the test of time through explicit boundaries,
reproducible inputs, stable stage names, small interfaces, machine-readable
artifacts, and human-readable publication reports.

## Repository responsibilities

```text
Source repository
  Immutable or append-only collection masters and source metadata.

Pipeline repository
  Code, configuration, workflows, tests, schemas, and documentation.

Publication repository
  Curated production publication plus isolated test publication history.
```

Generated build directories are temporary implementation artifacts. A
publication is the stable consumer-facing output.

## Canonical stages

| Stage | Responsibility |
|---|---|
| `STAGE_PREPROCESS` | Extract and normalize source images; create manifests and derivatives. |
| `STAGE_DETECT_CURRENT` | Run the current physical-page analysis. |
| `STAGE_DETECT_CANDIDATES` | Run registered candidate detectors independently. |
| `STAGE_VALIDATE_GEOMETRY` | Compare physical geometry against approved references and policy. |
| `STAGE_VALIDATE_OUTPUTS` | Verify required files, counts, continuity, and consistency. |
| `STAGE_PUBLISH_PRODUCTION` | Prepare and commit the curated production publication. |
| `STAGE_PUBLISH_TEST` | Prepare and commit the isolated test publication. |

Stage names are HTH concepts rather than GitHub Actions concepts. Future local,
server, or alternate-CI runners should preserve the same names.

## Stage timing contract

Each stage emits:

```json
{
  "stage": "STAGE_DETECT_CANDIDATES",
  "status": "success",
  "started_at_utc": "2026-07-18T19:48:23Z",
  "completed_at_utc": "2026-07-18T19:48:29Z",
  "elapsed_seconds": 6.124
}
```

Records are written as JSON Lines during the run and rendered in Pipeline
Health. UTC timestamps provide a durable event record; elapsed time supports
capacity planning and regression detection.

## Reporting responsibilities

### Publication Manifest

Answers: **What was published, from which exact evidence and software?**

It includes mode, source repository and commit, source file, pipeline repository
and commit, workflow identity, publication location, and publication commit.

### Pipeline Health

Answers: **How did the pipeline perform?**

It includes status, collection identity, processing counts, pipeline start and
summary timestamps, total duration, stage-level timing, and publication-output
presence.

The reports remain separate because provenance and operational health are
different responsibilities.

## Detector architecture

Candidate detectors register through `hth/geometry/registry.py`. Each detector
returns a candidate or a structured failure without preventing other detectors
from running. New detectors should require one implementation module, one
registry entry, and focused tests.

## Durable principles

1. Workflow wrappers remain thin; reusable behavior belongs in `_core-hth.yml`.
2. Python owns parsing and presentation; YAML supplies facts and orchestration.
3. Generated JSON is the source of truth for counts and collection identity.
4. Publications are validated before they are committed.
5. Test and production remain structurally parallel unless their purpose truly differs.
6. Documentation changes ship with architectural or behavioral changes.
