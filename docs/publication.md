# HTH Publication and Reporting

## Publication layouts

Production publishes the curated collection at the publication-repository root.
Test runs publish beneath:

```text
test/latest/
test/history/run-<workflow-run-number>/
```

Test publication never overwrites production publication content.

## Publication Manifest

The Publication Manifest is the provenance report displayed in the GitHub
Actions job summary. It records the exact source, pipeline, workflow, and
publication identities needed to reproduce or audit a publication.

## Pipeline Health

Pipeline Health complements the manifest with:

- overall status;
- collection and commit identities;
- pipeline started and summary-generated UTC timestamps;
- total duration;
- DOCX, page, processed-page, and error counts;
- per-stage start, completion, status, and elapsed time;
- presence of expected publication outputs.

Stage timing is collected in `$RUNNER_TEMP/hth-stage-timings.jsonl` and consumed
by `hth/write_run_summary.py`.

## Timing interpretation

Timing is operational evidence, not a benchmark by itself. Compare like modes,
source sizes, runner classes, and detector configurations. A meaningful runtime
change is a prompt to investigate—not proof of a regression.

Expected runtimes should be documented only after several representative runs.
The stage table makes those expectations measurable as detectors and later OCR,
transcription, translation, extraction, and reasoning stages are added.
