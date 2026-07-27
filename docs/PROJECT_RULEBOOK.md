# HTH Project Rulebook

This document records established HTH project conventions. It is the source of truth for recurring decisions that affect how the repository is changed, packaged, documented, and discussed.

Only agreed conventions belong here. Proposals, unresolved questions, and temporary plans belong elsewhere.

## Engineering Principle

Prefer maintainable, explicit, and reproducible designs over clever shortcuts.

## Overlay Policy

Unless explicitly requested otherwise, every patch deliverable must be named:

```text
overlay.zip
```

An overlay must:

- contain only files added or modified for the current task;
- include all changed source files, tests, and documentation;
- include a terse `COMMIT-MESSAGE.md` containing the recommended commit title and body;
- omit unchanged files, generated artifacts, caches, and the repository root directory;
- preserve the existing repository layout; and
- extract directly over the repository root.

The filename is always `COMMIT-MESSAGE.md`; do not create topic-specific or alternate commit-note filenames. The file is part of the overlay, not a separate optional deliverable.

Do not move or rename existing files unless the task explicitly includes that change. File moves, renames, and repository reorganizations must be deliberate changes described in the associated commit note.

## Established HTH Conventions

HTH has established conventions that supersede common project defaults.

Do not reintroduce legacy conventions from other projects.

Examples include:

- `unittest` (not `pytest`)
- `COMMIT-MESSAGE.md` (not alternate commit filenames)
- descriptive documentation filenames (not `README-*`)
- actual repository layout (do not invent paths)
- modify existing convention documents instead of creating parallel ones

## Repository Accuracy

When describing or illustrating HTH:

- use the actual repository paths and layout;
- say when a path or layout is unknown; or
- use an unmistakably fictitious example such as `example_project/`.

Never invent a plausible-looking HTH path for illustration.

## Stage Names

Do not number pipeline stages in their names. Stage names must remain stable when stages are inserted, removed, reordered, or expanded.

## Detector Architecture

- Keep the regression framework detector-agnostic.
- Keep detector-specific behavior behind the detector interface or adapter.
- A detector should expose its polygon result, metadata, timing, and diagnostics through the common framework.

## Configuration Boundaries

Keep reusable pipeline and detector configuration separate from source-document configuration. Detector configuration may define algorithm defaults and calibration search spaces, but must not hard-code collection identity, page ordinals, or document-specific exceptions. Source-document configuration currently present in this repository must remain separable so it can move to a dedicated source repository later.

## Regression Report Conventions

Use this section order:

```text
Run Information
    Build Provenance
    Golden Set
    Parameter Space
    Outputs

Results
    Result
    Metric Definitions
    Regression Statistics for Detector Calibration
    Top Parameter Sets

Page Analysis
    Golden Set Winner Summary
    Status Definitions
    Problem Pages
```

Additional conventions:

- The Result table uses `Parameter Short Name` and `Parameter Set ID` as separate columns.
- A Parameter Short Name is a human-assigned alias. When none exists, display the Parameter Set ID as the short name.
- Do not generate short names from rank, run date, or other unstable run metadata.
- Rank Top Parameter Sets by Avg IoU.
- Top Parameter Sets columns are: Rank, Parameter Short Name, Avg IoU, Min IoU, StdDev, Δ Avg IoU, and Failures.
- The Golden Set Winner Summary omits StdDev.
- Sort page-oriented tables by Golden Set page number unless a report explicitly requires another order.
- Keep Metric Definitions with Results.
- Keep Status Definitions with Page Analysis.

## Documentation and Commit Notes

Commit notes should be terse. Explain why a change was made when the reason is not obvious; do not narrate obvious edits.

## Updating This Rulebook

Update this file when the project adopts a recurring convention expressed as “always,” “never,” “from now on,” or “we decided.” Do not add a rule until it has actually been agreed.

## Regression Parallelism and Telemetry

- Detector calibration and regression operate against the Golden Set.
- `--threads` defaults to `1`; supported explicit values are powers of two from `1` through `1024`.
- Parallelism must not change parameter generation, result metrics, deterministic ranking, or report ordering.
- Keep one aggregate heartbeat; do not emit independent heartbeat streams from evaluation threads.
- Report search strategy and parameter-space scope before evaluation begins.
- Preserve runner CPU, thread, throughput, memory, and workload telemetry for every regression so exhaustive runs can guide future automatic thread selection and non-exhaustive search design.
