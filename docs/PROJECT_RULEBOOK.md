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
- omit unchanged files, generated artifacts, caches, and the repository root directory;
- preserve the existing repository layout; and
- extract directly over the repository root.

Do not move or rename existing files unless the task explicitly includes that change.

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

## Regression Report Conventions

Use this section order:

```text
Results
    Result
    Metric Definitions
    Regression Statistics
    Top Parameter Sets

Page Analysis
    Golden Set Winner Summary
    Problem Pages
    Status Definitions
```

Additional conventions:

- Rank Top Parameter Sets by Avg IoU.
- Top Parameter Sets columns are: Rank, Parameter Set, Avg IoU, Min IoU, StdDev, Δ Avg IoU, and Failures.
- The Golden Set Winner Summary omits StdDev.
- Sort page-oriented tables by Golden Set page number unless a report explicitly requires another order.
- Keep Metric Definitions with Results.
- Keep Status Definitions with Page Analysis.

## Documentation and Commit Notes

Commit notes should be terse. Explain why a change was made when the reason is not obvious; do not narrate obvious edits.

## Updating This Rulebook

Update this file when the project adopts a recurring convention expressed as “always,” “never,” “from now on,” or “we decided.” Do not add a rule until it has actually been agreed.
