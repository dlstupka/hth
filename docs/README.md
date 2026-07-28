=========================================================================
HTH DEVELOPMENT STANDARD

This document is the authoritative source for:

- engineering standards
- repository conventions
- execution contract
- standing commands
- architecture principles

Before making ANY implementation changes:

1. Read this document.
2. Inspect the repository.
3. Inspect analogous implementations.
4. Follow the conventions herein.

Failure to do so invalidates the implementation.
=========================================================================



=========================================================================
# HTH Project Rulebook
=========================================================================

This document is the single authoritative HTH development standard and documentation entry point. It records established project conventions. It is the source of truth for recurring decisions that affect how the repository is changed, packaged, documented, and discussed.

Only agreed conventions belong here. Proposals, unresolved questions, and temporary plans belong elsewhere.

## Engineering Principle

Prefer maintainable, explicit, and reproducible designs over clever shortcuts.

## Execution Contract

Imperative implementation requests are executable commands. When the user says `create the overlay`, `implement`, `add`, `update`, or `fix`, inspect the repository and analogous implementations, perform the work, validate it, and deliver the completed overlay. Do not substitute a plan, proposal, design note, or description of intended work.

Interrupt execution only when a material requirement is genuinely ambiguous, required source material is missing, the request conflicts with this standard or the repository, or the proposed change is technically unsound. State the issue plainly rather than agreeing for the sake of agreement.

## Standing Commands

- `create the overlay` means implement, test, document, package `overlay.zip`, and include `COMMIT-MESSAGE.md`.
- `review the overlay` means inspect and report; do not modify unless requested.
- `critique this idea` means evaluate critically and identify weak assumptions or unnecessary complexity.

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

### Detector Debug Standard

Every detector participating in regression shall produce a comparable
debug artifact for the winning parameter set.

The purpose of debug artifacts is not only troubleshooting but also
cross-detector analysis, regression reproducibility, and research.

Each detector should produce:

- original page
- detector-specific intermediate images
- final overlay
- detector diagnostics (JSON)
- detector metadata

The exact intermediate artifacts are detector-specific, but the overall
structure and availability shall be consistent across all detectors.

## Configuration Boundaries

Keep reusable pipeline and detector configuration separate from source-document configuration. Detector configuration may define algorithm defaults and calibration search spaces, but must not hard-code collection identity, page ordinals, or document-specific exceptions. Source-document configuration currently present in this repository must remain separable so it can move to a dedicated source repository later.

## Regression Report Conventions

Use this section order for individual detector reports:

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

The top-level multi-detector `Detector Regression Manifest` places `Detector Calibration Report` immediately after the top-level Metric Definitions and before the individual detector report sections. The calibration report must characterize the evaluated search landscape without claiming behavior beyond the current Golden Set and configured parameter grid.

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

- Documentation filenames are lowercase except established special files such as `README.md` and `COMMIT-MESSAGE.md`.
- Update existing documentation before creating a new document.
- `docs/README.md` is the authoritative documentation entry point and development standard.
- New documents must describe a distinct, implemented concern; do not create speculative design packages or Word documents.
- Documentation must describe the current implementation, not an implementation that has not been made.

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



=========================================================================
# HTH Documentation
=========================================================================

This directory contains the design, operating, and project-reference documentation for the Hidden Texas History Research Framework. Start with the repository-level [README](../README.md) for the project overview and current pipeline.

## Project and architecture

- [Architecture](architecture.md) — system boundaries and durable contracts.
- [Project status](project-status.md) — current implementation and research status.
- [Workflow architecture](workflow-architecture.md) — GitHub Actions structure and workflow responsibilities.
- [Development](development.md) — development, testing, and update-package practices.
- [Toolchain](toolchain.md) — supported execution environments and dependencies.

## Acquisition and preprocessing

- [Acquisition pipeline](acquisition-pipeline.md) — source-image capture and ingestion.
- [Preprocessor](preprocessor.md) — DOCX extraction and normalized publication inputs.
- [Physical-page analysis](analyze-pages.md) — page analysis and review-queue generation.

## Geometry and detectors

- [Multi-detector geometry](multidetector-geometry.md) — detector registry and geometry pipeline.
- [Physical geometry evaluation](physical-geometry-evaluation.md) — geometry validation and evaluation rules.
- [Detector components](detector-components.md) — connected-components detector.
- [Contour detector](detector-contour.md) — contour-based detector.
- [Contour quadrilateral detector](detector-contour-quad.md) — contour-derived quadrilateral detector with edge scoring.
- [Contour + Components detector](detector-contour-components.md) — contour hypotheses ranked with connected-component envelope evidence.
- [Contour + Projection detector](detector-contour-projection.md) — contour hypotheses ranked with perspective-normalized text projection evidence.
- [Consensus Quad detector](detector-consensus-quad.md) — agreement and confidence fusion across contour quadrilateral voters.
- [Edge-Contour Hybrid detector](detector-edge-contour.md) — contour hypotheses verified by independent line-segment evidence.
- [GrabCut detector](detector-grabcut.md) — GrabCut-based detector.
- [RANSAC detector](detector-ransac.md) — robust four-edge line-fitting detector.
- [Hough Lines detector](detector-hough.md) — probabilistic Hough-transform detector.
- [Line Segment Detector](detector-lsd.md) — LSD-based detector.

## Calibration and regression

- [Golden Set](golden-set.md) — approved references and evaluation inputs.
- [Calibration selection](calibration-selection.md) — selection and promotion of calibrated parameters.
- [Detector regression](regression.md) — regression execution, telemetry, reports, and debug artifacts.

## Publication and review tools

- [Publication](publication.md) — publication layout, provenance, and outputs.
- [Reference collection editor](reference-collection-editor.md) — single-detector review tool.
- [Multi-detector reference collection editor](reference-collection-editor-multidetector.md) — multi-detector review tool.
