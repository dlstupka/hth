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
- include `COMMIT-MESSAGE.md` with a short, meaningful subject line, an appropriately detailed body, and a trailing blank line;
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

Manual regression builds expose three debug levels:

- `none` (default) writes no debug images and avoids results-repository bloat;
- `basic` writes the established comparable debug package; and
- `verbose` adds detector-specific engineering evidence.

At `basic`, each participating detector should produce:

- original page
- detector-specific intermediate images already established for that detector
- final overlay
- detector diagnostics (JSON)
- detector metadata

At `verbose`, detectors may add feature-specific evidence without changing the common result contract. The exact intermediate artifacts are detector-specific, but the overall structure and availability shall remain consistent.

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
    Golden Set Page Issues
```

The top-level multi-detector `Detector Regression Manifest` places `Detector Calibration Report` immediately after the top-level Metric Definitions and before the individual detector report sections. The calibration report must characterize the evaluated search landscape without claiming behavior beyond the current Golden Set and configured parameter grid.
It must include a ranked calibration overview, source-specific corpus recommendation, detector roles, evidence tables, plain-English legends and summaries, ROI guidance, and Calibration Evidence. Median Avg IoU is not used as an engineering decision metric; the report uses winner Avg IoU, Min IoU, StdDev, baseline delta, basin width, failure behavior, and parameter influence instead.
An all-zero result field is not automatically a flat calibration landscape: when no valid page candidates or no positive-overlap measurements exist, reports must identify the missing calibration signal and withhold parameter-influence, domain-reduction, and ROI claims until valid measurements exist.

Additional conventions:

- The Result table uses `Parameter Short Name` and `Parameter Set ID` as separate columns.
- A Parameter Short Name is a human-assigned alias. When none exists, display the Parameter Set ID as the short name.
- Do not generate short names from rank, run date, or other unstable run metadata.
- Rank Top Parameter Sets by Avg IoU.
- Top Parameter Sets columns are: Rank, Parameter Short Name, Avg IoU, Min IoU, StdDev, Δ Avg IoU, Failures, Discovery Time, and Search Space %.
- The Golden Set Winner Summary omits StdDev.
- Sort page-oriented tables by Golden Set page number unless a report explicitly requires another order.
- Keep Metric Definitions with Results.
- Keep Status Definitions with Page Analysis.
- Multi-detector manifests retain the established nested navigation hierarchy.
- Multi-detector current-run rankings are titled `Ranked Detector Smoke Test Results`, include the Golden Set ID, and omit detector short names.
- `Best Known Detector Calibrations` first resolves the newest detector implementation revision represented by calibration evidence, then prefers compatible full calibration evidence within that revision and falls back to smoke/partial evidence until the revised implementation has a new full calibration. Older exhaustive results therefore cannot stand in for newer detector code. The table identifies detectors by Detector ID without a separate short-name display, omits redundant Coverage, and includes Search Type, deterministic Calibration Evidence, automatic Golden Set-scoped Approval Level, a compact linked `Build*` run number whose temporary logs and artifacts complement the persistent `calibration-intelligence.json`, and `Est. Serial Runtime**` showing the estimated single-detector serial runtime; the same table appears before Calibration Intelligence in single-detector manifests.
- Parameter-space terminology: **declared/nominal** sets are the configured discrete Cartesian grid; **invalid** sets violate detector constraints and should be rejected before evaluation; **redundant/no-op** sets are legal configurations that collapse to identical effective behavior and should be canonicalized rather than counted repeatedly; **effective** sets are unique valid behaviors worth evaluating; **evaluated** sets are effective sets actually completed. `exhaustive` means all valid sets in the declared discrete grid were evaluated, not every possible value in a continuous mathematical domain. Reports should keep the concise definition near the Best Known table and leave implementation detail here.
- Parameter influence uses the canonical HTH one-way η² classification defined in `docs/regression.md`: Zombie (<0.0005 η² and <0.0005 Avg-IoU range), Dormant (<0.005 η² after excluding Zombie), Low (<0.02), Moderate (<0.06), Important (<0.14), and Critical (≥0.14). These are engineering effect-size bands scoped to the characterized Golden Set/grid. η² remains canonical for the current document/baptismal collection; comparing ω² as a less-biased estimator is a documented future investigation, not a mixed-metric change to current reports.
- Single, custom, and manual detector manifests use an always-visible flat navigation menu with Back to Navigation links after major sections.
- Do not add nonfunctional Expand All / Collapse All controls to GitHub-rendered Markdown.
- Multi-detector regression summaries place Regression Execution and Detector Queueing immediately after Regression Completion Summary.
- Regression Recommendations use separate Execution Configuration and Estimated Runtime tables.

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
- `--threads` defaults to `1` and accepts any whole number from `1` through `1024`; the manual workflow currently retains its established dropdown values, while `auto` may supply any integer within that range.
- Runner thread limits are aggregate detector-thread budgets across concurrently active detector pipelines, not per-pipeline allowances. The GitHub-hosted budget is `8`; named self-hosted optimization runners use a maximum of twice their logical CPU count: E7K is `192` and E9K is `64`. One canonical execution plan supplies the queue, launcher, telemetry, and manifest. Explicit thread requests are capped by the equal per-pipeline budget; `auto` uses `floor(runner budget / active pipelines)`, including non-power-of-two values such as `21` when three E7K pipelines share 64 threads, so unused budget is minimized without exceeding the maximum. Named runner budgets are policy limits and may intentionally oversubscribe the logical CPUs reported inside a GitHub-hosted runner.
- Regression execution summaries distinguish unique `Detectors`, concurrent `Detector pipelines`, and queued `Shards`; shard jobs must never be reported as detector algorithms.
- Preferred detector execution is resolved before GitHub dispatches the regression job. A compatible optimizer preference carries its measured runner target, pipelines, threads per pipeline, and validated thread budget as one execution configuration; the regression job is routed to that runner profile before its `runs-on` choice is fixed. Auto and manual modes retain the runner explicitly requested by the workflow. The in-job resolver revalidates the pre-resolved preferred shape against the dispatched budget rather than silently substituting a GitHub-hosted default.
- A parameter-set limit is the total number of evaluated parameter sets including the baseline.
- Parallelism must not change parameter generation, result metrics, deterministic ranking, or report ordering.
- Parallel regressions record `completion_index` in actual parameter-completion order; merged shards reconstruct one global sequence for discovery and stabilization reporting.
- Sharded reports distinguish measured wall-clock elapsed time from estimated serial runtime and report effective acceleration as serial runtime divided by wall-clock elapsed time.
- Queue reports display `no history` when no compatible runtime observation exists.
- Keep one aggregate heartbeat; do not emit independent heartbeat streams from evaluation threads. Queue-job lifecycle telemetry records UTC-timestamped `LOAD`, `START`, and `UNLOAD` events for every shard, including `1/1` for unsharded work. Pipeline staggering occurs between `LOAD` and `START`, making queue residence and execution time distinguishable.
- Execution-optimizer aggregate and per-shard observations are durable experiment evidence and are not aged out of `parallelism-index.json`. Compatible completed optimizer runs may deliberately fill missing pipeline/thread shapes in later executions; historical optimizer intelligence coalesces those compatible measurements by detector, workload, and concrete runner profile while run-local tables remain auditable. Optimizer reports include the preferred measured executor shape and basic runner specifications, and the processing profile may combine compatible completed shapes across runs.
- Report search strategy and parameter-space scope before evaluation begins.
- Preserve runner CPU, thread, throughput, memory, and workload telemetry for every regression so exhaustive runs can guide future automatic thread selection and non-exhaustive search design.
- `parallelism-index.json` preserves raw compatible execution-shape observations and optimizer shard-completion checkpoints, plus compatibility-scoped best shapes and per-shape aggregate statistics including shard count, active pipelines, threads per pipeline, allocated threads, wall time, effective acceleration, parallel efficiency, throughput, workload identity, and concrete runner identity. Execution-optimizer shard checkpoints are written immediately when each shard completes.
- `optimizer-index.json` is a derived execution-planning abstraction built from `parallelism-index.json`. Historical intelligence remains detector-specific and runner-specific so different algorithms may retain different preferred execution shapes. Each optimizer run also stores its early-stop assessment and heartbeat-aligned runner `/proc` samples. The optimizer build summary keeps its detailed shape table current-run only while the preferred configuration and profile may use compatible completed history. Standalone `all` reporting is explicitly cumulative and coalesces completed compatible evidence across detectors; incomplete executions are never included.
- The manual `HTH execution optimizer` workflow is one direct GitHub Actions job on the selected runner. It performs the same checkout, Python/ABI/toolchain/OpenCV setup and benchmark sequence as `regress-detector.yml` once, then serially reruns the same `tools/run-detector-regressions.sh` execution path for each pipeline/thread shape. Shards equal pipelines. Pipeline enumeration supports exhaustive integer progression, `powers-of-2` sampling, and adaptive peak/plateau search. `powers-of-2` samples powers of two plus the requested range endpoints. Adaptive begins with the lowest and highest clean/common legal shapes when possible, then narrows toward the best known throughput. Once an interior peak is bracketed, adaptive explicitly measures the immediate neighboring pipeline counts and expands outward until both edges of the measured <=2% preferred-shape region are bounded by completed shapes outside that band (or by the requested range). Threads per pipeline equal `min(configured thread maximum, floor(runner aggregate budget / pipelines))`, and shapes below the configured thread minimum are excluded. By default, non-adaptive optimization stops after three consecutive successfully completed shapes improve parameter sets/second by no more than 2% from the perceived maximum; the assessment is made only after shape completion and is persisted with the run. Adaptive records the same 2% assessment but defers that generic plateau stop until its <=2% preferred-shape boundaries are resolved. The optimizer does not dispatch child workflows and does not route through `_core-hth.yml`; its Actions log is the normal detector-regression log repeated once per execution shape. Manual optimizer runs also expose `resume`, defaulting to `auto`: a compatible unpublished local optimizer checkpoint on the same self-hosted runner is imported into the new execution, completed pipeline/thread shapes are skipped, and only missing shapes are executed. `no` forces a fresh run; a prior optimizer run ID may be supplied to require that specific local checkpoint. Resume is shape-level only and does not join a still-running optimizer or partially completed shape. Reused completed shapes are replayed before run-local report generation, so the current execution's optimization-data table includes both newly completed and resumed checkpoint shapes. Optimizer runs do not publish calibration intelligence, regression manifests, or normal regression artifacts; they persist only optimizer parallelism/intelligence data and the current-run summary/profile. Execution-optimizer reports use the section order `Preferred Detector Run Configuration`, `Detector Run Profile Plot`, then `Detector Pipeline-Thread Shape Optimization Data`; all three sections are collapsible, the first two are expanded by default, and the detailed shape table is collapsed by default. Navigation follows the established manifest-style expandable hierarchy with Back to Navigation links. Standalone optimizer reporting defaults to `all`, which summarizes preferred configurations for every detector with completed persisted optimizer evidence and provides nested per-detector profiles and shape data; a single detector may still be selected explicitly. The preferred shape is selected by one canonical ranking shared by tables, plots, persisted executor preferences, and legacy report reconstruction: highest parameter-sets/second at the report's two-decimal precision, then lowest allocated threads, then lowest pipeline count, then lowest threads per pipeline (with wall time/sequence only as deterministic final tie-breakers). The preferred-configuration table also reports the search method and full optimizer wall time for the run that produced the preferred shape, plus the observed near-best shape range: completed compatible measured shapes whose throughput is within 2% of the best observed parameter-sets/second rate for that detector and runner profile. The qualifying shapes are enumerated explicitly as measured pipeline/thread pairs so the report never implies that unmeasured intermediate shapes qualified. The run-profile plot and pipeline/thread optimization-data section also identify the search method, and shape-data rows are ordered by increasing pipeline count.
- The manual-only `HTH generate report` workflow regenerates human-facing reports from persisted results without rerunning preprocessing or detector evaluation. Its report dropdown currently supports the detector-calibration manifest and the execution-optimizer report. It delegates to `_core-hth.yml` in `report` mode, defaults to GitHub-hosted execution, exposes the common manual runner choices, and republishes only the selected report files. Calibration manifest regeneration selects the best compatible persisted calibration record per detector for the configured Golden Set. Execution-optimizer regeneration defaults to `all` and may instead target one detector; it uses completed persisted optimizer evidence only. If a requested detector has no completed optimizer evidence, report generation falls back to the all-detector view and notes that the report shows all currently available optimization data. The all-detector report coalesces compatible completed evidence into preferred configurations and nested detector profiles, while single-detector report execution data remains tied to the latest completed run and its preferred configuration/profile may use compatible completed history.
- Runner health sampling for the execution optimizer is deliberately coarse and non-invasive. Only on the existing optimizer heartbeat cadence, read `/proc/loadavg`, `/proc/stat`, and `/proc/meminfo`; print one adjacent line in the form `[runner <label>/<name>] load=... cpu=... iowait=... ram=... swap=...`, retain the raw samples in optimizer intelligence, and summarize them by execution shape. Do not add a higher-frequency telemetry sampler unless later evidence requires it.



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
- [Preprocessor](preprocessor.md)
- [Immutable source releases](source-releases.md) — DOCX extraction and normalized publication inputs.
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
- [Radial Edge Search detector](detector-radial-edge.md) — center-outward gradient search for independent boundary generation.
- [Adaptive Radial Edge Search detector](detector-adaptive-radial-edge.md) — two-pass radial search that refines weak document sides at one-degree spacing.
- [Multi-Scale Radial Edge Search detector](detector-multi-scale-radial-edge.md) — scale-space radial fusion for boundaries that appear differently across spatial scales.
- [Page Background detector](detector-page-background.md) — robustly models the surrounding capture/background and extracts the enclosed non-background page region.
- [Projective Gradient Vote detector](detector-projective-gradient-vote.md) — long gradient-supported line families intersected into a perspective-aware quadrilateral.
- [Border Fusion Quad detector](detector-border-fusion-quad.md) — side-level fusion across radial, polar, and gradient boundary hypotheses.
- [Border Energy Validator detector](detector-border-energy.md) — contour geometry validated by gradient energy along all four borders.
- [Edge-Contour Hybrid detector](detector-edge-contour.md) — contour hypotheses verified by independent line-segment evidence.
- [GrabCut detector](detector-grabcut.md) — GrabCut-based detector.
- [RANSAC detector](detector-ransac.md) — robust four-edge line-fitting detector.
- [Hough Lines detector](detector-hough.md) — probabilistic Hough-transform detector.
- [Line Segment Detector](detector-lsd.md) — LSD-based detector.

## Calibration and regression

- [Golden Set](golden-set.md) — approved references and evaluation inputs.
- [Calibration selection](calibration-selection.md) — selection and promotion of calibrated parameters.
- [Detector regression](regression.md) — regression execution, telemetry, reports, and debug artifacts.
- [Detector parameter liveness audit](parameter-liveness-audit.md) — conservative zombie/deceased-parameter policy, current audit findings, and revalidation contract.
- [Orli Page-Mask detector](detector-orli-page-mask.md) — learned historical-document page-mask detector and calibration contract.
- [Orli learned-evidence persistence](orli-evidence-persistence.md) — deterministic inference reuse, persistent evidence index, identity, and invalidation.

## Publication and review tools

- [Publication](publication.md) — publication layout, provenance, and outputs.
- [Reference collection editor](reference-collection-editor.md) — single-detector review tool.
- [Multi-detector reference collection editor](reference-collection-editor-multidetector.md) — multi-detector review tool.

- Manual regression defaults to `preferred` execution shape: compatible persisted optimizer intelligence supplies the detector pipeline count and threads/pipeline as one exact execution contract. If no compatible preference exists, HTH falls back to the existing `auto` planner. `manual` accepts a compact shape such as `8p/48t`. Automatic smoke runs continue to use `auto`.

### Parallelism experimentation

- A self-hosted manual regression with `Algorithm = all` fans detectors into independent matrix jobs using the selected runner labels. GitHub Actions therefore consumes as many matching runners as are available in parallel; each detector resolves its preferred shape after landing on its actual runner. Exact runner optimizer evidence is preferred, with hardware-equivalent CPU/core-profile evidence allowed as a fallback before the generic auto planner.
- `parallelism-index.json` records detector execution shapes independently from runtime queue history, including shards, active pipelines, threads per pipeline, allocated threads, measured wall-clock time, estimated serial runtime, and effective acceleration.
- Parallelism experiments compare execution shapes such as `1×64`, `4×16`, and `8×8`; equal aggregate thread counts are not assumed to have equal performance.


### Manual runner targeting

Manual HTH workflows retain the existing runner-class selector and also expose a
`Specific self-hosted runner` selector. `any` preserves class-based scheduling;
`custom` uses the value entered in `Custom self-hosted runner label` as an exact
self-hosted runner label. Add a unique label matching the runner name when exact
runner targeting is desired, for example `rh8-al320`.

For a custom exact runner, the execution optimizer derives its default thread
budget from the selected runner itself (`2 × nproc`) rather than from a static
runner-name table. Explicitly requested pipeline/thread search bounds remain
unchanged in run metadata and display. Without `allow_thread_oversubscription`,
only legal shapes within the detected runner budget are executed; with the
explicit override enabled, oversubscribed shapes are allowed and reported as
such.

- Optimizer measurements use an optimizer-owned exact execution-shape contract: the optimizer selects the pipeline/thread shape and the regression driver executes it without applying a second thread clamp.

- Manual detector regression exposes `all-without-exhaustive` as the first/default detector target. It filters against the persisted calibration index using the current Golden Set and detector-configuration hashes, then dispatches one ordinary full, unlimited exhaustive regression per missing detector. Runner and execution-shape choices are preserved, allowing GitHub Actions to spread those child runs across all online runners matching the selected self-hosted labels.


### Preferred-shape fallback and prediction history

Normal full/exhaustive regression resolves execution shape in this order:

1. **Measured preferred** — use the canonical optimizer preference for the exact runner, or a hardware-equivalent runner profile.
2. **Predicted** — when no compatible measured preference exists, estimate detector pipelines from that detector's observed preferred pipeline counts versus runner vCPU, estimate useful allocated-thread fraction from the same evidence, and derive threads/pipeline from the detected runner thread budget.
3. **Auto** — if there is not enough compatible optimizer history to make a responsible detector-specific prediction, use the generic regression planner.

Predicted shapes are explicit execution contracts, just like measured preferred shapes. The run log identifies the source as `predicted-low`, `predicted-moderate`, or `predicted-high`.

Every prediction is saved in `optimizer-predictions.json` with the target runner, predicted shape, evidence vCPU anchors, confidence, and workload hashes. When later optimizer data arrives for the predicted detector/runner profile, optimizer publication verifies the saved prediction against the new canonical preferred shape and records pipeline/thread error. Verified pipeline error is then used as a bounded detector-specific correction for later predictions.

The execution-optimizer report includes shape-prediction coverage for each detector: observed vCPU anchors, readiness, prediction verification counts, and the desired/missing optimizer evidence needed to improve confidence.

### Additional boundary proposal detectors

- **Multi-Scale Radial Edge Search (`multi_scale_radial_edge`)** fuses independently normalized gradient evidence across several Gaussian scales before center-outward boundary sampling.
- **Adaptive Multi-Scale Radial Edge Search (`adaptive_multi_scale_radial_edge`)** holds the calibrated MSRE scale-space fixed and adds only ARE-style weak-side angular refinement, directly testing whether adaptive ray allocation improves MSRE without changing its multiscale evidence.
- **Page Background (`page_background`)** learns robust Lab color statistics from the outer capture border and extracts the dominant page region that differs from that background.
- **Segment-Supported Polar Voting (`segment_supported_polar_vote`)** combines polar boundary votes with LSD segment support; its Generation-3 exhaustive domain collapses support thresholds that were dormant in the prior winning basin and concentrates 180,000 sets on segment length/distance, radial bounds, ray density, and gradient percentile.
- **Projective Gradient Vote (`projective_gradient_vote`)** groups long gradient-supported segments into two near-orthogonal projective side families and intersects opposing lines into a quadrilateral.
- **Border Fusion Quad (`border_fusion_quad`)** recombines top/right/bottom/left hypotheses from Radial Edge Search, Polar Boundary Voting, and Gradient Boundary Voting, then validates the mixed-source quadrilateral against side gradients.
- **Polar Boundary Voting (`polar_boundary_vote`)** samples gradient evidence on center-outward polar rays, votes for strong outer transitions, and fits a document envelope from angularly distributed boundary support.
- **Star-Convex Boundary Optimization (`star_convex`)** uses the foreground mask to trace an outer supported radius around an interior anchor, smooths the radial boundary, and fits a quadrilateral around the resulting star-convex support.
- **Distance-Transform Rectangle Proposal (`distance_transform_rect`)** thresholds robust distance-transform interior support and expands its core envelope into a directly scored rectangle proposal. It is intentionally distinct from `distance_transform`, which selects core-supported connected components before fitting a hull/rectangle.

Their calibration JSON files define the complete discrete search grids used by exhaustive regression. Only behaviorally active controls are exposed; implementation-only constants and known no-op dimensions are not included as calibration parameters.

## Model-Backed Detector Lifecycle

- Detector execution has canonical config-driven `prepare` and `finalize` lifecycle hooks owned by `tools/run-detector-regressions.sh`, the shared detector executor used by normal regressions and the execution optimizer. Workflow YAML does not implement detector-specific lifecycle logic.
- Each unique detector is prepared exactly once before any shard/pipeline worker starts and finalized exactly once after all of its shards are complete. Ordinary detectors with no lifecycle declaration are no-ops; model-backed detectors may provision and validate external assets.
- `orli_page_mask` uses a fixed managed Orli historical-document base model. Its calibration parameters consume immutable model evidence rather than modifying or retraining the neural model; compatible learned evidence is persisted in the results repository and indexed by `orli-evidence-index.json`. See [Orli learned-evidence persistence](orli-evidence-persistence.md).
- `learned_page_mask` uses the released PageNet Ohio Death Records model from `ctensmeyer/pagenet` (BSD-3-Clause). HTH does not train on the Golden Set, avoiding evaluation leakage.
- On first execution the prepare hook checks `results-repo/models/pagenet-ohio/`; when absent it downloads the released prototxt and weights, derives an inference-only OpenCV-DNN prototxt, records SHA-256 provenance, exports the asset paths, and continues through the ordinary detector flow.
- Subsequent executions validate and reuse the persisted model. Calibration tunes only deterministic mask-to-boundary post-processing.
- The finalize hook revalidates provenance, and normal results persistence includes `models/` so the first successful run makes later executions self-contained.


- [Fusion Gen1 — MSRE + BFQ + SPBV + Page Background](detector-msre-bfq-spbv-pbg.md)

- **Fusion Gen2 (`amsre_bfq_spbv_pbg`)** replaces only Fusion Gen1's calibrated MSRE child with calibrated AMSRE while preserving BFQ, SPBV, Page Background, and the Gen1 fusion decision surface.

- **dhSegment Page Mask (`dhsegment_page_mask`)** provides an independent TensorFlow/ResNet-50 learned page-segmentation family using the upstream dhSegment v0.2 page-extraction model.
- Parameter provenance includes an additive `Parameter Set Equivalence Family ID` immediately before `Parameter Set ID` in report tables. It groups exact configurations only across durably enrolled equivalence dimensions; exact Parameter Set IDs and historical provenance are never rewritten.

- [Doc-UFCN Page-Mask Detector](detector-doc-ufcn-page-mask.md)
