# Detector Regression

HTH regression is a black-box experiment framework. It knows parameter names and values, invokes a detector adapter, and measures results against the approved Golden Set. It does not interpret detector semantics.

## Run

```bash
python -m hth.regress_detector \
  --detector-config config/detectors/ransac.json  # or components.json / contour.json / contour_quad.json / contour_components.json / contour_projection.json / edge_contour.json / cross_edge_contour.json / gradient_vote.json / radial_edge.json / border_energy.json / grabcut.json / hough.json / lsd.json \
  --golden-set config/golden_set.json \
  --image-root path/to/preprocessed \
  --output build/regression \
  --strategy binary-refine
```

For a development smoke test add `--limit 2` with the exhaustive strategy.

## Canonical run directory

```text
build/regression/<detector>/run-YYYYMMDD-HHMMSS/
  manifest.json
  RUN-INFO.json
  parameters.json
  raw/results.csv
  reports/summary.json
  reports/rankings.csv
  reports/top20.csv
  logs/

build/regression/debug/<detector>/run-YYYYMMDD-HHMMSS/
  README.txt
  <parameter-set-id>/page-NNNN/
```

`raw/results.csv` is canonical: one row per parameter set × Golden Set page. Reports are derived and may be regenerated without rerunning the detector. The detector root also receives `<detector>-regression-results.csv` as a convenience copy of the latest full ranking.

`RUN-INFO.json` records the Python, OpenCV, platform and Git commit available to the runner. `manifest.json` begins in `running` state and ends as `complete` or `failed`.

## Strategies

- `exhaustive`: authoritative Cartesian product of configured values.
- `binary-refine`: black-box interval refinement followed by a local Cartesian pass.

The `baseline` profile is treated as a named production reference, not a privileged optimization rule.

## GitHub Actions

The manually dispatched **HTH detector regression** workflow checks out a results repository, runs the selected detector against the Golden Set, uploads the complete canonical run directory, and writes a **Regression Manifest** to the Actions job summary. The manifest records provenance, Golden Set pages, parameter-space size, winner and baseline metrics, and output validation.

When multiple detectors run together, the top-level **Detector Regression Manifest** identifies the source document and begins with a ranked detector-results table. Detectors are sorted by the regression quality ordering: average IoU, minimum IoU, failures, standard deviation, and evaluation time. The table places every detector's winner, IoU metrics, failure count, evaluated parameter-set count, estimated winner page throughput, source-document processing time derived from the configured image count, and complete detector-tuning run elapsed time side by side before the detailed per-detector manifests. The detail sections remain authoritative; the ranked table intentionally duplicates their key results to support rapid comparison.

For a manual run, the **Algorithm** input is a choice of `all`, `contour`, `contour_quad`, `contour_components`, `contour_grabcut`, `grabcut_contour`, `contour_projection`, `edge_contour`, `cross_edge_contour`, `gradient_vote`, `radial_edge`, `adaptive_radial_edge`, `border_energy`, `components`, `ransac`, `grabcut`, `hough`, or `lsd`. Automatic smoke runs continue to exercise all configured detector algorithms.

### Regression execution shape and concurrent runners

Manual regression exposes one **Execution shape** control instead of separate pipeline/thread knobs:

- `preferred` (default) resolves the detector's persisted execution-optimizer preference for the current workload and runner profile. Exact runner evidence is preferred; when that runner has no observation, hardware-equivalent evidence with the same CPU model/core topology may be reused. If no compatible preference exists, the workflow falls back to `auto`. Preferred intelligence is used only for full, unlimited, exhaustive regressions matching the optimizer workload.
- `auto` retains the normal wall-clock shard planner and bounded thread planner.
- `manual` accepts one compact shape such as `8p/48t`. The pipeline and thread counts are treated as an explicit execution contract.

A preferred/manual shape sets shards equal to pipelines and preserves the selected threads per active pipeline, so a persisted `23p/16t` preference is executed as 23 concurrent detector shards with 16 threads each rather than being re-planned downstream. The runner budget is validated before execution.

When a manually dispatched self-hosted regression selects **Algorithm = all**, the workflow fans the detector list into independent matrix jobs. Every job uses the same selected runner labels, so GitHub Actions naturally places detectors concurrently on every available matching runner; with one matching runner, the remaining jobs simply queue. Each detector resolves its own preferred execution shape after landing on its actual runner. GitHub-hosted and automatic smoke runs retain the original single-job multi-detector queue to avoid unexpectedly multiplying hosted-runner consumption.

Inside any single regression job, detector shards still use the existing dynamic queue. Each local pipeline takes the next unclaimed shard, runs it with the thread count from the canonical execution plan, and immediately takes another queued shard when it finishes.

Every queue job emits UTC-timestamped `LOAD`, `START`, and `UNLOAD` lifecycle lines with detector ID, shard identity, thread count, status, and wall time. Unsharded work is shown consistently as shard `1/1`. A configured pipeline stagger is applied after `LOAD` and before `START`, preserving the distinction between queue residence and detector execution.

The optional **Pipeline stagger minutes** input defaults to `0`, which starts every detector pipeline immediately. A positive whole-number value loads the first job for pipeline 2 and delays its `START` by one stagger interval, pipeline 3 by two intervals, and so forth. Staggering can reduce simultaneous startup pressure for expensive detectors without reducing the selected steady-state pipeline count.

Pipeline count, assigned pipeline number, and stagger interval are recorded in `parameters.json`, `RUN-INFO.json`, `reports/summary.json`, and `reports/calibration-intelligence.json` for each detector run. Concurrent console output is prefixed with the pipeline number and detector ID, while the combined manifest remains ordered deterministically by detector configuration.

Manual runs default to the `exhaustive` strategy. Limit handling is explicit:

- Smoke mode with a blank limit evaluates 10 parameter sets (`smoke default`).
- Any numeric limit is treated as a user-specified exhaustive cap.
- Full mode with a blank limit is unlimited; with `exhaustive`, it evaluates the complete configured Cartesian space.
- Full mode with `binary-refine` remains optimizer-directed rather than exhaustive.

The invocation table reports the effective limit as `10 (smoke default)`, `<n> (user specified)`, or `unlimited`, so a capped run cannot be mistaken for a complete regression.

Timeout policy is mode- and runner-specific:

- Automatic and manually dispatched smoke regressions: 30 minutes.
- GitHub-hosted full regressions: 360 minutes (6 hours).
- Self-hosted full regressions: 7200 minutes (5 days).

Progress telemetry reports the metrics for the most recently completed parameter set alongside the best value seen so far for each metric:

The heartbeat uses a two-row header so current and best-so-far values stay aligned:

```text
...  Avg   Best   Min   Best          Best  ...      Eval  Parameter
     IoU    IoU   IoU    IoU    SD      SD           Time  Set
```

The current columns describe the most recently completed parameter set. Higher is better for average and minimum IoU; lower is better for standard deviation. `Eval Time` is the wall-clock time for that parameter set. Sparse `New best` notes remain outside the fixed-width table. For long regressions, the two-row header and separator are repeated after every 50 telemetry rows, with a blank line before the repeated header so the columns remain readable in a fixed-height GitHub log window.

After the normal result summary, regression statistics report record counts for average IoU, minimum IoU, and standard deviation; total metric improvements; parameter sets that improved at least one metric; overall winner changes; and whether the baseline was surpassed. These counts show whether a long search continued to produce useful gains or mostly traversed a plateau. Individual detector manifests place a compact **Preferred Execution Shape** table immediately after these statistics and before **Top Parameter Sets**. The table records the actual shape used for that calibration, its resolution source (measured preferred, predicted, manual, auto fallback, or legacy/unknown), allocated threads, runner, and detected runner budget.

## Regression manifest navigation and engineering guidance

Multi-detector manifests retain the established nested report hierarchy. Single, custom, and manual detector manifests use an always-visible flat navigation menu with stable anchors and Back to Navigation links after major sections. Every top-level section heading in an individual detector manifest includes the detector ID as visible context, so a reader does not have to return to Build Provenance to identify which detector report is open. GitHub job summaries do not allow the JavaScript required for a functional Expand All / Collapse All control, so HTH does not emit a misleading substitute.

The multi-detector `Detector Regression Reports` section begins with:

1. **Regression Completion Summary** — completed runs, evaluated parameter sets and pages, aggregate detector runtime, regression wall-clock span, effective detector concurrency, pipeline count, loading strategy, stagger, and source-document image count. The Notes column defines each measure and distinguishes aggregate detector work from user-observed elapsed time.
2. **Regression Execution and Detector Queueing** — the pipeline count, loading strategy, per-detector threads, stagger, intelligence indexes, and the persisted detector queue. Queue position, pipeline assignment, runtime estimate, and estimate source are recorded with each detector run.
3. **Regression Recommendations Summary** — separate Execution Configuration and Estimated Runtime tables. Runtime estimates cover all-detector exhaustive, non-dormant, and critical-only scopes. Estimates scale measured detector runtimes to the applicable effect-size domains, preserve the configured effect-group fallback rules, apply the normal bounded full-regression shard plan, and simulate shard-level LPT placement across the recommended detector pipelines.


Report regeneration is concurrency-safe with calibration/optimizer publication. Before each publish attempt the report writer fetches and resets to the current results-repository `main`, regenerates the requested report from that fresh snapshot, and retries a bounded number of times if another writer wins the push race. This avoids publishing a stale report merely because calibration jobs updated the results repository while the report was being generated.

Every report ends with one **Engineering Continuous Improvement** section. Its Calibration Intelligence Persistence subsection lists, in order, the linked results commit, workflow run, pipeline repository, results repository, `calibration-index.json`, and `runtime-index.json`. The displayed commit, repository names, workflow label, and filenames are the hyperlinks; separate “open repository,” “open file,” and “open commit” helper links are not emitted. No duplicate persistence or workflow footer is appended after the report. The section explains how the two indexes preserve independent quality and execution evidence. Runtime and thread guidance must remain grounded in compatible historical measurements and is specific to the Golden Set, detector configuration, parameter grid, strategy, thread count, and runner characteristics represented by those observations.

## Detector calibration intelligence

Every completed detector regression writes `reports/calibration-intelligence.json`. The top-level multi-detector `Detector Regression Manifest` renders this evidence in a `Detector Calibration Report` immediately after the top-level Metric Definitions and before the individual detector sections. Its current-run ranking is titled **Ranked Detector Smoke Test Results**, includes the Golden Set ID, and omits the separate Short Name column. Single-detector manifests render the same **Best Known Detector Calibrations** table immediately before Calibration Intelligence so a detector run can be compared with all compatible detector calibrations for the same Golden Set.

The report includes:

The report uses **exhaustive** to mean complete evaluation of the detector's declared discrete calibration grid, after invalid combinations are excluded. This is not a claim of exhaustive coverage of an underlying continuous mathematical domain. Legal configurations that are behaviorally redundant or become no-ops under another setting should be canonicalized so repeated equivalent behavior does not inflate search-space or calibration-basin statistics.

- a **Best Known Detector Calibrations** table that prefers compatible full calibrations from `calibration-index.json`, falls back to smoke evidence when no full calibration exists, and records rank, detector ID, Golden Set ID, calibration date, a compact linked GitHub Actions build number, estimated single-detector serial runtime, parameter set ID, parameter-set count, search type, success rate, winner metrics, baseline delta, basin width, equivalent-best coverage, deterministic Calibration Evidence, and automatic Golden Set-scoped Approval Level; the redundant Coverage column is omitted, build run time is the wall-clock duration recorded for that build, and build links are temporary operational shortcuts while the persisted `calibration-intelligence.json` remains authoritative;
- generator, validator, and hybrid role definitions plus an evidence-source legend;
- search coverage and fully-successful parameter-set rate;
- best, minimum, standard-deviation, percentile, equivalent-winner, and near-best-basin calibration-landscape metrics;
- plain-English detector summaries and source-specific evidence-of-ROI recommendations;
- one-way parameter influence using eta-squared association with average IoU;
- parameter classifications (`Critical`, `Important`, `Moderate`, `Low`, and `Dormant`);
- near-best value coverage and the best observed values for each parameter;
- exploratory pairwise interaction estimates from a deterministic bounded sample;
- per-page mean, minimum, maximum, standard deviation, and success rate;
- dormant-parameter recommendations for reducing future searches;
- deterministic Calibration Evidence scored as 2 points for exhaustive completion, 1 point for at least 90% fully successful parameter sets, and 1 point for a near-best basin of at least 1% (Low 0–1, Medium 2–3, High 4);
- automatic Approval Level derived from Search Type and Calibration Evidence (Provisional, Candidate, Recommended, or Approved), scoped only to the identified Golden Set; and
- a corpus recommendation naming the highest-ranked detector and the evidence supporting a source-specific freeze-or-continue decision.

Each individual detector result also includes an evidence table describing the visual evidence it generates or validates. The Golden Set Winner Summary records winner-change count, first and last winner-change times, and search completion time.

All calibration-intelligence conclusions are explicitly limited to the evaluated Golden Set and configured parameter grid. A parameter classified as dormant may be omitted from future calibration for the same corpus, but it remains part of detector configuration and must be re-evaluated whenever the Golden Set changes. Interaction estimates are exploratory associations rather than causal claims.

The default image root is `test/latest/preprocessed` in the results repository. Override it at dispatch time when running against another published build.

## Regression runner toolchains

The workflow uses one regression engine on every runner, but Python provisioning differs by runner type:

- GitHub-hosted and self-hosted Linux runners use `actions/setup-python@v6` with Python 3.12.
- The self-hosted Windows runner uses a pre-provisioned Python 3.12.10 installation in the Actions tool cache. The expected interpreter is `$RUNNER_TOOL_CACHE/Python/3.12.10/x64/python.exe`, and `$RUNNER_TOOL_CACHE/Python/3.12.10/x64.complete` must exist. `pip` must also be available through `python -m pip`.

The Windows runner should be started from a normal, non-elevated PowerShell session. The workflow intentionally avoids reinstalling Python on that runner because the downloaded Windows tool-cache installer performs protected machine-wide registry cleanup. Do not delete the runner's `_work/_tool` directory unless the Python tool cache will be rebuilt afterward.

Bash-oriented steps run through `shell: bash` on both Linux and Windows. On Windows, Git Bash must resolve before the WSL launcher. A healthy command lookup begins with:

```text
C:\Program Files\Git\bin\bash.exe
```

Verify the ordering from PowerShell with:

```powershell
Get-Command bash -All
```

The **Show toolchain environment** step records the resolved Bash, Git, Python, and pip executables and versions in every regression log. On Windows it also prints `where.exe` results, making PATH-order regressions immediately visible. `HTH_RUNNER_LABELS` is set per selected runner so the detector banner reports GitHub-hosted versus self-hosted execution correctly.

## Supported detectors

- `components` — `config/detectors/components.json`
- `contour` — `config/detectors/contour.json`
- `contour_quad` — `config/detectors/contour_quad.json`
- `contour_components` — `config/detectors/contour_components.json`
- `contour_grabcut` — `config/detectors/contour_grabcut.json`
- `grabcut_contour` — `config/detectors/grabcut_contour.json`
- `contour_projection` — `config/detectors/contour_projection.json`
- `consensus_quad` — `config/detectors/consensus_quad.json`
- `edge_contour` — `config/detectors/edge_contour.json`
- `cross_edge_contour` — `config/detectors/cross_edge_contour.json`
- `gradient_vote` — `config/detectors/gradient_vote.json`
- `radial_edge` — `config/detectors/radial_edge.json`
- `adaptive_multi_scale_radial_edge` — `config/detectors/adaptive_multi_scale_radial_edge.json`
- `adaptive_radial_edge` — `config/detectors/adaptive_radial_edge.json`
- `multi_scale_radial_edge` — `config/detectors/multi_scale_radial_edge.json`
- `projective_gradient_vote` — `config/detectors/projective_gradient_vote.json`
- `border_fusion_quad` — `config/detectors/border_fusion_quad.json`
- `border_energy` — `config/detectors/border_energy.json`
- `ransac` — `config/detectors/ransac.json`
- `grabcut` — `config/detectors/grabcut.json`
- `hough` — `config/detectors/hough.json`
- `lsd` — `config/detectors/lsd.json`

RANSAC contributes ordered boundary-sampling, robust-line-fitting, and candidate-tuning tables before telemetry begins. Its winner debug package records boundary samples, fitted edge models, accepted inliers, and the candidate quadrilateral.

All use the identical black-box regression path and canonical output contract.

## Debug artifacts

Manual regression builds expose a `Debug level` choice:

- `none` (default) writes no page images;
- `basic` preserves the established detector debug package; and
- `verbose` includes the basic package plus detector-specific engineering evidence.

The workflow forwards the selected level through `--debug-level`. Automatic smoke builds use `none` so routine calibration and runtime intelligence do not bloat the results repository. The existing `regression.debug_artifacts` / `--debug-artifacts` policy still selects which parameter sets and pages are eligible (`failures`, `winner`, or `all`); `Debug level: none` overrides that selection and emits no debug tree.

At `basic`, regression runs create the existing top-level `debug/` tree beside detector run directories. Artifacts are grouped by detector and run ID and retain the original page, detector input mask, established detector-specific intermediates, final overlay, and complete JSON diagnostics.

At `verbose`, selected detectors add feature-specific evidence:

- Radial Edge Search adds the complete sampled-ray field and accepted-ray paths.
- Gradient Boundary Voting adds vertical and horizontal gradient-vote evidence and selected vote maxima.
- GrabCut adds class labels, definite-foreground seeds, and extracted foreground contours.
- Border Energy Validator adds border sampling bands and per-side energy annotations.

Verbose evidence supplements rather than replaces the basic package. Detector adapters continue to execute through the authoritative geometry registry, so serialized candidates retain detector name, origin, foundation, authors, version, and repository provenance.

## Golden Set provenance

Each regression records the SHA-256 digest of the exact Golden Set configuration file in `parameters.json`, `RUN-INFO.json`, `reports/summary.json`, the console environment banner, and the GitHub Actions job summary. This distinguishes runs that used the same configuration path after the file changed.

The console summary reports **Fully successful parameter sets**. A parameter set is fully successful only when every selected Golden Set page produced a valid candidate. A value of zero can therefore coexist with a ranked winner when every parameter set missed at least one page; individual successful and failed page-evaluation counts are reported separately.

Calibration intelligence distinguishes a genuinely flat landscape from an all-zero failure field. If no page evaluation produces a valid candidate, or valid candidates never achieve positive overlap with an approved Golden Set bounding box, the report marks the calibration signal as unavailable and withholds parameter-dormancy, effect-size, domain-space-reduction, and tuning-ROI conclusions. Zero IoU values from those cases are failure/zero-overlap evidence, not proof that parameter settings are equivalent.

## Human-readable winner analysis

The GitHub Actions regression summary includes a top-five parameter-set ranking, a Golden Set Winner Summary, a status legend, and a Golden Set Page Issues section. Parameter-set and provenance hashes are displayed as consistent 12-character prefixes while the canonical JSON artifacts retain the complete values.

Golden Set winner rows are ordered by winner IoU from best match to worst match, with Golden Set page number as the tie-breaker. The same regression threshold drives both the `Regressed` status and the Golden Set Page Issues regression count, so the summary cannot label pages as regressed while reporting zero regressed pages. Threshold values are printed directly in the legend and Golden Set Page Issues labels.

GitHub Actions job summaries render static Markdown and sanitized HTML; they do not permit the JavaScript required for click-sortable table headers. Complete machine-readable page results remain available in `reports/winner-pages.json` for external sorting and analysis.

## Parallel exhaustive regression

Detector regression accepts:

```text
--threads 1|2|4|8|16|32|64|96|128|256|512|1024|48
```

The default is `--threads 1`, preserving the existing serial behavior. Exhaustive
search evaluates parameter sets concurrently when the value is greater than one.
Result ordering and ranking remain deterministic: parallelism changes runtime, not
the parameter space, metrics, or report ordering.

Adaptive strategies retain their own evaluation order. The current
`binary-refine` strategy is sequential because each step depends on earlier
observations; `--threads` is still recorded as run configuration but does not
change that dependency.

A single aggregate heartbeat remains responsible for progress output. Individual
threads do not write independent progress lines.

## Up-front regression scope

Before evaluation begins, the runner reports:

- search strategy;
- possible parameter-set count;
- planned parameter-set count, or `adaptive / unknown`;
- Golden Set page count;
- planned page-evaluation count;
- parameter-set limit; and
- configured thread count.

For exhaustive search, the planned count is exact. Smoke runs are exhaustive runs
with a parameter-set limit. For adaptive strategies, the actual count is recorded
when the run finishes.

### Execution optimizer adaptive shape refinement

The execution optimizer's `adaptive` pipeline enumeration is designed to find a preferred runtime shape without filling the complete pipeline-count curve. It first brackets a promising peak using sparse measurements. Once the current best shape has completed measurements on both sides, adaptive measures the immediately adjacent pipeline counts (for example, a peak at 8 pipelines requires 7- and 9-pipeline measurements) and continues outward while shapes remain within 2% of the measured peak. Refinement stops only after each side of that <=2% preferred-shape region is bounded by a completed shape outside the band or by the requested pipeline limit.

The generic three-shape / 2% throughput early-stop assessment remains recorded for adaptive runs, but it does not terminate an adaptive run before this 2% boundary refinement is complete.

## Runner performance telemetry

Every regression writes periodic machine-readable samples to:

```text
logs/runner-performance.jsonl
```

Samples include elapsed time, completed parameter sets and page evaluations,
parameter-set and page-evaluation throughput, configured and active threads,
process CPU utilization, process CPU time, and peak resident memory. The final
summary and `RUN-INFO.json` also record runner CPU topology, configured threads,
parameter-space scope, sample count, and peak memory.

Exhaustive results preserve every evaluated parameter set and its page-level
metrics. These runs therefore provide both calibration results and the evidence
needed to compare future non-exhaustive search strategies against the known
exhaustive outcome.

## Persistent calibration intelligence

Every successful detector regression on a push or manual workflow run publishes its
machine-readable calibration intelligence to the results repository. Pull-request runs
remain artifact-only because repository secrets are not available to untrusted changes.

The top-level lookup file is:

```text
calibration-index.json
```

Each permanent record is stored beneath:

```text
source-documents/<source-document-id>/golden-sets/<golden-set-id>/<golden-set-sha>/
  calibrations/<detector-id>/<calibration-id>/
```

A record includes `calibration-intelligence.json`, `manifest.json`, `parameters.json`,
`RUN-INFO.json`, `summary.json`, and `winner-pages.json`. Smoke tests publish provisional
records, incomplete or reduced full regressions publish partial records, and complete
exhaustive full regressions publish authoritative records. The index preserves all records
and selects the newest strongest compatible record without allowing a smoke test to replace
an authoritative calibration.

Concurrent smoke, manual, and full regressions may finish together and attempt to publish to
the same results repository. Publication therefore refreshes to the latest `origin/main`,
reapplies the completed run records, and deterministically rebuilds both
`calibration-index.json` and `runtime-index.json` before every push attempt. A rejected push
is treated as a publication collision rather than a merge problem: the publisher waits 5,
10, 15, then 20 seconds across subsequent attempts, refreshes again, and rebuilds the indexes
from the new authoritative repository state. Generated intelligence indexes are never rebased
or manually merged.

Effect-size regression strategies resolve prior intelligence through the index using the
Golden Set SHA-256, detector ID, and detector-configuration SHA-256. If no compatible record
exists, the existing strategy fallback resolves to exhaustive.

## Automatic thread selection and bounded regression shards

Full exhaustive regressions use smoke-test runtime history to estimate the detector's serial-equivalent workload. When `threads` is `auto`, the planner selects the smallest useful thread count: one thread below five minutes, up to four threads from five to fifteen minutes, up to eight threads from fifteen to thirty minutes, and the runner-profile maximum above thirty minutes. Named optimization runner budgets are 192 threads for `e7k` and 64 threads for `e9k`; other runner profiles use their configured aggregate budgets.

After conservative thread-speedup adjustment and a 20% planning margin, work estimated to exceed the configured 30-minute shard target is divided into deterministic interleaved parameter-set shards. Manual full exhaustive runs may instead provide an explicit shard count; that count takes precedence over wall-clock planning and is capped at the number of possible parameter sets so every shard receives at least one parameter set. Interleaving distributes clustered expensive configurations more evenly than contiguous parameter ranges. Smoke tests, limited searches, and non-exhaustive strategies remain unsharded.

Shard claims are leases rather than permanent locks. Active workers renew their lease every minute. Another detector pipeline may reclaim a shard after the configured lease expiration, minimizing work stranded by a terminated worker. Completed shards are merged into one canonical detector regression before calibration intelligence, summaries, and winner debug artifacts are published. Shard metadata, source run IDs, selected threads, and the interleaved assignment method are retained in the merged provenance.

Parallel shards share one run-local baseline cache per detector. The first shard to reach the baseline evaluates and publishes it; sibling shards wait for and reuse that result instead of repeating the same Golden Set inference. The cache is discarded with the regression output, so it never becomes cross-run calibration state. Progress accounting includes the baseline as a completed parameter set, keeping the progress numerator/denominator consistent with **Planned Parameter Sets** and **Planned Page Evaluations**. Shard optimizer telemetry separately records locally evaluated parameter sets so reused baseline results do not inflate per-shard throughput.

`orli_page_mask` additionally persists its deterministic parameter-invariant neural evidence in the results repository. Compatible model/page/inference identities hydrate the run-local shared-evidence manifest without rerunning Orli, so execution-optimizer shapes and later builds reuse the same expensive inference work. See [Orli learned-evidence persistence](orli-evidence-persistence.md) for the identity and `orli-evidence-index.json` contract.

### Parallel completion ordering

Parallel parameter evaluation records `completion_index` in actual parameter-completion order. Shard coalescing reconstructs one global completion sequence from shard start times and per-result completion elapsed time, then derives discovery time, search-space percentage, winner history, and stabilization from that sequence. When runtime history is absent, queue reports display `no history` rather than `unknown`.


## Execution optimizer intelligence

The manual `HTH execution optimizer` workflow evaluates execution shapes serially in one direct job on one selected runner. It performs the same checkout, Python/ABI/toolchain/OpenCV setup and benchmark sequence as the normal detector regression once, then repeats the normal detector-regression execution driver for every selected shape. Shards equal detector pipelines.

Pipeline enumeration supports exhaustive integer progression, `powers-of-2` sampling, and adaptive peak/plateau search. `powers-of-2` samples powers of two within the range plus both range endpoints. Adaptive starts from the lowest and highest clean/common legal shapes when possible and narrows toward the best measured throughput instead of filling the full shape curve. Threads per active pipeline are calculated as the smaller of the configured per-pipeline maximum and `floor(runner aggregate thread budget / pipelines)`; shapes that would fall below the configured minimum thread count are excluded. Thus runner policy controls total allocation while manual thread limits constrain each detector process.

Optimizer resume is shape-level. The manual `resume` input defaults to `auto` and can reuse completed shapes from a compatible unpublished local checkpoint left by a failed optimizer job on the same self-hosted runner; `no` forces a fresh run and an explicit prior optimizer run ID requires that checkpoint. Completed shapes are imported into the new optimizer execution and skipped, while incomplete shapes are rerun. Resume does not yet join a live optimizer or partially completed shape. Run-local optimizer reporting includes both newly completed shapes and compatible completed shapes reused from the checkpoint.

After every successfully completed shape, the optimizer compares parameter sets/second with the best throughput seen so far. By default, three consecutive shapes that improve by no more than 2% from that perceived maximum stop the remaining sweep. The stop rule is evaluated only after a complete shape and its state is retained in `optimizer-index.json`.

Every optimizer shard writes a shard-completion checkpoint into the raw optimizer parallelism data as soon as that shard finishes. Shape aggregates are written after merge. This preserves partial experimental evidence without turning optimizer runs into calibration runs. Optimizer execution does not publish calibration intelligence, regression manifests, or normal regression artifacts.

Runner health sampling is intentionally coarse. On the existing optimizer heartbeat only, Linux runners read `/proc/loadavg`, `/proc/stat`, and `/proc/meminfo` and emit a companion line such as `[runner e7k/rh8-a197] load=1042.7 cpu=93.4% iowait=1.2% ram=8.1G/2.0T swap=0`. Raw heartbeat samples are retained in optimizer intelligence and summarized for each shape; there is no separate high-frequency sampler.

Persistent artifacts are:

- `parallelism-index.json` — raw shape observations plus per-shard optimizer checkpoints;
- `optimizer-index.json` — detector- and runner-specific historical execution preferences plus individual optimizer-run metadata and runner samples;
- `execution-optimizer/<detector>/summary.md` — a table containing only shapes completed in the current optimizer execution; and
- `execution-optimizer/<detector>/heatmap.svg` — the current execution processing profile with detector pipelines on the X axis and parameter sets/second on the Y axis, with threads per pipeline annotated at each point.

Historical observations remain available to derive future detector-specific recommendations, but they are never mixed into the current-run table or processing profile.


### Optimizer-owned execution shapes

The execution optimizer is the authoritative planner for a measured pipeline/thread shape. When it invokes the regression driver it exports `HTH_EXACT_EXECUTION_SHAPE=1`, the runner thread budget, and the requested per-pipeline thread count. In that mode the regression driver must execute the supplied shape exactly; it must not reapply standalone regression thread heuristics or silently clamp the requested thread count. Normal regression runs continue to use the regular bounded thread/shard planner.

This ownership boundary keeps policy in one layer: the optimizer chooses a legal experimental shape, while the regression driver executes and measures it. The driver still validates that an exact shape does not exceed the supplied runner budget unless the optimizer explicitly enabled oversubscription.


### Execution optimizer target modes

The manual execution optimizer detector selector also provides `all` and `all-without-preference`. `all` dispatches one normal optimizer workflow run for every configured detector while preserving the selected runner, bounds, resume policy, and search method. `all-without-preference` dispatches only detectors that do not yet appear in the persisted preferred executor configuration index. Individual dispatched runs remain ordinary single-detector optimizer runs, so persistence, resume, reporting, and runner scheduling retain the same behavior. The default pipeline-shape search is `adaptive`.

### Fill missing exhaustive calibration evidence

The manual detector-regression selector places `all-without-exhaustive` first. This mode reads the current results-repository `calibration-index.json` and dispatches one ordinary detector regression for each configured detector that lacks compatible completed exhaustive evidence for the current Golden Set hash and detector-configuration hash. The dispatched child runs are always `full`, `exhaustive`, and unlimited; execution-shape and runner selections are preserved. Each child is an ordinary single-detector workflow run, so GitHub Actions can schedule them concurrently across every online self-hosted runner matching the selected labels, while excess detector jobs simply queue.


### Preferred-shape fallback and prediction history

Normal full/exhaustive regression resolves execution shape in this order:

1. **Measured preferred** — use the canonical optimizer preference for the exact runner, or a hardware-equivalent runner profile.
2. **Predicted** — when no compatible measured preference exists, estimate detector pipelines from that detector's observed preferred pipeline counts versus runner vCPU, estimate useful allocated-thread fraction from the same evidence, and derive threads/pipeline from the detected runner thread budget.
3. **Auto** — if there is not enough compatible optimizer history to make a responsible detector-specific prediction, use the generic regression planner.

Predicted shapes are explicit execution contracts, just like measured preferred shapes. The run log identifies the source as `predicted-low`, `predicted-moderate`, or `predicted-high`.

Every prediction is saved in `optimizer-predictions.json` with the target runner, predicted shape, evidence vCPU anchors, confidence, and workload hashes. When later optimizer data arrives for the predicted detector/runner profile, optimizer publication verifies the saved prediction against the new canonical preferred shape and records pipeline/thread error. Verified pipeline error is then used as a bounded detector-specific correction for later predictions.

The execution-optimizer report includes shape-prediction coverage for each detector: observed vCPU anchors, readiness, prediction verification counts, and the desired/missing optimizer evidence needed to improve confidence.

### Preferred-shape replay and optimizer startup overhead

A measured or predicted preferred execution shape is an atomic `pipelines × threads/pipeline` contract. For a single detector, replaying an exact shape therefore creates one regression shard per requested active pipeline before fan-out; the normal single-detector shortcut and runtime-based standalone shard planner are not allowed to collapse a preferred `Np/Mt` shape to `1p/Mt`. The executor fails loudly if shard expansion cannot sustain the requested exact pipeline count.

Each optimizer shape also records **executor startup overhead** from entry into `run-detector-regressions.sh` through detector lifecycle preparation, sharding/planning, shared learned-evidence resolution or preparation, and initial queue setup immediately before worker fan-out. This value is retained in optimizer intelligence and shown in the shape table. It is intentionally **not subtracted** from shape wall time or shape-level parameter sets/second: the report keeps the incurred end-to-end cost visible. Per-shard parameter-set throughput is timed from each worker's START through completion, after fan-out, so pre-fan-out startup overhead does not contaminate shard-level throughput analysis.

### Search-strategy legend and zombie parameters

HTH distinguishes ordinary exhaustive calibration from deliberate revalidation of parameters that prior complete calibration has shown to be effectively deceased for the current Golden Set and declared grid:

- `exhaustive` — evaluates the complete current live Cartesian calibration space. Parameters explicitly isolated as `zombie` are held at their detector **baseline value** and therefore do not consume ordinary exhaustive search budget.
- `exhaustive-with-zombies` — evaluates the same live Cartesian space while restoring every retained value of any configured `zombie_parameters`. This mode exists for deliberate regression/revalidation when an engineer wants to challenge the prior liveness conclusion.
- `non-dormant`, `low+`, `moderate+`, `important+`, and `critical` — use persisted calibration intelligence to restrict the current live space by measured effect-size classification, with the established fallback toward broader domains when a requested domain is empty. **Every parameter excluded by a contracted strategy is pinned to its detector baseline value.** Contracted parameter dictionaries therefore remain canonical and their Parameter Set IDs do not drift when a different historic winner is published. Under the current classification thresholds `non-dormant` and `low+` select the same Low-or-higher domain; both names are intentionally retained for compatibility and possible future policy differentiation.
- `binary-refine` — retains the sequential local-refinement strategy and is not a sharded exhaustive search.

Parameter influence has one canonical HTH classification, based on one-way η² over Avg IoU for the characterized Golden Set/grid:

| Class | Criterion | Engineering interpretation |
|---|---|---|
| Zombie | η² < 0.0005 **and** Avg-IoU range < 0.0005 | Practically indistinguishable from zero |
| Dormant | η² < 0.005, excluding Zombie | Measurable or potentially measurable, but operationally negligible |
| Low | 0.005 ≤ η² < 0.02 | Small effect |
| Moderate | 0.02 ≤ η² < 0.06 | Meaningful secondary influence |
| Important | 0.06 ≤ η² < 0.14 | Strong influence |
| Critical | η² ≥ 0.14 | Dominant influence |

Zombie and Dormant are therefore adjacent **measured effect-size classes**, not separate policy concepts and not synonyms. A parameter classified Zombie may be isolated from the ordinary exhaustive space only when its prior domain, pinned value, audit scope, and last compatible measured evidence are retained so `exhaustive-with-zombies` can revalidate the conclusion. Dormant parameters remain distinct: they are non-zombie dimensions below the normal Low threshold. All classifications are scoped to the document/Golden Set/declared grid and must be reconsidered when that characterization scope materially changes.

HTH currently standardizes on η² so the present document/baptismal collection remains internally comparable. **TODO:** evaluate ω² as a less-biased effect-size estimator and compare η²/ω² classifications on retained calibration evidence before considering any future metric migration. Do not mix estimators within the current collection's canonical reports.

The configuration-level liveness audit is intentionally conservative. It never promotes a parameter to zombie merely because it is singleton, baseline-only, or historically described as dormant. Run `python tools/audit-parameter-liveness.py` to validate all detector liveness metadata; use `--json` for machine-readable output.

The calibration report's **Parameter Set Domain Space Reduction** table always starts with `Exhaustive-with-zombies`, followed by ordinary `Exhaustive`. The first row is the widest retained revalidation universe and is the denominator for the table's percentage and reduction-factor columns. Detectors with no configured zombie parameters therefore show identical counts for those two rows. A detector with audited zombies, such as Orli, shows the actual cost avoided by normal exhaustive calibration before effect-size reductions are considered.

Search-space and parameter-set reporting is canonicalized from detector configuration plus the resolved strategy. The report records the zombie-inclusive universe, live exhaustive universe, resolved strategy universe, and evaluated search-space member count as one authoritative accounting contract. Mandatory baseline and historic-best evaluations are reference observations: if their exact parameter set lies outside the resolved search universe, they remain visible for comparison but cannot add parameter values, inflate effect-size domains, alter exhaustive counts, or participate in parameter-influence statistics. This prevents a historic reference from making a reduced search appear larger than the exhaustive space that actually ran.

