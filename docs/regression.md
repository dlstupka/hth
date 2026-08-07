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

### Concurrent detector pipelines

Every multi-detector regression uses a dynamic detector queue. Four detector pipelines run by default for automatic smoke tests and for manually dispatched runs with **Algorithm = all**. Manual builds may select any whole-number detector pipeline count from 1 through the selected runner's aggregate detector-thread budget. A single-detector selection always uses one detector pipeline regardless of the requested multi-detector value.

Each pipeline takes the next unclaimed detector shard, runs it with the thread count from the build's canonical execution plan, and immediately takes another queued shard when it finishes. Pipelines remain occupied until the queue is empty; the workflow then waits only for the slowest remaining regression jobs. The runner budget is aggregate across active pipelines: GitHub-hosted runs use 8 detector threads total and E7K uses 64. Explicit thread requests act as a per-pipeline cap. `auto` divides the aggregate budget equally across active pipelines and may choose a non-power-of-two value, such as 21 threads for each of three active E7K pipelines.

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

After the normal result summary, regression statistics report record counts for average IoU, minimum IoU, and standard deviation; total metric improvements; parameter sets that improved at least one metric; overall winner changes; and whether the baseline was surpassed. These counts show whether a long search continued to produce useful gains or mostly traversed a plateau.

## Regression manifest navigation and engineering guidance

Multi-detector manifests retain the established nested report hierarchy. Single, custom, and manual detector manifests use an always-visible flat navigation menu with stable anchors and Back to Navigation links after major sections. GitHub job summaries do not allow the JavaScript required for a functional Expand All / Collapse All control, so HTH does not emit a misleading substitute.

The multi-detector `Detector Regression Reports` section begins with:

1. **Regression Completion Summary** — completed runs, evaluated parameter sets and pages, aggregate detector runtime, regression wall-clock span, effective detector concurrency, pipeline count, loading strategy, stagger, and source-document image count. The Notes column defines each measure and distinguishes aggregate detector work from user-observed elapsed time.
2. **Regression Execution and Detector Queueing** — the pipeline count, loading strategy, per-detector threads, stagger, intelligence indexes, and the persisted detector queue. Queue position, pipeline assignment, runtime estimate, and estimate source are recorded with each detector run.
3. **Regression Recommendations Summary** — separate Execution Configuration and Estimated Runtime tables. Runtime estimates cover all-detector exhaustive, non-dormant, and critical-only scopes. Estimates scale measured detector runtimes to the applicable effect-size domains, preserve the configured effect-group fallback rules, and simulate LPT placement across the recommended detector pipelines.

Every report ends with one **Engineering Continuous Improvement** section. Its Calibration Intelligence Persistence subsection lists, in order, the linked results commit, workflow run, pipeline repository, results repository, `calibration-index.json`, and `runtime-index.json`. The displayed commit, repository names, workflow label, and filenames are the hyperlinks; separate “open repository,” “open file,” and “open commit” helper links are not emitted. No duplicate persistence or workflow footer is appended after the report. The section explains how the two indexes preserve independent quality and execution evidence. Runtime and thread guidance must remain grounded in compatible historical measurements and is specific to the Golden Set, detector configuration, parameter grid, strategy, thread count, and runner characteristics represented by those observations.

## Detector calibration intelligence

Every completed detector regression writes `reports/calibration-intelligence.json`. The top-level multi-detector `Detector Regression Manifest` renders this evidence in a `Detector Calibration Report` immediately after the top-level Metric Definitions and before the individual detector sections. Its current-run ranking is titled **Ranked Detector Smoke Test Results**, includes the Golden Set ID, and omits the separate Short Name column. Single-detector manifests render the same **Best Known Detector Calibrations** table immediately before Calibration Intelligence so a detector run can be compared with all compatible detector calibrations for the same Golden Set.

The report includes:

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
- `adaptive_radial_edge` — `config/detectors/adaptive_radial_edge.json`
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

### Parallel completion ordering

Parallel parameter evaluation records `completion_index` in actual parameter-completion order. Shard coalescing reconstructs one global completion sequence from shard start times and per-result completion elapsed time, then derives discovery time, search-space percentage, winner history, and stabilization from that sequence. When runtime history is absent, queue reports display `no history` rather than `unknown`.


## Execution optimizer intelligence

The manual `HTH execution optimizer` workflow evaluates execution shapes serially in one direct job on one selected runner. It performs the same checkout, Python/ABI/toolchain/OpenCV setup and benchmark sequence as the normal detector regression once, then repeats the normal detector-regression execution driver for every selected shape. Shards equal detector pipelines.

Pipeline enumeration defaults to exhaustive integer progression from the configured minimum through maximum. The optional binary enumeration samples powers of two within the range plus both range endpoints. Threads per active pipeline are calculated as the smaller of the configured per-pipeline maximum and `floor(runner aggregate thread budget / pipelines)`; shapes that would fall below the configured minimum thread count are excluded. Thus runner policy controls total allocation while manual thread limits constrain each detector process.

After every successfully completed shape, the optimizer compares parameter sets/second with the best throughput seen so far. By default, three consecutive shapes that improve by less than 1% from that perceived maximum stop the remaining sweep. The stop rule is evaluated only after a complete shape and its state is retained in `optimizer-index.json`.

Every optimizer shard writes a shard-completion checkpoint into the raw optimizer parallelism data as soon as that shard finishes. Shape aggregates are written after merge. This preserves partial experimental evidence without turning optimizer runs into calibration runs. Optimizer execution does not publish calibration intelligence, regression manifests, or normal regression artifacts.

Runner health sampling is intentionally coarse. On the existing optimizer heartbeat only, Linux runners read `/proc/loadavg`, `/proc/stat`, and `/proc/meminfo` and emit a companion line such as `[runner e7k/rh8-a197] load=1042.7 cpu=93.4% iowait=1.2% ram=8.1G/2.0T swap=0`. Raw heartbeat samples are retained in optimizer intelligence and summarized for each shape; there is no separate high-frequency sampler.

Persistent artifacts are:

- `parallelism-index.json` — raw shape observations plus per-shard optimizer checkpoints;
- `optimizer-index.json` — detector- and runner-specific historical execution preferences plus individual optimizer-run metadata and runner samples;
- `execution-optimizer/<detector>/summary.md` — a table containing only shapes completed in the current optimizer execution; and
- `execution-optimizer/<detector>/heatmap.svg` — the current execution processing profile with detector pipelines on the X axis and parameter sets/second on the Y axis, with threads per pipeline annotated at each point.

Historical observations remain available to derive future detector-specific recommendations, but they are never mixed into the current-run table or processing profile.

