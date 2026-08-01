# Detector Regression

HTH regression is a black-box experiment framework. It knows parameter names and values, invokes a detector adapter, and measures results against the approved Golden Set. It does not interpret detector semantics.

## Run

```bash
python -m hth.regress_detector \
  --detector-config config/detectors/ransac.json  # or components.json / contour.json / contour_quad.json / contour_components.json / contour_projection.json / edge_contour.json / cross_edge_contour.json / gradient_vote.json / grabcut.json / hough.json / lsd.json \
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

For a manual run, the **Algorithm** input is a choice of `all`, `contour`, `contour_quad`, `contour_components`, `contour_grabcut`, `grabcut_contour`, `contour_projection`, `edge_contour`, `cross_edge_contour`, `gradient_vote`, `components`, `ransac`, `grabcut`, `hough`, or `lsd`. Automatic smoke runs continue to exercise all configured detector algorithms.

### Concurrent detector pipelines

Every multi-detector regression uses a dynamic detector queue. Four detector pipelines run by default for automatic smoke tests and for manually dispatched runs with **Algorithm = all**. Manual builds may select 1, 2, 4, or 8 detector pipelines. A single-detector selection always uses one detector pipeline regardless of the requested multi-detector value.

Each pipeline takes the next unclaimed detector configuration, runs that detector's complete regression using the separately configured per-detector `threads` value, and immediately takes another queued detector when it finishes. Pipelines remain occupied until the detector queue is empty; the workflow then waits only for the slowest remaining detector regressions. Detector pipelines and regression threads are independent forms of parallelism: four pipelines with 16 threads may run as many as four detector processes, each configured for 16 regression threads. Select values appropriate for the runner's CPU and memory capacity.

The optional **Pipeline stagger minutes** input defaults to `0`, which starts every detector pipeline immediately. A positive whole-number value delays pipeline 2 by one stagger interval, pipeline 3 by two intervals, and so forth. Staggering can reduce simultaneous startup pressure for expensive detectors without reducing the selected steady-state pipeline count.

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

## Detector calibration intelligence

Every completed detector regression writes `reports/calibration-intelligence.json`. The top-level multi-detector `Detector Regression Manifest` renders this evidence in a `Detector Calibration Report` immediately after the top-level Metric Definitions and before the individual detector sections.

The report includes:

- a ranked calibration overview with detector role, winner Avg IoU, Min IoU, StdDev, and change from each detector's named baseline;
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
- Calibration Evidence based on search completeness, success rate, basin width, and sample size; and
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
- `ransac` — `config/detectors/ransac.json`
- `grabcut` — `config/detectors/grabcut.json`
- `hough` — `config/detectors/hough.json`
- `lsd` — `config/detectors/lsd.json`

RANSAC contributes ordered boundary-sampling, robust-line-fitting, and candidate-tuning tables before telemetry begins. Its winner debug package records boundary samples, fitted edge models, accepted inliers, and the candidate quadrilateral.

All use the identical black-box regression path and canonical output contract.

## Debug artifacts

Regression runs create a first-class top-level `debug/` tree beside the detector run directories. Debug artifacts are grouped by detector and run ID, for example `build/regression/debug/contour/run-20260724-081500/`, so forensic evidence remains attributable without becoming part of the canonical run package. The detector
configuration may set `regression.debug_artifacts` to `none`, `failures`,
`winner`, or `all`; the command-line `--debug-artifacts` option overrides it.
`failures` writes failed pages from the winning parameter set and is the default.
Each page directory contains the original image, detector input mask, bounding
box overlay, and complete JSON diagnostics.

Regression adapters execute detectors through the authoritative geometry
registry, so serialized candidates include detector name, origin, foundation,
authors, version, and repository provenance.

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

Effect-size regression strategies resolve prior intelligence through the index using the
Golden Set SHA-256, detector ID, and detector-configuration SHA-256. If no compatible record
exists, the existing strategy fallback resolves to exhaustive.
