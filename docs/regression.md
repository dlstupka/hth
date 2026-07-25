# Detector Regression

HTH regression is a black-box experiment framework. It knows parameter names and values, invokes a detector adapter, and measures results against the approved Golden Set. It does not interpret detector semantics.

## Run

```bash
python -m hth.regress_detector \
  --detector-config config/detectors/ransac.json  # or components.json / contour.json / grabcut.json / hough.json / lsd.json \
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

For a manual run, the **Algorithm** input is a choice of `all`, `contour`, `components`, `ransac`, `grabcut`, `hough`, or `lsd`. Automatic smoke runs continue to exercise all configured detector algorithms.

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
