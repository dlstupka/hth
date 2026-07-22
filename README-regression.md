# Detector Regression

HTH regression is a black-box experiment framework. It knows parameter names and values, invokes a detector adapter, and measures results against the approved Golden Set. It does not interpret detector semantics.

## Run

```bash
python -m hth.regress_detector \
  --detector-config config/detectors/grabcut.json \
  --golden-set config/golden_set.json \
  --image-root path/to/preprocessed \
  --output build/regression \
  --strategy binary-refine
```

For a development smoke test add `--limit 2` with the exhaustive strategy.

## Canonical run directory

```text
build/regression/grabcut/run-YYYYMMDD-HHMMSS/
  manifest.json
  RUN-INFO.json
  parameters.json
  raw/results.csv
  reports/summary.json
  reports/rankings.csv
  reports/top20.csv
  logs/
```

`raw/results.csv` is canonical: one row per parameter set × Golden Set page. Reports are derived and may be regenerated without rerunning the detector. The detector root also receives `grabcut-regression-results.csv` as a convenience copy of the latest full ranking.

`RUN-INFO.json` records the Python, OpenCV, platform and Git commit available to the runner. `manifest.json` begins in `running` state and ends as `complete` or `failed`.

## Strategies

- `exhaustive`: authoritative Cartesian product of configured values.
- `binary-refine`: black-box interval refinement followed by a local Cartesian pass.

The `baseline` profile is treated as a named production reference, not a privileged optimization rule.

## GitHub Actions

The manually dispatched **HTH detector regression** workflow checks out a results repository, runs the selected detector against the Golden Set, uploads the complete canonical run directory, and writes a **Regression Manifest** to the Actions job summary. The manifest records provenance, Golden Set pages, parameter-space size, winner and baseline metrics, and output validation.

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
