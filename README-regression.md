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

Manual dispatch supports two execution targets through the same job and the same regression command:

- `github-hosted` (default): `ubuntu-latest`
- `self-hosted-hth`: `[self-hosted, linux, x64, hth]`

Push and pull-request smoke tests remain on GitHub-hosted runners. A local runner therefore cannot unexpectedly claim ordinary CI work; it is selected only during manual dispatch.

The default image root is `test/latest/preprocessed` in the results repository. Override it at dispatch time when running against another published build.
