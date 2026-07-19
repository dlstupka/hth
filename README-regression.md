# HTH Detector Regression — GrabCut Draft

This draft adds a black-box regression runner and the first detector parameter space.
The runner does not know what GrabCut parameters mean. It supplies declared values,
runs the detector against every approved Golden Set page, and ranks the measured results.

## Files

```text
config/detectors/grabcut.json
hth/regress_detector.py
hth/geometry/detector_grabcut.py
```

The current production settings are named `baseline`. Calling the detector without a
parameter mapping preserves the v0.6.1 behavior.

## Exhaustive Golden Set regression

From the repository root:

```bash
python hth/regress_detector.py \
  --detector-config config/detectors/grabcut.json \
  --golden-set config/golden_set.json \
  --image-root /path/to/results/test/latest \
  --output build/regression/grabcut-exhaustive
```

`--image-root` may point either to a directory containing `raw/fs_0005.png` or directly
to a directory containing the `fs_*.png` files.

The initial GrabCut search space contains 13,122 combinations. This is deliberately a
full Cartesian product. For a smoke test, add `--limit 10`.

## Expedited interval-halving regression

```bash
python hth/regress_detector.py \
  --detector-config config/detectors/grabcut.json \
  --golden-set config/golden_set.json \
  --image-root /path/to/results/test/latest \
  --output build/regression/grabcut-binary \
  --strategy binary-refine
```

`binary-refine` treats each parameter as an ordered black-box value list. It performs
interval-halving coordinate passes, then exhaustively tests the local Cartesian
neighborhood. It is exploratory; the exhaustive strategy remains authoritative.

## Outputs

```text
regression-results.json
regression-ranking.csv
```

Each parameter set retains:

- complete parameter JSON and stable parameter-set ID;
- overall rank;
- mean and minimum IoU;
- mean edge error;
- failure count and runtime;
- per-page approved and predicted boxes;
- complete detector candidate diagnostics.

The report identifies both the statistical winner and the `baseline` rank/results.
