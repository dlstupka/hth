# HTH Stage 2 Physical-Geometry Evaluation

This replaces the original all-or-nothing golden-set validator.

## Grades

**PASS**
- IoU >= 0.95
- Maximum absolute edge error <= 20 px

**NEAR_PASS**
- IoU >= 0.90
- Maximum normalized edge error <= 6% of the approved document width/height

**FAIL**
- Wrong object, no valid prediction, or outside the near-pass envelope

**SKIP**
- No approved reference box yet

## Config

Add these fields under `acceptance` in `config/golden_set.json`:

```json
{
  "minimum_iou": 0.95,
  "maximum_edge_error_px": 20,
  "near_pass_minimum_iou": 0.90,
  "near_pass_maximum_edge_error_ratio": 0.06,
  "require_layout_match": false
}
```

Layout matching is intentionally not evaluated here. That belongs in a separate Stage 2B layout-classification validator.

## Workflow

```yaml
- name: Validate Stage 2 physical geometry
  shell: bash
  run: |
    set -euo pipefail

    python hth-pipeline/hth/validate_physical_geometry.py \
      --reference hth-pipeline/config/golden_set.json \
      --analysis "$OUTPUT_DIRECTORY/page-analysis/page-analysis.json" \
      --output "$OUTPUT_DIRECTORY/page-analysis/physical-geometry-report.json" \
      --fail-on never
```

Exit policies:

- `--fail-on never`: always publish the report
- `--fail-on fail`: hard failures stop CI; near-passes do not
- `--fail-on near-pass`: both hard failures and near-passes stop CI

During v0.6 calibration, use `--fail-on never`. Once the detector is redesigned, move to `--fail-on fail`.

Add to output verification:

```bash
test -f "$OUTPUT_DIRECTORY/page-analysis/physical-geometry-report.json"
```

## Metrics emitted

Per page:

- IoU
- signed edge deltas
- normalized edge errors
- maximum pixel and normalized error
- width, height, and area deltas
- Stage 2 quality status and score

Collection summary:

- mean and median IoU
- mean and median maximum edge error
- mean normalized edge error
- mean signed correction by edge
- mean absolute correction by edge
